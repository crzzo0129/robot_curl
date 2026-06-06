"""四足折叠环境 — MJX GPU 加速版。"""
import os
import numpy as np
import mujoco
from mujoco import mjx
from gymnasium import Env
from gymnasium.spaces import Box

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_XML_PATH = os.path.join(_SCRIPT_DIR, "quadruped.xml")

JOINT_NAMES = [
    "torso_hinge",
    "fl_hip_abd", "fl_hip_flex", "fl_knee",
    "fr_hip_abd", "fr_hip_flex", "fr_knee",
    "hl_hip_abd", "hl_hip_flex", "hl_knee",
    "hr_hip_abd", "hr_hip_flex", "hr_knee",
]
N_JOINTS = len(JOINT_NAMES)


def _get_joint_limits(model):
    low, high = [], []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low.append(model.jnt_range[jid][0])
        high.append(model.jnt_range[jid][1])
    return np.array(low), np.array(high)


class QuadrupedFoldEnvMJX(Env):
    """MJX 加速版：物理在 GPU 上跑，观测/奖励在 CPU 上算。"""

    def __init__(self):
        super().__init__()

        # CPU 端模型（用于读取几何/索引信息）
        self.model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # MJX 模型和数据（GPU 端）
        self.mjx_model = mjx.put_model(self.model)
        self.mjx_data = mjx.put_data(self.model, self.data)

        self.action_repeat = 20
        self.max_episode_steps = 250

        self.jnt_low, self.jnt_high = _get_joint_limits(self.model)

        self.dof_addr = {}
        self.qpos_addr = {}
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.dof_addr[name] = self.model.jnt_dofadr[jid]
            self.qpos_addr[name] = self.model.jnt_qposadr[jid]

        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_front")

        # 目标姿态
        self.stand_pose = np.array([
            0.0, 0.0, -0.4, 0.2, 0.0, -0.4, 0.2,
            0.0, 0.4, -0.2, 0.0, 0.4, -0.2,
        ])
        self.fold_pose = np.array([
            -1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0,
            0.0, 1.0, -1.0, 0.0, 1.0, -1.0,
        ])

        obs_dim = N_JOINTS * 2 + 4 + 3 + 4
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(low=-0.1, high=0.1, shape=(N_JOINTS,), dtype=np.float32)

        self.Kp = np.array([
            100.0, 60.0, 80.0, 60.0, 60.0, 80.0, 60.0,
            60.0, 80.0, 60.0, 60.0, 80.0, 60.0,
        ])
        self.Kd = np.array([
            2.0, 1.0, 1.5, 1.0, 1.0, 1.5, 1.0,
            1.0, 1.5, 1.0, 1.0, 1.5, 1.0,
        ])

        self.target_q = self.stand_pose.copy()
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置 CPU data，再同步到 GPU
        mujoco.mj_resetData(self.model, self.data)

        noise = self.np_random.uniform(-0.05, 0.05, size=N_JOINTS)
        init_q = np.clip(self.stand_pose + noise, self.jnt_low, self.jnt_high)
        for i, name in enumerate(JOINT_NAMES):
            self.data.qpos[self.qpos_addr[name]] = init_q[i]
        self.data.qpos[2] = self.np_random.uniform(0.38, 0.45)

        self.target_q = init_q.copy()
        self.step_count = 0

        # 同步到 MJX
        self.mjx_data = mjx.put_data(self.model, self.data)

        # 稳定几步
        for _ in range(50):
            self._mjx_step()

        # 记录初始形状误差
        q_init = np.array([self.mjx_data.qpos[self.qpos_addr[n]] for n in JOINT_NAMES])
        self.init_shape_error = float(np.mean(np.abs(q_init - self.fold_pose)))
        self.best_shape_error = self.init_shape_error

        return self._get_obs(), {}

    def step(self, action):
        self.target_q = np.clip(self.target_q + action, self.jnt_low, self.jnt_high)

        for _ in range(self.action_repeat):
            self._apply_pd_mjx()
            self._mjx_step()

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def _mjx_step(self):
        """单步 MJX 物理步进。"""
        self.mjx_data = mjx.step(self.mjx_model, self.mjx_data)

    def _apply_pd_mjx(self):
        """PD 控制：直接操作 mjx_data.qfrc_applied。"""
        qfrc = self.mjx_data.qfrc_applied
        for i, name in enumerate(JOINT_NAMES):
            adr = self.dof_addr[name]
            q = self.mjx_data.qpos[self.qpos_addr[name]]
            qvel = self.mjx_data.qvel[adr]
            error = self.target_q[i] - q
            qfrc[adr] = self.Kp[i] * error - self.Kd[i] * qvel

    def _get_obs(self):
        d = self.mjx_data
        qpos = np.array([d.qpos[self.qpos_addr[n]] for n in JOINT_NAMES], dtype=np.float32)
        qvel = np.array([d.qvel[self.dof_addr[n]] for n in JOINT_NAMES], dtype=np.float32)

        torso_quat = d.xquat[self.torso_id].copy()
        torso_vel = d.cvel[self.torso_id][3:6].copy()

        return np.concatenate([qpos, qvel, torso_quat, torso_vel, np.zeros(4, dtype=np.float32)])

    def _compute_reward(self, action):
        d = self.mjx_data
        q_actual = np.array([d.qpos[self.qpos_addr[n]] for n in JOINT_NAMES])
        shape_error = float(np.mean(np.abs(q_actual - self.fold_pose)))

        r_shape = -shape_error
        r_progress = (self.init_shape_error - shape_error) * 3.0

        r_tier = 0.0
        for threshold in [0.25, 0.20, 0.15, 0.10, 0.05]:
            if shape_error < threshold and self.best_shape_error >= threshold:
                r_tier += 1.0
        self.best_shape_error = min(self.best_shape_error, shape_error)

        r_smooth = -np.mean(np.square(action))
        torso_vel = d.cvel[self.torso_id][3:6]
        torso_angvel = d.cvel[self.torso_id][0:3]
        r_stable = -np.sum(np.square(torso_vel)) - np.sum(np.square(torso_angvel))

        # 倾斜惩罚
        torso_quat = d.xquat[self.torso_id]
        z_proj = 1.0 - 2.0 * (torso_quat[1]**2 + torso_quat[2]**2)
        r_upright = -2.0 * max(0, 0.9 - z_proj)

        # 存活奖励
        r_alive = 0.05

        return float(r_shape + r_progress + r_tier + 0.02 * r_smooth + 0.01 * r_stable + r_upright + r_alive)

    def _is_terminated(self):
        d = self.mjx_data
        torso_quat = d.xquat[self.torso_id]
        z_proj = 1.0 - 2.0 * (torso_quat[1]**2 + torso_quat[2]**2)
        if z_proj < 0.3 or d.xpos[self.torso_id][2] < 0.05:
            return True
        return False
