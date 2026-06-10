"""四足折叠环境：Gymnasium Env，PPO 训练用。"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import mujoco
from gymnasium import Env
from gymnasium.spaces import Box

# 路径
_REPO_ROOT = Path(__file__).resolve().parents[1]
_XML_PATH = _REPO_ROOT / "assets" / "quadruped.xml"

# 关节名列表（顺序固定）
JOINT_NAMES = [
    "torso_hinge",
    "fl_hip_flex", "fl_knee",
    "fr_hip_flex", "fr_knee",
    "hl_hip_flex", "hl_knee",
    "hr_hip_flex", "hr_knee",
]
N_JOINTS = len(JOINT_NAMES)

# 关节限位（通过 model 读取）
@dataclass(frozen=True)
class CurlTaskConfig:
    curl_goal: float = 0.45
    max_episode_steps: int = 250
    action_repeat: int = 20
    action_scale: float = 0.1
    reward_curl: float = 3.0
    reward_progress: float = 2.0
    reward_tier: float = 0.5
    curl_tiers: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    reward_contact: float = 0.15
    reward_low_contact: float = 0.25
    min_contacts: float = 2.0
    contact_cap: float = 3.0
    reward_smooth: float = 0.03
    reward_stable: float = 0.02
    reward_upright: float = 4.0
    upright_threshold: float = 0.9
    reward_alive: float = 0.05
    penalty_overcurl: float = 10.0
    terminate_upright: float = 0.3
    terminate_height: float = 0.05


def _get_joint_limits(model):
    low, high = [], []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low.append(model.jnt_range[jid][0])
        high.append(model.jnt_range[jid][1])
    return np.array(low), np.array(high)


class QuadrupedFoldEnv(Env):
    """四足机器人从站立折叠成球的强化学习环境。

    Observation (29,):
        [0:9]   关节角度 (rad)
        [9:18]  关节角速度 (rad/s)
        [18:22] 躯干姿态四元数 (w,x,y,z)
        [22:25] 躯干线速度 (m/s)
        [25:29] 四足触地标志 (0/1)

    Action (9,):
        每个关节的目标位置偏移 (rad), 范围 [-0.1, 0.1]
        实际目标 = clip(当前目标 + action, joint_limit_low, joint_limit_high)
    """

    def __init__(self, render_mode=None, config=None):
        super().__init__()
        self.config = config or CurlTaskConfig()

        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep  # 物理步长

        # 控制频率：每 N 个物理步发一次动作
        self.action_repeat = self.config.action_repeat  # 20 * 0.001s = 0.02s, 50Hz
        self.max_episode_steps = self.config.max_episode_steps  # 250 * 0.02s = 5s

        # 关节限位
        self.jnt_low, self.jnt_high = _get_joint_limits(self.model)

        # 关节 dof 地址
        self.dof_addr = {}
        self.qpos_addr = {}
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.dof_addr[name] = self.model.jnt_dofadr[jid]
            self.qpos_addr[name] = self.model.jnt_qposadr[jid]

        # 躯干 body id
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_front")

        # 脚 geom id（用于触地检测）
        self.foot_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_lower")
            for leg in ["fl", "fr", "hl", "hr"]
        ]

        # 站立和折叠目标
        self.stand_pose = np.array([
            0.0,                        # torso_hinge
            -0.4,  0.2,                 # fl
            -0.4,  0.2,                 # fr
             0.4, -0.2,                 # hl
             0.4, -0.2,                 # hr
        ])
        self.fold_pose = np.array([
             1.0,                       # torso_hinge
            -1.0,  1.0,                 # fl
            -1.0,  1.0,                 # fr
             1.0, -1.0,                 # hl
             1.0, -1.0,                 # hr
        ])

        # 观测/动作空间
        obs_dim = N_JOINTS * 2 + 4 + 3 + 4  # 9+9+4+3+4 = 29
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = Box(
            low=-self.config.action_scale,
            high=self.config.action_scale,
            shape=(N_JOINTS,),
            dtype=np.float32,
        )

        # PD 参数
        self.Kp = np.array([
            100.0,              # torso_hinge
            80.0, 60.0,         # fl: flex, knee
            80.0, 60.0,         # fr
            80.0, 60.0,         # hl
            80.0, 60.0,         # hr
        ])
        self.Kd = np.array([
            2.0,                # torso_hinge
            1.5, 1.0,           # fl
            1.5, 1.0,           # fr
            1.5, 1.0,           # hl
            1.5, 1.0,           # hr
        ])
        self.torque_limits = np.array([
            25.0,               # torso_hinge
            16.0, 12.0,          # fl
            16.0, 12.0,          # fr
            16.0, 12.0,          # hl
            16.0, 12.0,          # hr
        ])

        # 当前目标位置（在 reset/step 中更新）
        self.target_q = self.stand_pose.copy()
        self.curl_goal = self.config.curl_goal

        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 重置物理
        mujoco.mj_resetData(self.model, self.data)

        # 初始角度：站姿 + 小噪声
        noise = self.np_random.uniform(-0.05, 0.05, size=N_JOINTS)
        init_q = self.stand_pose + noise
        init_q = np.clip(init_q, self.jnt_low, self.jnt_high)
        for i, name in enumerate(JOINT_NAMES):
            self.data.qpos[self.qpos_addr[name]] = init_q[i]

        # 初始高度
        self.data.qpos[2] = self.np_random.uniform(0.38, 0.45)

        # 目标 = 当前初始角度
        self.target_q = init_q.copy()
        self.step_count = 0

        # 记录初始形状误差，用于进度奖励
        q_init = np.array([self.data.qpos[self.qpos_addr[n]] for n in JOINT_NAMES])
        self.init_curl = self._curl_amount()
        self.best_curl = self.init_curl

        # 跑几步让物理稳定
        for _ in range(50):
            self._apply_pd()
            mujoco.mj_step(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        # 更新目标：累加偏移，限幅
        self.target_q = np.clip(self.target_q + action, self.jnt_low, self.jnt_high)

        # 执行 N 个物理步
        for _ in range(self.action_repeat):
            self._apply_pd()
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def _apply_pd(self):
        """PD 控制：把每个关节拉向 self.target_q。"""
        for i, name in enumerate(JOINT_NAMES):
            q = self.data.qpos[self.qpos_addr[name]]
            qvel = self.data.qvel[self.dof_addr[name]]
            error = self.target_q[i] - q
            torque = self.Kp[i] * error - self.Kd[i] * qvel
            torque = np.clip(torque, -self.torque_limits[i], self.torque_limits[i])
            self.data.qfrc_applied[self.dof_addr[name]] = torque

    def _get_obs(self):
        qpos = np.array([self.data.qpos[self.qpos_addr[n]] for n in JOINT_NAMES], dtype=np.float32)
        qvel = np.array([self.data.qvel[self.dof_addr[n]] for n in JOINT_NAMES], dtype=np.float32)

        torso_quat = self.data.xquat[self.torso_id].copy()  # w,x,y,z
        torso_vel = self.data.cvel[self.torso_id][3:6].copy()
        foot_contact = self._foot_contacts()

        return np.concatenate([qpos, qvel, torso_quat, torso_vel, foot_contact])

    def _foot_contacts(self):
        foot_contact = np.zeros(4, dtype=np.float32)
        for j, geom_id in enumerate(self.foot_geom_ids):
            for k in range(self.data.ncon):
                contact = self.data.contact[k]
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    foot_contact[j] = 1.0
                    break
        return foot_contact

    def _curl_amount(self):
        torso_angle = self.data.qpos[self.qpos_addr["torso_hinge"]]
        return max(0.0, float(torso_angle))

    def _compute_reward(self, action):
        cfg = self.config
        curl = self._curl_amount()
        effective_curl = min(curl, self.curl_goal)
        curl_progress = max(0.0, effective_curl - self.init_curl)
        overcurl = max(0.0, curl - self.curl_goal)
        r_curl = cfg.reward_curl * min(curl, self.curl_goal) / max(self.curl_goal, 1e-6)
        r_progress = cfg.reward_progress * curl_progress

        r_tier = 0.0
        for threshold in cfg.curl_tiers:
            if curl > threshold and self.best_curl <= threshold:
                r_tier += cfg.reward_tier
        self.best_curl = max(self.best_curl, curl)

        r_smooth = -np.mean(np.square(action))

        torso_vel = self.data.cvel[self.torso_id][3:6]
        torso_angvel = self.data.cvel[self.torso_id][0:3]
        r_stable = -0.5 * np.sum(np.square(torso_vel)) - 0.2 * np.sum(np.square(torso_angvel))

        torso_quat = self.data.xquat[self.torso_id]
        z_proj = 1.0 - 2.0 * (torso_quat[1]**2 + torso_quat[2]**2)
        r_upright = -cfg.reward_upright * max(0, cfg.upright_threshold - z_proj)

        foot_contacts = self._foot_contacts()
        contact_count = float(np.sum(foot_contacts))
        r_contact = cfg.reward_contact * min(contact_count, cfg.contact_cap) - cfg.reward_low_contact * max(0.0, cfg.min_contacts - contact_count)

        r_alive = cfg.reward_alive

        reward = (
            r_curl
            + r_progress
            + r_tier
            + r_contact
            + cfg.reward_smooth * r_smooth
            + cfg.reward_stable * r_stable
            + r_upright
            + r_alive
            - cfg.penalty_overcurl * overcurl
        )
        return reward

    def _is_terminated(self):
        torso_quat = self.data.xquat[self.torso_id]
        z_proj = 1.0 - 2.0 * (torso_quat[1]**2 + torso_quat[2]**2)
        if z_proj < self.config.terminate_upright:
            return True
        if self.data.xpos[self.torso_id][2] < self.config.terminate_height:
            return True
        return False


# 快速验证
if __name__ == "__main__":
    env = QuadrupedFoldEnv()
    obs, _ = env.reset()
    print(f"obs dim: {obs.shape}, action dim: {env.action_space.shape}")
    total_reward = 0.0
    for _ in range(250):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"random policy total reward: {total_reward:.2f}")
