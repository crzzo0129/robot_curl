"""Minimal MJX environment for robot curl smoke testing.

This backend intentionally stays separate from the Gymnasium/MuJoCo environment
until reset/step/reward parity is verified on the target JAX GPU environment.
"""
from pathlib import Path

import numpy as np

from robot_curl.env import CurlTaskConfig, JOINT_NAMES, N_JOINTS, _get_joint_limits
from robot_curl_mjx.reward import curl_reward_terms


_REPO_ROOT = Path(__file__).resolve().parents[1]
_XML_PATH = _REPO_ROOT / "assets" / "quadruped.xml"


def _load_mjx_deps():
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError(
            "MJX dependencies are missing. Activate the mjx312 conda environment "
            "or install jax, jaxlib, mujoco, and mujoco-mjx."
        ) from exc
    return jax, jp, mujoco, mjx


class QuadrupedCurlMJXEnv:
    """Small stateful MJX backend with the same action/observation intent."""

    def __init__(self, config=None, seed=0, settle_steps=50):
        self.config = config or CurlTaskConfig()
        self.seed = seed
        self.settle_steps = settle_steps
        self.jax, self.jp, self.mujoco, self.mjx = _load_mjx_deps()
        self.model = self.mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self.cpu_data = self.mujoco.MjData(self.model)
        self.mjx_model = self.mjx.put_model(self.model)
        self.key = self.jax.random.PRNGKey(seed)

        self.jnt_low, self.jnt_high = _get_joint_limits(self.model)
        self.qpos_addr = {}
        self.dof_addr = {}
        for name in JOINT_NAMES:
            jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            self.qpos_addr[name] = int(self.model.jnt_qposadr[jid])
            self.dof_addr[name] = int(self.model.jnt_dofadr[jid])

        self.qpos_indices = self.jp.array([self.qpos_addr[name] for name in JOINT_NAMES])
        self.dof_indices = self.jp.array([self.dof_addr[name] for name in JOINT_NAMES])
        self.torso_id = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "torso_front"))
        self.foot_geom_ids = self.jp.array(
            [
                int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_lower"))
                for leg in ["fl", "fr", "hl", "hr"]
            ]
        )

        self.stand_pose_np = np.array(
            [0.0, -0.4, 0.2, -0.4, 0.2, 0.4, -0.2, 0.4, -0.2],
            dtype=np.float32,
        )
        self.stand_pose = self.jp.array(self.stand_pose_np)
        self.jnt_low_jp = self.jp.array(self.jnt_low)
        self.jnt_high_jp = self.jp.array(self.jnt_high)
        self.kp = self.jp.array([100.0, 80.0, 60.0, 80.0, 60.0, 80.0, 60.0, 80.0, 60.0])
        self.kd = self.jp.array([2.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0, 1.5, 1.0])
        self.torque_limits = self.jp.array([25.0, 16.0, 12.0, 16.0, 12.0, 16.0, 12.0, 16.0, 12.0])

        self.target_q = self.stand_pose
        self.data = None
        self.step_count = 0
        self.init_curl = 0.0
        self.best_curl = 0.0

    @property
    def observation_size(self):
        return N_JOINTS * 2 + 4 + 3 + 4

    @property
    def action_size(self):
        return N_JOINTS

    def reset(self):
        self.mujoco.mj_resetData(self.model, self.cpu_data)
        self.key, noise_key, height_key = self.jax.random.split(self.key, 3)
        noise = self.jax.random.uniform(noise_key, shape=(N_JOINTS,), minval=-0.05, maxval=0.05)
        init_q = self.jp.clip(self.stand_pose + noise, self.jnt_low_jp, self.jnt_high_jp)
        height = self.jax.random.uniform(height_key, minval=0.38, maxval=0.45)

        qpos = self.jp.array(self.cpu_data.qpos)
        qpos = qpos.at[self.qpos_indices].set(init_q)
        qpos = qpos.at[2].set(height)
        self.data = self.mjx.put_data(self.model, self.cpu_data).replace(qpos=qpos)
        self.target_q = init_q
        self.step_count = 0

        for _ in range(self.settle_steps):
            self._apply_pd()
            self.data = self.mjx.step(self.mjx_model, self.data)

        self.init_curl = float(self._curl_amount())
        self.best_curl = self.init_curl
        return self.obs()

    def step(self, action):
        action = self.jp.clip(self.jp.asarray(action), -self.config.action_scale, self.config.action_scale)
        self.target_q = self.jp.clip(self.target_q + action, self.jnt_low_jp, self.jnt_high_jp)
        for _ in range(self.config.action_repeat):
            self._apply_pd()
            self.data = self.mjx.step(self.mjx_model, self.data)
        self.step_count += 1
        reward = self.reward(action)
        terminated = self.terminated()
        truncated = self.step_count >= self.config.max_episode_steps
        return self.obs(), reward, terminated, truncated

    def _apply_pd(self):
        q = self.data.qpos[self.qpos_indices]
        qvel = self.data.qvel[self.dof_indices]
        torque = self.kp * (self.target_q - q) - self.kd * qvel
        torque = self.jp.clip(torque, -self.torque_limits, self.torque_limits)
        self.data = self.data.replace(qfrc_applied=self.data.qfrc_applied.at[self.dof_indices].set(torque))

    def obs(self):
        qpos = self.data.qpos[self.qpos_indices]
        qvel = self.data.qvel[self.dof_indices]
        torso_quat = self.data.xquat[self.torso_id]
        torso_vel = self.data.cvel[self.torso_id][3:6]
        return self.jp.concatenate([qpos, qvel, torso_quat, torso_vel, self._foot_contacts()])

    def _curl_amount(self):
        torso_angle = self.data.qpos[self.qpos_addr["torso_hinge"]]
        return self.jp.maximum(0.0, torso_angle)

    def _foot_contacts(self):
        contact = self.data.contact
        if hasattr(contact, "geom1") and hasattr(contact, "geom2"):
            geom1 = contact.geom1
            geom2 = contact.geom2
        elif hasattr(contact, "geom"):
            geom1 = contact.geom[:, 0]
            geom2 = contact.geom[:, 1]
        else:
            return self.jp.zeros(4)
        valid = (contact.dist <= 0.0) & (geom1 >= 0) & (geom2 >= 0)
        contacts = []
        for geom_id in self.foot_geom_ids:
            touched = self.jp.any(valid & ((geom1 == geom_id) | (geom2 == geom_id)))
            contacts.append(touched.astype(float))
        return self.jp.array(contacts)

    def reward(self, action):
        torso_vel = self.data.cvel[self.torso_id][3:6]
        torso_angvel = self.data.cvel[self.torso_id][0:3]
        torso_quat = self.data.xquat[self.torso_id]
        upright = 1.0 - 2.0 * (torso_quat[1] ** 2 + torso_quat[2] ** 2)
        reward, self.best_curl = curl_reward_terms(
            self.config,
            float(self._curl_amount()),
            self.init_curl,
            self.best_curl,
            float(self.jp.sum(self._foot_contacts())),
            float(self.jp.mean(self.jp.square(action))),
            float(self.jp.sum(self.jp.square(torso_vel))),
            float(self.jp.sum(self.jp.square(torso_angvel))),
            float(upright),
        )
        return float(reward)

    def terminated(self):
        torso_quat = self.data.xquat[self.torso_id]
        upright = 1.0 - 2.0 * (torso_quat[1] ** 2 + torso_quat[2] ** 2)
        low_height = self.data.xpos[self.torso_id][2] < self.config.terminate_height
        return bool((upright < self.config.terminate_upright) | low_height)
