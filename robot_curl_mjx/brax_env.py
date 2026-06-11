"""Brax-compatible MJX environment factory for robot curl PPO training.

The module is intentionally light at import time. JAX, Brax, and MJX are loaded
inside ``make_brax_env`` so command-line help and local tests do not initialize
the GPU runtime.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np

from robot_curl.task_config import CurlTaskConfig, JOINT_NAMES, N_JOINTS


_REPO_ROOT = Path(__file__).resolve().parents[1]
_XML_PATH = _REPO_ROOT / "assets" / "quadruped.xml"


def _load_training_deps():
    try:
        import jax
        import jax.numpy as jp
        import mujoco
        from brax.envs.base import Env, State
        from mujoco import mjx
    except ImportError as exc:
        raise RuntimeError(
            "MJX/Brax training dependencies are missing. Activate the mjx312 "
            "conda environment or install jax, mujoco, mujoco-mjx, and brax."
        ) from exc
    return jax, jp, mujoco, mjx, Env, State


def make_brax_env(config=None, seed=0, settle_steps=0):
    """Creates a Brax PPO environment backed by MuJoCo MJX."""

    jax, jp, mujoco, mjx, Env, State = _load_training_deps()
    task_config = config or CurlTaskConfig()
    task_config = replace(task_config, action_repeat=max(1, int(task_config.action_repeat)))

    class RobotCurlBraxEnv(Env):
        def __init__(self):
            self.config = task_config
            self.seed = seed
            self.settle_steps = settle_steps
            self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
            self.cpu_data = mujoco.MjData(self.model)
            self.mjx_model = mjx.put_model(self.model)
            self.base_data = mjx.put_data(self.model, self.cpu_data)

            qpos_addr = []
            dof_addr = []
            jnt_low = []
            jnt_high = []
            for name in JOINT_NAMES:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qpos_addr.append(int(self.model.jnt_qposadr[jid]))
                dof_addr.append(int(self.model.jnt_dofadr[jid]))
                jnt_low.append(float(self.model.jnt_range[jid][0]))
                jnt_high.append(float(self.model.jnt_range[jid][1]))

            self.qpos_indices = jp.array(qpos_addr)
            self.dof_indices = jp.array(dof_addr)
            self.jnt_low = jp.array(jnt_low)
            self.jnt_high = jp.array(jnt_high)
            self.torso_qpos_index = qpos_addr[0]
            self.torso_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_front"))
            self.foot_geom_ids = jp.array(
                [
                    int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg}_lower"))
                    for leg in ["fl", "fr", "hl", "hr"]
                ]
            )
            self.stand_pose = jp.array(
                [0.0, -0.4, 0.2, -0.4, 0.2, 0.4, -0.2, 0.4, -0.2],
                dtype=jp.float32,
            )
            self.leg_fold_pose = jp.array(
                [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
                dtype=jp.float32,
            )
            self.curl_tiers = jp.array(self.config.curl_tiers, dtype=jp.float32)

        @property
        def observation_size(self):
            return N_JOINTS * 2 + 4 + 3 + 4

        @property
        def action_size(self):
            return N_JOINTS

        @property
        def backend(self):
            return "mjx"

        def reset(self, rng):
            rng = jax.random.fold_in(rng, self.seed)
            noise_key, height_key = jax.random.split(rng)
            noise = jax.random.uniform(noise_key, shape=(N_JOINTS,), minval=-0.05, maxval=0.05)
            init_q = jp.clip(self.stand_pose + noise, self.jnt_low, self.jnt_high)
            height = jax.random.uniform(height_key, minval=0.38, maxval=0.45)

            qpos = self.base_data.qpos.at[self.qpos_indices].set(init_q)
            qpos = qpos.at[2].set(height)
            qvel = jp.zeros_like(self.base_data.qvel)
            data = self.base_data.replace(qpos=qpos, qvel=qvel, ctrl=init_q)
            data = mjx.forward(self.mjx_model, data)

            def settle_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            if self.settle_steps > 0:
                data = jax.lax.scan(settle_step, data, (), length=self.settle_steps)[0]

            curl = self._curl_amount(data)
            obs = self._obs(data)
            info = {
                "target_q": init_q,
                "init_curl": curl,
                "best_curl": curl,
                "step_count": jp.array(0, dtype=jp.int32),
            }
            reward_metrics = {
                "reward": jp.array(0.0),
                "reward_total": jp.array(0.0),
                "reward_curl": jp.array(0.0),
                "reward_progress": jp.array(0.0),
                "reward_tier": jp.array(0.0),
                "reward_contact": jp.array(0.0),
                "reward_leg_fold": jp.array(0.0),
                "reward_smooth": jp.array(0.0),
                "reward_stable": jp.array(0.0),
                "reward_upright": jp.array(0.0),
                "reward_overcurl": jp.array(0.0),
                "reward_alive": jp.array(0.0),
            }
            return State(data, obs, jp.array(0.0), jp.array(0.0), metrics=reward_metrics, info=info)

        def step(self, state, action):
            action = jp.clip(action, -self.config.action_scale, self.config.action_scale)
            target_q = jp.clip(state.info["target_q"] + action, self.jnt_low, self.jnt_high)
            data = state.pipeline_state.replace(ctrl=target_q)

            def physics_step(carry, _):
                return mjx.step(self.mjx_model, carry), None

            data = jax.lax.scan(physics_step, data, (), length=self.config.action_repeat)[0]
            step_count = state.info["step_count"] + 1
            curl = self._curl_amount(data)
            best_curl = jp.maximum(state.info["best_curl"], curl)
            contacts = self._foot_contacts(data)
            contact_count = jp.sum(contacts)
            upright = self._upright(data)
            torso_vel = data.cvel[self.torso_id][3:6]
            torso_angvel = data.cvel[self.torso_id][0:3]
            reward, reward_metrics = self._reward(
                curl,
                state.info["init_curl"],
                state.info["best_curl"],
                best_curl,
                contact_count,
                action,
                torso_vel,
                torso_angvel,
                upright,
                data.qpos[self.qpos_indices][1:],
            )
            done = jp.where(
                (upright < self.config.terminate_upright)
                | (data.xpos[self.torso_id][2] < self.config.terminate_height)
                | (step_count >= self.config.max_episode_steps),
                1.0,
                0.0,
            )
            info = {
                **state.info,
                "target_q": target_q,
                "best_curl": best_curl,
                "step_count": step_count,
            }
            return State(data, self._obs(data), reward, done, metrics=reward_metrics, info=info)

        def _obs(self, data):
            qpos = data.qpos[self.qpos_indices]
            qvel = data.qvel[self.dof_indices]
            torso_quat = data.xquat[self.torso_id]
            torso_vel = data.cvel[self.torso_id][3:6]
            return jp.concatenate([qpos, qvel, torso_quat, torso_vel, self._foot_contacts(data)])

        def _curl_amount(self, data):
            return jp.maximum(0.0, -data.qpos[self.torso_qpos_index])

        def _upright(self, data):
            torso_quat = data.xquat[self.torso_id]
            return 1.0 - 2.0 * (torso_quat[1] ** 2 + torso_quat[2] ** 2)

        def _foot_contacts(self, data):
            contact = data.contact
            if hasattr(contact, "geom1") and hasattr(contact, "geom2"):
                geom1 = contact.geom1
                geom2 = contact.geom2
            else:
                geom1 = contact.geom[:, 0]
                geom2 = contact.geom[:, 1]
            valid = (contact.dist <= 0.0) & (geom1 >= 0) & (geom2 >= 0)
            touched = (geom1[:, None] == self.foot_geom_ids[None, :]) | (geom2[:, None] == self.foot_geom_ids[None, :])
            return jp.any(valid[:, None] & touched, axis=0).astype(jp.float32)

        def _reward(
            self,
            curl,
            init_curl,
            previous_best_curl,
            best_curl,
            contact_count,
            action,
            torso_vel,
            torso_angvel,
            upright,
            leg_qpos,
        ):
            cfg = self.config
            effective_curl = jp.minimum(curl, cfg.curl_goal)
            curl_progress = jp.maximum(0.0, effective_curl - init_curl)
            overcurl = jp.maximum(0.0, curl - cfg.curl_goal)
            r_curl = cfg.reward_curl * effective_curl / jp.maximum(cfg.curl_goal, 1e-6)
            r_progress = cfg.reward_progress * curl_progress
            r_tier = cfg.reward_tier * jp.sum((curl > self.curl_tiers) & (previous_best_curl <= self.curl_tiers))
            r_contact = cfg.reward_contact * jp.minimum(contact_count, cfg.contact_cap)
            r_contact -= cfg.reward_low_contact * jp.maximum(0.0, cfg.min_contacts - contact_count)
            r_smooth = -jp.mean(jp.square(action))
            r_stable = -0.5 * jp.sum(jp.square(torso_vel)) - 0.2 * jp.sum(jp.square(torso_angvel))
            r_upright = -cfg.reward_upright * jp.maximum(0.0, cfg.upright_threshold - upright)
            leg_error = jp.mean(jp.square(leg_qpos - self.leg_fold_pose))
            r_leg_fold = cfg.reward_leg_fold * jp.exp(-4.0 * leg_error)
            reward_metrics = {
                "reward_curl": r_curl,
                "reward_progress": r_progress,
                "reward_tier": r_tier,
                "reward_contact": r_contact,
                "reward_leg_fold": r_leg_fold,
                "reward_smooth": cfg.reward_smooth * r_smooth,
                "reward_stable": cfg.reward_stable * r_stable,
                "reward_upright": r_upright,
                "reward_overcurl": -cfg.penalty_overcurl * overcurl,
                "reward_alive": jp.asarray(cfg.reward_alive),
            }
            reward = (
                r_curl
                + r_progress
                + r_tier
                + r_contact
                + r_leg_fold
                + cfg.reward_smooth * r_smooth
                + cfg.reward_stable * r_stable
                + r_upright
                + cfg.reward_alive
                - cfg.penalty_overcurl * overcurl
            )
            return reward, {"reward": reward, "reward_total": reward, **reward_metrics}

    # Force arrays to be JAX arrays during construction, but keep the class lazy
    # until make_brax_env is called from the training entrypoint.
    np.asarray([0.0])
    return RobotCurlBraxEnv()
