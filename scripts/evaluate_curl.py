"""Headless evaluation metrics for the robot-curl policy."""
import argparse
from pathlib import Path

import numpy as np

from robot_curl.env import QuadrupedFoldEnv
from robot_curl.policy_search import (
    CurlPolicyParams,
    FeedbackPolicyParams,
    make_action,
    make_closed_loop_action,
    make_feedback_action,
)


SCRIPTED_PARAMS = CurlPolicyParams(
    torso=-0.03,
    front_hip=-0.015,
    front_knee=0.025,
    hind_hip=-0.025,
    hind_knee=0.0,
    switch_step=40,
    release_step=160,
)

CLOSED_LOOP_PARAMS = CurlPolicyParams(
    torso=-0.02,
    front_hip=-0.02,
    front_knee=0.04,
    hind_hip=-0.02,
    hind_knee=0.0,
    switch_step=40,
    release_step=160,
)

CEM_PARAMS = CurlPolicyParams(
    torso=-0.0376,
    front_hip=0.0114,
    front_knee=0.0271,
    hind_hip=-0.0264,
    hind_knee=0.0190,
    switch_step=48,
    release_step=168,
)

FEEDBACK_CEM_PARAMS = FeedbackPolicyParams(
    torso_gain=-0.0570,
    front_hip_gain=-0.0154,
    front_knee_gain=0.0047,
    hind_hip_gain=-0.0447,
    hind_knee_gain=0.0064,
    phase_split=0.200,
    min_contacts=1.00,
)


def _load_policy(model_path, norm_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    env = DummyVecEnv([lambda: QuadrupedFoldEnv()])
    if norm_path and norm_path.exists():
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    return PPO.load(model_path), env


def _torso_up(env):
    quat = env.data.xquat[env.torso_id]
    return float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))


def evaluate_open_loop(policy_name, episodes):
    env = QuadrupedFoldEnv()
    rows = []
    for ep in range(episodes):
        env.reset(seed=ep)
        total_reward = 0.0
        max_curl = 0.0
        min_up = 1.0
        contacts = []
        terminated = truncated = False
        for step in range(env.max_episode_steps):
            if policy_name == "random":
                action = env.action_space.sample()
            elif policy_name == "scripted":
                action = make_action(env, SCRIPTED_PARAMS, step)
            elif policy_name == "closed_loop":
                action = make_closed_loop_action(env, CLOSED_LOOP_PARAMS, step)
            elif policy_name == "cem":
                action = make_closed_loop_action(env, CEM_PARAMS, step)
            elif policy_name == "feedback_cem":
                action = make_feedback_action(env, FEEDBACK_CEM_PARAMS)
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            max_curl = max(max_curl, env._curl_amount())
            min_up = min(min_up, _torso_up(env))
            contacts.append(float(np.sum(env._foot_contacts())))
            if terminated or truncated:
                break
        rows.append((ep, step + 1, total_reward, max_curl, env.data.xpos[env.torso_id][2], min_up, np.mean(contacts), terminated))
    return rows


def evaluate_model(model_path, norm_path, episodes):
    model, vec_env = _load_policy(model_path, norm_path)
    base_env = vec_env.envs[0]
    rows = []
    for ep in range(episodes):
        obs = vec_env.reset()
        total_reward = 0.0
        max_curl = 0.0
        min_up = 1.0
        contacts = []
        done = [False]
        for step in range(base_env.max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec_env.step(action)
            total_reward += float(reward[0])
            max_curl = max(max_curl, base_env._curl_amount())
            min_up = min(min_up, _torso_up(base_env))
            contacts.append(float(np.sum(base_env._foot_contacts())))
            if done[0]:
                break
        rows.append((ep, step + 1, total_reward, max_curl, base_env.data.xpos[base_env.torso_id][2], min_up, np.mean(contacts), bool(done[0])))
    return rows


def print_rows(rows):
    print("ep,steps,total_reward,max_curl,final_z,min_up,mean_contacts,done")
    for row in rows:
        print(f"{row[0]},{row[1]},{row[2]:.3f},{row[3]:.3f},{row[4]:.3f},{row[5]:.3f},{row[6]:.3f},{row[7]}")
    arr = np.array([[r[1], r[2], r[3], r[4], r[5], r[6], float(r[7])] for r in rows], dtype=float)
    print(
        "mean,"
        f"{arr[:,0].mean():.1f},{arr[:,1].mean():.3f},{arr[:,2].mean():.3f},"
        f"{arr[:,3].mean():.3f},{arr[:,4].mean():.3f},{arr[:,5].mean():.3f},{arr[:,6].mean():.3f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=["zero", "random", "scripted", "closed_loop", "cem", "feedback_cem", "model"],
        default="zero",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--norm", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    if args.policy == "model":
        if args.model is None:
            raise SystemExit("--model is required for --policy model")
        rows = evaluate_model(args.model, args.norm, args.episodes)
    else:
        rows = evaluate_open_loop(args.policy, args.episodes)
    print_rows(rows)


if __name__ == "__main__":
    main()
