"""Grid-search simple curl policies without RL dependencies.

This is a diagnostic bridge: it quantifies whether foot-adjustment action
patterns can outperform pure standing before spending time on PPO training.
"""
import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from env import QuadrupedFoldEnv


@dataclass(frozen=True)
class CurlPolicyParams:
    torso: float
    front_hip: float
    front_knee: float
    hind_hip: float
    hind_knee: float
    switch_step: int
    release_step: int


def make_action(env, params, step):
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[0] = params.torso
    if step < params.release_step:
        sign = 1.0 if step < params.switch_step else -0.5
        action[2] = sign * params.front_hip
        action[5] = sign * params.front_hip
        action[3] = sign * params.front_knee
        action[6] = sign * params.front_knee
        action[8] = sign * params.hind_hip
        action[11] = sign * params.hind_hip
        action[9] = sign * params.hind_knee
        action[12] = sign * params.hind_knee
    return np.clip(action, env.action_space.low, env.action_space.high)


def make_closed_loop_action(env, params, step):
    action = make_action(env, params, step)
    curl = env._curl_amount()
    contact_count = float(np.sum(env._foot_contacts()))
    if curl >= env.curl_goal or contact_count < 2.0:
        action[0] = 0.0
    elif curl > 0.75 * env.curl_goal:
        action[0] = 0.5 * params.torso
    return np.clip(action, env.action_space.low, env.action_space.high)


def _torso_up(env):
    quat = env.data.xquat[env.torso_id]
    return float(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2))


def evaluate_params(params, episodes, closed_loop=False):
    rows = []
    for seed in range(episodes):
        env = QuadrupedFoldEnv()
        env.reset(seed=seed)
        total_reward = 0.0
        max_curl = 0.0
        min_up = 1.0
        contacts = []
        done = False
        for step in range(env.max_episode_steps):
            if closed_loop:
                action = make_closed_loop_action(env, params, step)
            else:
                action = make_action(env, params, step)
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            max_curl = max(max_curl, env._curl_amount())
            min_up = min(min_up, _torso_up(env))
            contacts.append(float(np.sum(env._foot_contacts())))
            if terminated or truncated:
                done = bool(terminated)
                break
        rows.append(
            {
                "steps": step + 1,
                "total_reward": total_reward,
                "max_curl": max_curl,
                "final_z": float(env.data.xpos[env.torso_id][2]),
                "min_up": min_up,
                "mean_contacts": float(np.mean(contacts)),
                "done": done,
            }
        )
    result = {}
    for key in ["steps", "total_reward", "max_curl", "final_z", "min_up", "mean_contacts"]:
        result[key] = float(np.mean([row[key] for row in rows]))
    result["done"] = float(np.mean([row["done"] for row in rows]))
    result.update(asdict(params))
    result["closed_loop"] = float(closed_loop)
    result["score"] = score_row(result)
    return result


def score_row(row):
    fall_penalty = 200.0 if row["done"] else 0.0
    low_up_penalty = 50.0 * max(0.0, 0.85 - row["min_up"])
    flat_penalty = 120.0 * max(0.0, 0.12 - row["max_curl"])
    return 1000.0 * row["max_curl"] + 0.05 * row["total_reward"] - fall_penalty - low_up_penalty - flat_penalty


def iter_grid(limit):
    torso_values = [-0.02, -0.035, -0.05]
    front_hip_values = [-0.02, 0.0, 0.02]
    front_knee_values = [-0.04, 0.0, 0.04]
    hind_hip_values = [-0.02, 0.0, 0.02]
    hind_knee_values = [-0.04, 0.0, 0.04]
    switch_values = [40, 80]
    release_values = [100, 160]
    count = 0
    for torso in torso_values:
        for front_hip in front_hip_values:
            for front_knee in front_knee_values:
                for hind_hip in hind_hip_values:
                    for hind_knee in hind_knee_values:
                        for switch_step in switch_values:
                            for release_step in release_values:
                                yield CurlPolicyParams(
                                    torso=torso,
                                    front_hip=front_hip,
                                    front_knee=front_knee,
                                    hind_hip=hind_hip,
                                    hind_knee=hind_knee,
                                    switch_step=switch_step,
                                    release_step=release_step,
                                )
                                count += 1
                                if limit and count >= limit:
                                    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("quick_runs/policy_search.csv"))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--closed-loop", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for params in iter_grid(args.limit):
        row = evaluate_params(params, args.episodes, closed_loop=args.closed_loop)
        rows.append(row)
    rows.sort(key=score_row, reverse=True)

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {args.out}")
    for i, row in enumerate(rows[: args.top], 1):
        print(
            f"{i:02d} score={row['score']:.2f} curl={row['max_curl']:.3f} "
            f"reward={row['total_reward']:.1f} done={row['done']:.2f} "
            f"up={row['min_up']:.3f} contacts={row['mean_contacts']:.2f} "
            f"torso={row['torso']} front=({row['front_hip']},{row['front_knee']}) "
            f"hind=({row['hind_hip']},{row['hind_knee']}) switch={row['switch_step']} release={row['release_step']}"
        )


if __name__ == "__main__":
    main()
