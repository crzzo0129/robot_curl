"""Lightweight CEM optimizer for curl policy parameters.

This gives us a train-like loop that only depends on NumPy and MuJoCo. It is
not a replacement for PPO, but it is useful while PyTorch is unavailable.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from policy_search import CurlPolicyParams, evaluate_params, score_row


PARAM_BOUNDS = [
    (-0.06, -0.005),  # torso
    (-0.05, 0.05),    # front_hip
    (-0.06, 0.06),    # front_knee
    (-0.05, 0.05),    # hind_hip
    (-0.06, 0.06),    # hind_knee
    (20, 100),        # switch_step
    (80, 220),        # release_step
]


def params_from_vector(vector):
    clipped = []
    for value, (low, high) in zip(vector, PARAM_BOUNDS):
        clipped.append(float(np.clip(value, low, high)))
    switch_step = int(round(clipped[5]))
    release_step = int(round(max(clipped[6], switch_step + 20)))
    release_step = int(np.clip(release_step, PARAM_BOUNDS[6][0], PARAM_BOUNDS[6][1]))
    return CurlPolicyParams(
        torso=clipped[0],
        front_hip=clipped[1],
        front_knee=clipped[2],
        hind_hip=clipped[3],
        hind_knee=clipped[4],
        switch_step=switch_step,
        release_step=release_step,
    )


def vector_from_params(params):
    return np.array(
        [
            params.torso,
            params.front_hip,
            params.front_knee,
            params.hind_hip,
            params.hind_knee,
            params.switch_step,
            params.release_step,
        ],
        dtype=float,
    )


def update_distribution(samples, scores, elite_frac):
    elite_count = max(1, int(round(len(samples) * elite_frac)))
    elite_idx = np.argsort(scores)[-elite_count:]
    elite = samples[elite_idx]
    mean = elite.mean(axis=0)
    std = elite.std(axis=0)
    default_floor = np.full(samples.shape[1], 1e-6, dtype=float)
    policy_floor = np.array([0.002, 0.004, 0.004, 0.004, 0.004, 2.0, 4.0])
    default_floor[: min(len(default_floor), len(policy_floor))] = policy_floor[: len(default_floor)]
    std = np.maximum(std, default_floor)
    return mean, std


def initial_distribution():
    low = np.array([b[0] for b in PARAM_BOUNDS], dtype=float)
    high = np.array([b[1] for b in PARAM_BOUNDS], dtype=float)
    mean = (low + high) / 2.0
    std = (high - low) / 4.0
    return mean, std


def optimize(generations, population, episodes, elite_frac, seed, closed_loop):
    rng = np.random.default_rng(seed)
    mean, std = initial_distribution()
    rows = []
    best = None

    for generation in range(generations):
        samples = rng.normal(mean, std, size=(population, len(PARAM_BOUNDS)))
        scores = []
        for sample_id, sample in enumerate(samples):
            params = params_from_vector(sample)
            row = evaluate_params(params, episodes=episodes, closed_loop=closed_loop)
            row["generation"] = generation
            row["sample_id"] = sample_id
            row["score"] = score_row(row)
            rows.append(row)
            scores.append(row["score"])
            if best is None or row["score"] > best["score"]:
                best = row
        mean, std = update_distribution(samples, np.array(scores), elite_frac)
        mean = np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(mean, PARAM_BOUNDS)], dtype=float)
        print(
            f"gen={generation} best_score={best['score']:.2f} "
            f"best_curl={best['max_curl']:.3f} done={best['done']:.2f}"
        )
    return rows, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--closed-loop", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("quick_runs/cem_optimize.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows, best = optimize(
        generations=args.generations,
        population=args.population,
        episodes=args.episodes,
        elite_frac=args.elite_frac,
        seed=args.seed,
        closed_loop=args.closed_loop,
    )
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out}")
    print(
        "best "
        f"score={best['score']:.2f} curl={best['max_curl']:.3f} reward={best['total_reward']:.1f} "
        f"done={best['done']:.2f} up={best['min_up']:.3f} contacts={best['mean_contacts']:.2f} "
        f"torso={best['torso']:.4f} front=({best['front_hip']:.4f},{best['front_knee']:.4f}) "
        f"hind=({best['hind_hip']:.4f},{best['hind_knee']:.4f}) "
        f"switch={best['switch_step']} release={best['release_step']}"
    )


if __name__ == "__main__":
    main()
