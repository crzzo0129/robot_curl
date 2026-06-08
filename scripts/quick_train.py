"""Small PPO training entrypoint for fast robot-curl iterations."""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from robot_curl.config_args import add_task_config_args, task_config_from_args
from robot_curl.env import QuadrupedFoldEnv


def make_env(config):
    return QuadrupedFoldEnv(config=config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("quick_runs/curl_smoke"))
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    add_task_config_args(parser)
    args = parser.parse_args()
    task_config = task_config_from_args(args)

    args.out.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([lambda: make_env(task_config)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=5,
        learning_rate=3e-4,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device="cpu",
    )

    model.learn(total_timesteps=args.steps, progress_bar=False)
    model.save(args.out / "model")
    env.save(args.out / "vec_normalize.pkl")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
