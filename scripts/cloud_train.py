"""Cloud-oriented PPO training entrypoint.

This is the CPU/SubprocVecEnv version. MJX support should be added as a
separate environment backend so its observations, rewards, and action semantics
can be kept aligned with ``robot_curl.env.QuadrupedFoldEnv``.
"""
import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from robot_curl.env import QuadrupedFoldEnv


def make_env():
    return QuadrupedFoldEnv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=Path, default=Path("ppo_models_cloud"))
    parser.add_argument("--log-dir", type=Path, default=Path("ppo_logs_cloud"))
    parser.add_argument("--mjx", action="store_true", help="Reserved for the future MJX backend.")
    args = parser.parse_args()

    if args.mjx:
        raise SystemExit("MJX backend is not wired yet; use the MuJoCo/Gymnasium backend for now.")

    args.out.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    env = SubprocVecEnv([make_env for _ in range(args.envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = SubprocVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.envs, 1),
        save_path=str(args.out),
        name_prefix="robot_curl_cloud",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.out),
        log_path=str(args.log_dir),
        eval_freq=max(50_000 // args.envs, 1),
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256 if args.envs >= 16 else 64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=str(args.log_dir),
        device=args.device,
    )

    model.learn(total_timesteps=args.steps, callback=[checkpoint_callback, eval_callback], progress_bar=True)
    model.save(args.out / "robot_curl_cloud_final")
    env.save(args.out / "vec_normalize_cloud.pkl")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
