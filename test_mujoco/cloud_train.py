"""四足折叠 PPO 训练脚本 — 云端版 (Linux + GPU + MJX 可选)。"""
import os
import sys
import argparse

# ---- 命令行参数 ----
parser = argparse.ArgumentParser()
parser.add_argument("--mjx", action="store_true", help="使用 MJX GPU 加速")
parser.add_argument("--envs", type=int, default=32, help="并行环境数")
parser.add_argument("--steps", type=int, default=2_000_000, help="总训练步数")
parser.add_argument("--device", type=str, default="cuda", help="cuda 或 cpu")
args = parser.parse_args()

# ---- MJX 模式 ----
if args.mjx:
    import mujoco
    try:
        import mujoco.mjx as mjx
        MJX_AVAILABLE = True
        print("MJX GPU 加速已启用")
    except ImportError:
        print("错误: mujoco.mjx 未找到, pip install mujoco>=3.0 jax jaxlib")
        sys.exit(1)
else:
    MJX_AVAILABLE = False
    print(f"CPU 模式, {args.envs} 并行环境")

# ---- 主训练逻辑 ----
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

LOG_DIR = "./ppo_logs_cloud"
MODEL_DIR = "./ppo_models_cloud"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env():
    if MJX_AVAILABLE:
        # MJX 环境（GPU 物理 + CPU 策略）
        from env_mjx import QuadrupedFoldEnvMJX
        return QuadrupedFoldEnvMJX()
    else:
        from env import QuadrupedFoldEnv
        return QuadrupedFoldEnv()


if __name__ == "__main__":
    print(f"并行环境数: {args.envs} | 总步数: {args.steps:,} | device: {args.device}")

    # 多进程环境（Linux 原生 fork）
    env = SubprocVecEnv([make_env for _ in range(args.envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = SubprocVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // args.envs, 1),
        save_path=MODEL_DIR,
        name_prefix="quad_fold_cloud",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(50_000 // args.envs, 1),
        deterministic=True,
    )

    print("开始训练...")
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
        tensorboard_log=LOG_DIR,
        device=args.device,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(os.path.join(MODEL_DIR, "quad_fold_cloud_final"))
    env.save(os.path.join(MODEL_DIR, "vec_normalize_cloud.pkl"))
    print("训练完成!")
