"""四足折叠 PPO 训练脚本。"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import QuadrupedFoldEnv

NUM_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000
LOG_DIR = "./ppo_logs"
MODEL_DIR = "./ppo_models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def make_env():
    return QuadrupedFoldEnv()


if __name__ == "__main__":
    print(f"创建 {NUM_ENVS} 个并行环境...")
    env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    # 回调
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // NUM_ENVS,
        save_path=MODEL_DIR,
        name_prefix="quad_fold",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=50_000 // NUM_ENVS,
        deterministic=True,
    )

    print("开始训练 PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=LOG_DIR,
        device="cpu",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # 保存最终模型
    final_path = os.path.join(MODEL_DIR, "quad_fold_final")
    model.save(final_path)
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    print(f"模型已保存至 {final_path}")
