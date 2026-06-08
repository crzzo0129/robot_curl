"""加载已训练模型，可视化或跑 benchmark。"""
import os
import numpy as np
import mujoco
import mujoco.viewer
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import QuadrupedFoldEnv, JOINT_NAMES

# ---- 配置 ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ppo_models", "best_model.zip")
NORM_PATH  = os.path.join(os.path.dirname(__file__), "..", "ppo_models", "vec_normalize.pkl")
XML_PATH   = os.path.join(os.path.dirname(__file__), "quadruped.xml")
VIEW_MODE  = True   # True = viewer 可视化, False = 纯 benchmark

# ---- 加载 ----
print("加载模型...")
model = PPO.load(MODEL_PATH)

# 用 VecNormalize 包裹环境（和训练时一致）
def make_env():
    return QuadrupedFoldEnv()

env = DummyVecEnv([make_env])

if os.path.exists(NORM_PATH):
    env = VecNormalize.load(NORM_PATH, env)
    env.training = False
    env.norm_reward = False
    print("VecNormalize 已加载")
else:
    print("未找到 vec_normalize.pkl, 跳过")

# ---- 运行 ----
if VIEW_MODE:
    # MuJoCo viewer 模式
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)

    # 设置站立初始姿态
    stand = [0.0, 0.0, -0.4, 0.2, 0.0, -0.4, 0.2, 0.0, 0.4, -0.2, 0.0, 0.4, -0.2]
    for i, name in enumerate(JOINT_NAMES):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        d.qpos[m.jnt_qposadr[jid]] = stand[i]
    d.qpos[2] = 0.4

    obs = env.reset()
    step = 0

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25
        viewer.cam.lookat = [0, 0, 0.25]

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step += 1

            # 同步 viewer 状态
            d.qpos[:] = env.envs[0].data.qpos[:]
            d.qvel[:] = env.envs[0].data.qvel[:]
            d.time = env.envs[0].data.time
            mujoco.mj_forward(m, d)
            viewer.sync()
            time.sleep(0.001)

            if done[0]:
                print(f"episode 结束, step={step}, reward={reward[0]:.2f}")
                obs = env.reset()
                step = 0
else:
    # Benchmark 模式
    total_reward = 0.0
    episodes = 10
    for ep in range(episodes):
        obs = env.reset()
        ep_reward = 0.0
        for _ in range(250):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            if done[0]:
                break
        total_reward += ep_reward
        print(f"ep {ep+1}: reward={ep_reward:.2f}")
    print(f"平均 reward: {total_reward/episodes:.2f}")
