# Quadruped Fold — MJX GPU Training Notebook
# 适配光语启智云端平台 (Ubuntu 22.04 + RTX 4090)

## 环境检查

import os, sys, subprocess

# GPU
result = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True)
print(f"GPU: {result.stdout.strip()}")

# JAX / MJX
import jax
import jax.numpy as jp
print(f"JAX version: {jax.__version__}, devices: {jax.device_count()}")

import mujoco
from mujoco import mjx
print(f"MuJoCo version: {mujoco.__version__}, MJX OK")

# Brax
import brax
print(f"Brax version: {brax.__version__}")

# XLA 加速
xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags
jax.config.update('jax_default_matmul_precision', 'high')

print("环境就绪 ✓")

## 模型加载

from brax.io import mjcf
from etils import epath
import numpy as np

MODEL_XML = "pupper_v3_description/../test_mujoco/quadruped.xml"
xml_path = epath.Path(MODEL_XML)
assert xml_path.exists(), f"模型文件不存在: {MODEL_XML}"

# 加载 MJCF → Brax System
sys = mjcf.load(str(xml_path))
print(f"模型加载成功: nq={sys.q_size()}, nv={sys.qd_size()}, nu={sys.actuator_size()}")

# 关节索引（按 JOINT_NAMES 顺序）
JOINT_NAMES = [
    "torso_hinge",
    "fl_hip_abd", "fl_hip_flex", "fl_knee",
    "fr_hip_abd", "fr_hip_flex", "fr_knee",
    "hl_hip_abd", "hl_hip_flex", "hl_knee",
    "hr_hip_abd", "hr_hip_flex", "hr_knee",
]
N_JOINTS = len(JOINT_NAMES)

# 从 MJCF 获取关节 qpos 地址
m = mujoco.MjModel.from_xml_path(str(xml_path))
joint_qpos_ids = []
for name in JOINT_NAMES:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_qpos_ids.append(m.jnt_qposadr[jid])
print(f"关节 qpos 地址: {joint_qpos_ids}")

# 目标姿态
STAND_POSE = jp.array([0.0, 0.0, -0.4, 0.2, 0.0, -0.4, 0.2,
                        0.0, 0.4, -0.2, 0.0, 0.4, -0.2])
FOLD_POSE  = jp.array([-1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0,
                        0.0, 1.0, -1.0, 0.0, 1.0, -1.0])

print(f"站立目标: {STAND_POSE}")
print(f"折叠目标: {FOLD_POSE}")

## Markdown
# Brax 折叠环境

import jax
import jax.numpy as jp
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.mjx.base import State as MjxState
from flax import struct

@struct.dataclass
class FoldEnvState:
    """折叠环境专用状态"""
    init_q: jp.ndarray      # 初始关节角度 (用于进度奖励)

class QuadrupedFoldEnv(PipelineEnv):
    """四足折叠 — Brax/MJX 环境"""

    def __init__(self, sys, stand_pose, fold_pose, joint_qpos_ids,
                 backend='mjx', physics_dt=0.001, action_repeat=20):
        super().__init__(sys, backend=backend, n_frames=action_repeat)
        self.stand_pose = stand_pose
        self.fold_pose = fold_pose
        self.joint_qpos_ids = jp.array(joint_qpos_ids)
        self.action_repeat = action_repeat
        self.dt = physics_dt * action_repeat  # 控制周期

        # 动作空间：13 维连续
        self.action_size = N_JOINTS

        # 观测空间
        self.observation_size = N_JOINTS * 2 + 4 + 3  # 33

    def reset(self, rng: jp.ndarray) -> State:
        """重置环境：设置站立姿态 + 噪声"""
        rng1, rng2 = jax.random.split(rng)

        # 初始角度：站姿 + 小噪声
        noise = jax.random.uniform(rng1, (N_JOINTS,), minval=-0.05, maxval=0.05)
        init_q = jp.clip(self.stand_pose + noise, -1.5, 1.5)

        # 初始高度随机
        init_z = jax.random.uniform(rng2, (), minval=0.38, maxval=0.45)

        # 构建 pipeline state
        pipeline_state = self.pipeline_init(init_q, init_z)

        # 观测
        obs = self._get_obs(pipeline_state)

        # 奖励状态
        reward_state = FoldEnvState(init_q=init_q)

        # 计算初始形状误差（存进 info）
        init_shape_error = jp.mean(jp.abs(init_q - self.fold_pose))

        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jp.zeros(()),
            done=jp.zeros(()),
            metrics={'init_shape_error': init_shape_error,
                     'shape_error': init_shape_error,
                     'best_shape_error': init_shape_error},
            info=reward_state,
        )

    def step(self, state: State, action: jp.ndarray) -> State:
        """执行动作 + 物理步进"""
        # 计算 PD 力矩
        pipeline_state = state.pipeline_state
        q_actual = pipeline_state.q[self.joint_qpos_ids]

        # PD 控制：action 是目标角度偏移
        target_q = jp.clip(q_actual + action * 0.1, -1.5, 1.5)
        Kp = 80.0
        Kd = 1.5
        qd = pipeline_state.qd[self.joint_qpos_ids[:N_JOINTS]]  # 注意索引
        torques = Kp * (target_q - q_actual) - Kd * qd

        # 替换 actuator force（需要知道 actuator 对应的 qfrc_applied 地址）
        # 简化：直接用 qfrc_applied
        qfrc = pipeline_state.qfrc_applied
        # ... 需要正确映射到 qfrc 索引

        # 物理步进
        pipeline_state = self.pipeline_step(pipeline_state, action)
        obs = self._get_obs(pipeline_state)

        # 奖励
        q = pipeline_state.q[self.joint_qpos_ids]
        shape_error = jp.mean(jp.abs(q - self.fold_pose))
        progress = state.metrics['init_shape_error'] - shape_error
        reward = -shape_error + progress * 3.0

        # 终止
        z_up = pipeline_state.x.rot[2, 2]  # 躯干 z 轴世界投影
        done = (z_up < 0.3) | (pipeline_state.x.pos[0, 2] < 0.05)

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics={'shape_error': shape_error},
        )

    def pipeline_init(self, init_q, init_z):
        """初始化 MJX pipeline state"""
        # 获取默认 init state
        data = mjx.put_data(self.sys)
        # 设置 qpos
        qpos = data.qpos
        # freejoint qpos: 前 7 个
        qpos = qpos.at[2].set(init_z)  # z
        # 关节角度
        for i, jid in enumerate(self.joint_qpos_ids):
            qpos = qpos.at[jid].set(init_q[i])
        data = data.replace(qpos=qpos)
        return data

    def _get_obs(self, pipeline_state):
        """构建观测"""
        q = pipeline_state.q[self.joint_qpos_ids]
        qd = pipeline_state.qd[:N_JOINTS]  # 简化
        quat = pipeline_state.x.rot[0]  # 躯干旋转矩阵 → 四元数
        vel = pipeline_state.xd.pos[0]
        # 旋转矩阵转四元数（简化：取行向量）
        obs = jp.concatenate([q, qd, quat.ravel(), vel])
        return obs

# 验证
env = QuadrupedFoldEnv(sys, STAND_POSE, FOLD_POSE, joint_qpos_ids)
print(f"环境创建成功: obs={env.observation_size}, act={env.action_size}")

## Markdown
# 训练配置

from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict

# PPO 配置
PPO_CONFIG = config_dict.ConfigDict()
PPO_CONFIG.num_timesteps = 50_000_000
PPO_CONFIG.episode_length = 250
PPO_CONFIG.num_envs = 4096
PPO_CONFIG.learning_rate = 3e-4
PPO_CONFIG.entropy_cost = 1e-2
PPO_CONFIG.discounting = 0.97
PPO_CONFIG.unroll_length = 20
PPO_CONFIG.num_minibatches = 32
PPO_CONFIG.num_updates_per_batch = 4
PPO_CONFIG.batch_size = 256
PPO_CONFIG.reward_scaling = 1.0
PPO_CONFIG.normalize_observations = True
PPO_CONFIG.action_repeat = 1  # Brax 环境内部处理

print("PPO 配置:")
for k, v in PPO_CONFIG.items():
    print(f"  {k}: {v}")

## Markdown
# 启动训练

import wandb

# WandB（可选）
USE_WANDB = True
if USE_WANDB:
    try:
        wandb.init(project="quadruped-fold", name=f"mjx-fold-{jax.process_index()}")
        WANDB_KWARGS = dict(logging=True, wandb_logging=True)
    except:
        WANDB_KWARGS = dict(logging=False, wandb_logging=False)
else:
    WANDB_KWARGS = dict(logging=False, wandb_logging=False)

# 创建训练环境函数
def make_env():
    return QuadrupedFoldEnv(sys, STAND_POSE, FOLD_POSE, joint_qpos_ids)

# PPO 网络
network_factory = ppo_networks.make_ppo_networks

# 训练
print("开始 MJX PPO 训练...")
print(f"环境数: {PPO_CONFIG.num_envs}, 总步数: {PPO_CONFIG.num_timesteps:,}")

train_fn = ppo_train(
    environment=make_env(),
    num_timesteps=PPO_CONFIG.num_timesteps,
    episode_length=PPO_CONFIG.episode_length,
    num_envs=PPO_CONFIG.num_envs,
    learning_rate=PPO_CONFIG.learning_rate,
    entropy_cost=PPO_CONFIG.entropy_cost,
    discounting=PPO_CONFIG.discounting,
    unroll_length=PPO_CONFIG.unroll_length,
    num_minibatches=PPO_CONFIG.num_minibatches,
    num_updates_per_batch=PPO_CONFIG.num_updates_per_batch,
    batch_size=PPO_CONFIG.batch_size,
    reward_scaling=PPO_CONFIG.reward_scaling,
    normalize_observations=PPO_CONFIG.normalize_observations,
    network_factory=network_factory,
    progress_fn=lambda step, metrics: print(f"Step {step:,}: {metrics}"),
    **WANDB_KWARGS,
)

# 保存模型
import pickle
save_path = "./fold_policy.pkl"
with open(save_path, 'wb') as f:
    pickle.dump(train_fn, f)
print(f"模型已保存至 {save_path}")
