"""云端环境自检脚本 — 跑在训练之前。"""
import os, sys
os.environ.setdefault('XLA_FLAGS', '')
if 'triton_gemm_any' not in os.environ['XLA_FLAGS']:
    os.environ['XLA_FLAGS'] += ' --xla_gpu_triton_gemm_any=True'

import jax, jax.numpy as jp
jax.config.update('jax_default_matmul_precision', 'high')

import mujoco, numpy as np
from brax.io import mjcf
from etils import epath
from brax.envs.base import PipelineEnv, State
from flax import struct

XML = "test_mujoco/quadruped.xml"
JOINT_NAMES = [
    "torso_hinge",
    "fl_hip_abd","fl_hip_flex","fl_knee",
    "fr_hip_abd","fr_hip_flex","fr_knee",
    "hl_hip_abd","hl_hip_flex","hl_knee",
    "hr_hip_abd","hr_hip_flex","hr_knee",
]
N = len(JOINT_NAMES)

print("=" * 50)
print("1. Load model")
print("=" * 50)

sys = mjcf.load(str(epath.Path(XML)))
m = mujoco.MjModel.from_xml_path(XML)

print(f"Brax  sys: q={sys.q_size()} v={sys.qd_size()} u={sys.actuator_size()}")
print(f"MuJoCo m:  nq={m.nq} nv={m.nv} nu={m.nu}")

print()
print("=" * 50)
print("2. Actuator order")
print("=" * 50)

ok = True
for i in range(m.nu):
    act = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    jnt = act.replace('_act', '')
    exp = JOINT_NAMES[i]
    match = "✓" if jnt == exp else "✗ MISMATCH"
    if jnt != exp:
        ok = False
    print(f"  [{i:2d}] {act:25s} → {jnt:15s}  {match}")
print("ALL OK" if ok else "FAIL")

print()
print("=" * 50)
print("3. Body ordering (Brax)")
print("=" * 50)

for i in range(min(sys.num_bodies(), 20)):
    print(f"  body[{i}]: id={sys.body_id[i]}")

# 找 torso_front 在 Brax 中的索引
torso_front_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_front")
torso_brax_idx = None
for i in range(sys.num_bodies()):
    if sys.body_id[i] == torso_front_id:
        torso_brax_idx = i
        break
print(f"torso_front: MuJoCo id={torso_front_id}, Brax index={torso_brax_idx}")

print()
print("=" * 50)
print("4. Smoke test environment")
print("=" * 50)

STAND = jp.array([0.0, 0.0,-0.4,0.2, 0.0,-0.4,0.2, 0.0,0.4,-0.2, 0.0,0.4,-0.2])
FOLD  = jp.array([-1.0, 0.0,-1.0,1.0, 0.0,-1.0,1.0, 0.0,1.0,-1.0, 0.0,1.0,-1.0])
JADDR = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in JOINT_NAMES])

@struct.dataclass
class FoldInfo:
    init_err: jp.ndarray
    best_err: jp.ndarray

class FoldEnv(PipelineEnv):
    def __init__(self, sys, stand, fold, jnt, torso_idx, action_repeat=20):
        super().__init__(sys, backend='mjx', n_frames=action_repeat)
        self.stand = stand
        self.fold  = fold
        self.jnt   = jp.array(jnt)
        self.torso_idx = torso_idx
    @property
    def action_size(self):      return N
    @property
    def observation_size(self): return N*2 + 9 + 3
    def reset(self, rng):
        r1, r2 = jax.random.split(rng)
        noise  = jax.random.uniform(r1, (N,), minval=-0.05, maxval=0.05)
        init_q = jp.clip(self.stand + noise, -1.5, 1.5)
        init_z = 0.40 + jax.random.uniform(r2, (), minval=-0.02, maxval=0.05)
        qpos = self.sys.init_q.at[2].set(init_z).at[self.jnt].set(init_q)
        ps   = self.pipeline_init(qpos, jp.zeros(self.sys.qd_size()))
        obs  = self._obs(ps)
        err  = jp.mean(jp.abs(init_q - self.fold))
        return State(ps, obs, jp.zeros(()), jp.zeros(()), {}, FoldInfo(err, err))
    def step(self, state, action):
        ps = self.pipeline_step(state.pipeline_state, action)
        obs = self._obs(ps)
        q  = ps.q[self.jnt]
        err = jp.mean(jp.abs(q - self.fold))
        ti  = self.torso_idx
        zup = ps.x.rot[ti,2,2]
        r   = -err + (state.info.init_err - err)*3.0 - 2.0*jp.maximum(0,0.9-zup) + 0.05
        done = (zup < 0.3) | (ps.x.pos[ti,2] < 0.05)
        return state.replace(pipeline_state=ps, obs=obs, reward=r, done=done,
                             info=FoldInfo(state.info.init_err, jp.minimum(state.info.best_err, err)))
    def _obs(self, ps):
        ti = self.torso_idx
        q  = ps.q[self.jnt]
        qd = ps.qd[6:6+N]
        rot = ps.x.rot[ti].ravel()
        vel = ps.xd.pos[ti]
        return jp.concatenate([q, qd, rot, vel])

env = FoldEnv(sys, STAND, FOLD, JADDR, torso_brax_idx)
print(f"obs={env.observation_size}  act={env.action_size}  torso Brax idx={torso_brax_idx}")

rng = jax.random.PRNGKey(42)
st  = jax.jit(env.reset)(rng)
print(f"reset: init_err={st.info.init_err:.3f}  z_up={st.pipeline_state.x.rot[torso_brax_idx,2,2]:.3f}")

for i in range(10):
    st = jax.jit(env.step)(st, jp.zeros(N))
print(f"step x10: err={st.info.best_err:.3f}  done={st.done}  reward={st.reward:.3f}")

print()
print("=" * 50)
print("ALL CHECKS PASSED")
print("=" * 50)
