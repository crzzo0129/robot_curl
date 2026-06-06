import mujoco
import mujoco.viewer
import numpy as np
import time
import os

# ---------- 1. 加载模型 ----------
# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "pendulum.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ---------- 2. 找到每个关节在 qfrc_applied 里的地址 ----------
joint_names = ["shoulder", "elbow"]
dof_addresses = []
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    adr = model.jnt_dofadr[jid]        # 这个关节的第一个自由度的地址
    dof_addresses.append(adr)

print(f"自由度总数 nq={model.nq} nv={model.nv}")
print(f"jnt_dofadr: shoulder={dof_addresses[0]}, elbow={dof_addresses[1]}")

# 查 qpos 地址（虽然 hinge 下和 dof 一样，但确认一下）
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    print(f"  {name}: qposadr={model.jnt_qposadr[jid]}, dofadr={model.jnt_dofadr[jid]}")

# ---------- 3. 控制循环 ----------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 90
    viewer.cam.lookat = [0, 0, 2]   # 告诉相机看哪里，模型在z=2

    t = 0.0
    # 初始就给 elbow 一个弯折，跑起来立刻看到两杆不共线
    data.qpos[0] = 0.5   # shoulder 初始弯 ~29°
    data.qpos[1] = 0.5  # elbow 初始弯 ~46°
    while viewer.is_running():
        # ---- 计算控制力矩 ----
        # 肩关节：正弦波，让它来回摆
        torque_shoulder = 5.0
        # 肘关节：另一个频率的正弦波
        torque_elbow   = 0

        # ---- 直接把力矩灌进 qfrc_applied ----
        data.qfrc_applied[dof_addresses[0]] = torque_shoulder
        data.qfrc_applied[dof_addresses[1]] = torque_elbow

        # ---- 物理步进 ----
        mujoco.mj_step(model, data)

        # ---- 诊断：每 100 步打印一次 ----
        step = int(t / model.opt.timestep)
        if step % 100 == 0:
            deg0 = data.qpos[0] * 180 / np.pi
            deg1 = data.qpos[1] * 180 / np.pi
            forearm_global = data.qpos[0] + data.qpos[1]
            print(f"[{step:4d}] qpos=({deg0:+.1f}°, {deg1:+.1f}°)  "
                  f"forearm_global={forearm_global*180/np.pi:+.1f}°  "
                  f"torque_s={torque_shoulder:+.1f}")

        # ---- 时间推进和渲染 ----
        t += model.opt.timestep
        viewer.sync()
        time.sleep(0.001)
