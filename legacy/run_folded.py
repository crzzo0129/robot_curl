"""测试蜷缩折叠形态 — 手动设定目标角度后纯物理步进。"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "quadruped.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ---- 蜷缩目标姿态：躯干微弯 + 四肢包裹 ----
folded_pose = {
    "torso_hinge": -1,       # ~46° 躯干微弯，整体更圆

    # 侧摆：腿翻到躯干侧面
    "fl_hip_abd": 0.0, "fr_hip_abd": 0.0,
    "hl_hip_abd": 0.0, "hr_hip_abd": 0.0,

    # 前后髋：前腿向后折(+)、后腿向前折(-)，收敛到躯干
    "fl_hip_flex": -1.0, "fr_hip_flex": -1.0,
    "hl_hip_flex": 1.0,  "hr_hip_flex": 1.0,

    # 膝盖弯到极限
    "fl_knee": 1.0, "fr_knee": 1.0,
    "hl_knee": -1.0, "hr_knee": -1.0,
}

for name, angle in folded_pose.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = angle

# 抬高躯干，让球悬空后落地
data.qpos[2] = 0.25

print(f"蜷缩形态  nq={model.nq}  nv={model.nv}")

# ---- PD 锁定在蜷缩姿态 ----
joint_names = list(folded_pose.keys())
dof_addr = {}
target_q = {}
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    dof_addr[name] = model.jnt_dofadr[jid]
    target_q[name] = folded_pose[name]

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 130
    viewer.cam.elevation = -15
    viewer.cam.lookat = [0, 0, 0.25]

    while viewer.is_running():
        for name in joint_names:
            adr = dof_addr[name]
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            q    = data.qpos[model.jnt_qposadr[jid]]
            qvel = data.qvel[adr]
            torque = 100.0 * (target_q[name] - q) - 2.0 * qvel
            data.qfrc_applied[adr] = torque

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)
