"""四足机器人：先站立，再蜷缩成球。"""
import mujoco
import mujoco.viewer
import numpy as np
import time
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "quadruped.xml")

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ---- 站立目标姿态 ----
stand_pose = {
    "torso_hinge": 0.0,
    "fl_hip_abd": 0.0,  "fr_hip_abd": 0.0,
    "hl_hip_abd": 0.0,  "hr_hip_abd": 0.0,
    "fl_hip_flex": -0.4, "fr_hip_flex": -0.4,
    "hl_hip_flex": 0.4,  "hr_hip_flex": 0.4,
    "fl_knee": 0.2, "fr_knee": 0.2,
    "hl_knee": -0.2, "hr_knee": -0.2,
}

# ---- 蜷缩目标姿态（与 run_folded.py 同步）----
fold_pose = {
    "torso_hinge": -1.0,
    "fl_hip_abd": 0.0,  "fr_hip_abd": 0.0,
    "hl_hip_abd": 0.0,  "hr_hip_abd": 0.0,
    "fl_hip_flex": -1.0, "fr_hip_flex": -1.0,
    "hl_hip_flex": 1.0,  "hr_hip_flex": 1.0,
    "fl_knee": 1.0, "fr_knee": 1.0,
    "hl_knee": -1.0, "hr_knee": -1.0,
}

# 初始角度设为站立姿态
for name, angle in stand_pose.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[jid]] = angle

data.qpos[2] = 0.6  # 确保脚完全离地

joint_names = list(stand_pose.keys())
dof_addr = {}
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    dof_addr[name] = model.jnt_dofadr[jid]

print(f"四足模型  nq={model.nq}  nv={model.nv}")
print("阶段1: 站立站稳 (0-2s)")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 130
    viewer.cam.elevation = -25
    viewer.cam.lookat = [0, 0, 0.25]

    # PID 参数
    Kp_torso = 150.0; Kd_torso = 3.0; Ki_torso = 10.0
    Kp_abd   = 60.0;  Kd_abd   = 1.0; Ki_abd   = 5.0
    Kp_hip   = 150.0; Kd_hip   = 2.0; Ki_hip   = 12.0
    Kp_knee  = 60.0;  Kd_knee  = 1.0; Ki_knee  = 6.0

    integral = {name: 0.0 for name in joint_names}
    dt = model.opt.timestep

    sim_time = 0.0
    phase = "stand"  # stand -> transition -> fold

    while viewer.is_running():
        # 当前目标
        if sim_time < 2.0:
            current_target = stand_pose
        elif sim_time < 4.0:
            if phase == "stand":
                phase = "transition"
                print("阶段2: 蜷缩过渡 (2-4s)")
                integral = {name: 0.0 for name in joint_names}
            alpha = (sim_time - 2.0) / 2.0
            current_target = {
                name: (1 - alpha) * stand_pose[name] + alpha * fold_pose[name]
                for name in joint_names
            }
        else:
            if phase == "transition":
                phase = "fold"
                print("阶段3: 维持蜷缩 (>4s)")
            current_target = fold_pose

        # ---- PID 控制 ----
        for name in joint_names:
            adr = dof_addr[name]
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            q    = data.qpos[model.jnt_qposadr[jid]]
            qvel = data.qvel[adr]

            error = current_target[name] - q
            integral[name] += error * dt
            integral[name] = max(-1.0, min(1.0, integral[name]))

            if "abd" in name:
                Kp, Kd, Ki = Kp_abd, Kd_abd, Ki_abd
            elif "torso" in name:
                Kp, Kd, Ki = Kp_torso, Kd_torso, Ki_torso
            elif "knee" in name:
                Kp, Kd, Ki = Kp_knee, Kd_knee, Ki_knee
            else:
                Kp, Kd, Ki = Kp_hip, Kd_hip, Ki_hip

            torque = Kp * error + Ki * integral[name] - Kd * qvel
            data.qfrc_applied[adr] = torque

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)

        sim_time += dt

        # 诊断：每 500 步打印 fl_hip_flex 详情
        step = int(sim_time / dt)
        if step % 500 == 0:
            for name in ["fl_knee", "fr_knee", "hl_knee", "hr_knee"]:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                q = data.qpos[model.jnt_qposadr[jid]]
                tgt = current_target[name]
                err = tgt - q
                print(f"t={sim_time:.1f}s  {name:10s}  target={tgt:+7.3f}  actual={q:+7.3f}  err={err:+7.3f}")
