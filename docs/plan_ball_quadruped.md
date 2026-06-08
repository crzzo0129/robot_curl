# 四足-球体形态切换 实施计划

**目标：** 在现有 quadruped.xml 上完成躯干对折关节 + 腿侧摆关节的建模，Python 端确认所有关节 PD 可控。

**架构：** 一个 MJCF 文件 + 一个 Python 控制脚本。3 个改动点都在 XML 侧（躯干拆分、13 关节定义、腿几何调整），Python 侧仅扩展关节名列表和初始站姿。

**技术栈：** MuJoCo MJCF + Python mujoco bindings

---

## Task 1：躯干拆分为前后两段，加 y 轴对折关节

**文件：** `test_mujoco/quadruped.xml`

**改动：** 将单个 `torso` body 替换为 `torso_front` + `torso_rear`，中间 `torso_hinge` 铰链。

**替换内容：** 删除原躯干 block，写入新躯干 block（含 freejoint、两段 box、hinge）。

新结构：
```xml
<body name="torso_front" pos="0 0 0.28">
  <freejoint name="torso_free"/>
  <!-- 前半身盒子 (0.25×0.16×0.08m) -->
  <geom type="box" size="0.125 0.08 0.04" pos="0.125 0 0" rgba="0.85 0.4 0.15 1"/>

  <!-- 前左腿 / 前右腿 -->

  <!-- 后半身（铰接在前半身后端） -->
  <body name="torso_rear" pos="0.25 0 0">
    <joint name="torso_hinge" type="hinge" axis="0 1 0" range="0 180" damping="1.0"/>
    <geom type="box" size="0.125 0.08 0.04" pos="-0.125 0 0" rgba="0.85 0.4 0.15 1"/>

    <!-- 后左腿 / 后右腿 -->
  </body>
</body>
```

**验证：** 跑 `python run_quadruped.py`，看到两个橙色方块并排，中段有铰链。

---

## Task 2：每条腿在 hip 前插入 x 轴侧摆关节

**文件：** `test_mujoco/quadruped.xml`

**改动：** 每条腿从 2 层改为 3 层。以右前腿为例：

原结构：
```
torso → body fr_upper (pos: 0.2, -0.08, -0.04)
          → joint fr_hip (y-axis)
          → geom (capsule)
          → body fr_lower → joint fr_knee → ...
```

新结构：
```
torso → body fr_abd (pos: 0.2, -0.08, -0.04)
          → joint fr_hip_abd (x-axis, range -90 90)
          → body fr_upper
            → joint fr_hip_flex (y-axis, range -60 60)
            → geom (capsule)
            → body fr_lower → joint fr_knee → ...
```

完整腿模板（以右前腿为例）：
```xml
<body name="fr_abd" pos="0.2 -0.08 -0.04">
  <joint name="fr_hip_abd" type="hinge" axis="1 0 0" range="-90 90"/>
  <body name="fr_upper" pos="0 0 0">
    <joint name="fr_hip_flex" type="hinge" axis="0 1 0" range="-60 60"/>
    <geom class="leg" fromto="0 0 0 0 0 -0.15"/>
    <body name="fr_lower" pos="0 0 -0.15">
      <joint name="fr_knee" type="hinge" axis="0 1 0" range="-120 0"/>
      <geom class="leg" fromto="0 0 0 0 0 -0.12"/>
      <geom class="foot" pos="0 0 -0.12"/>
    </body>
  </body>
</body>
```

四条腿全部照此模式改造。注意左右腿的 y 坐标符号（左腿 +0.08，右腿 -0.08），前后腿的 x 坐标（前腿 +0.2，后腿 -0.2）。

**验证：** `python run_quadruped.py`，每条腿的根关节可以绕 x 轴侧摆。

---

## Task 3：更新 Python 控制脚本

**文件：** `test_mujoco/run_quadruped.py`

**改动：** 
1. 扩展 `initial_pose`，加入 4 个侧摆关节 + 躯干对折关节的初始角度
2. 扩展 `joint_names` 列表

```python
initial_pose = {
    # 躯干对折 — 初始展平
    "torso_hinge": 0.0,
    # 侧摆关节 — 初始全部朝下
    "fl_hip_abd": 0.0,  "fr_hip_abd": 0.0,
    "hl_hip_abd": 0.0,  "hr_hip_abd": 0.0,
    # 前后髋
    "fl_hip_flex": 0.35, "fr_hip_flex": 0.35,
    "hl_hip_flex": 0.35, "hr_hip_flex": 0.35,
    # 膝盖
    "fl_knee": -0.5, "fr_knee": -0.5,
    "hl_knee": -0.5, "hr_knee": -0.5,
}

joint_names = [
    "torso_hinge",
    "fl_hip_abd", "fl_hip_flex", "fl_knee",
    "fr_hip_abd", "fr_hip_flex", "fr_knee",
    "hl_hip_abd", "hl_hip_flex", "hl_knee",
    "hr_hip_abd", "hr_hip_flex", "hr_knee",
]
```

3. PD 参数里为侧摆和躯干对折设置合适的增益：

```python
# 躯干对折刚度
Kp_torso = 80.0
Kd_torso = 2.0
# 侧摆刚度
Kp_abd = 60.0
Kd_abd = 1.0
```

在循环内的 PD 判断逻辑扩展：
```python
if "abd" in name:
    torque = Kp_abd * (target_q[name] - q) - Kd_abd * qvel
elif "torso" in name:
    torque = Kp_torso * (target_q[name] - q) - Kd_torso * qvel
elif "knee" in name:
    torque = Kp_knee * (target_q[name] - q) - Kd_knee * qvel
else:
    torque = Kp_hip * (target_q[name] - q) - Kd_hip * qvel
```

**验证：** 机器人自由落体后站稳，13 个关节全部 PD 锁定在初始角度。

---

## Task 4：验证折叠序列

**手动测试：** 在控制循环外（before `while`）直接修改 `data.qpos` 为蜷缩姿态，然后跑纯物理步进（不加 PD），看折叠后形状是否合理。

蜷缩目标姿态（参考值，需按实际几何微调）：
```python
data.qpos[model.jnt_qposadr["torso_hinge"]] = 2.8  # ~160° 躯干对折
data.qpos[model.jnt_qposadr["fl_hip_abd"]]  = 1.0  # 腿侧摆包裹
data.qpos[model.jnt_qposadr["fr_hip_abd"]]  = 1.0
data.qpos[model.jnt_qposadr["hl_hip_abd"]]  = 1.0
data.qpos[model.jnt_qposadr["hr_hip_abd"]]  = 1.0
data.qpos[model.jnt_qposadr["fl_hip_flex"]] = -1.5 # 腿往躯干方向折
data.qpos[model.jnt_qposadr["fr_hip_flex"]] = -1.5
data.qpos[model.jnt_qposadr["hl_hip_flex"]] = 1.5
data.qpos[model.jnt_qposadr["hr_hip_flex"]] = 1.5
data.qpos[model.jnt_qposadr["fl_knee"]]     = -2.0 # 膝弯到最大
# ... 其他膝盖同理
```

**验证：** 跑一次，目视检查蜷缩后的形态是否接近球形，腿的脚球是否在外围形成接触面。
