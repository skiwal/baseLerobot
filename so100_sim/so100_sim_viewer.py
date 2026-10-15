#!/usr/bin/env python
"""SO-100 MuJoCo 交互窗口演示 —— 在 MuJoCo 原生 viewer 里实时看机械臂运动。

与 so100_sim_demo.py 的区别:
- so100_sim_demo.py  : 离屏渲染(EGL) -> 生成 mp4/png, 适用无显示环境
- 本脚本             : 打开 MuJoCo 原生窗口(GLFW), 需要图形显示

本机的运行方式(两台机器上的终端里执行效果不同):

  情况 A: 你在 RustDesk 远程桌面的终端里执行 —— 直接跑:
      python so100_sim/so100_sim_viewer.py

  情况 B: 你在 VSCode SSH 终端里执行 —— 需要显式指定显示和目标桌面:
      DISPLAY=:0 XAUTHORITY=/run/user/1000/.mutter-Xwaylandauth.JHXJW3 \
      XDG_RUNTIME_DIR=/run/user/1000 MUJOCO_GL=glfw \
      python so100_sim/so100_sim_viewer.py

  窗口会出现在 :0 桌面上(RustDesk 里可见)。

查看器内置操作(鼠标):
  - 左键拖拽: 旋转视角
  - 右键拖拽 / 中键拖拽: 平移 / 缩放
  - 双击: 选中物体
  - 空格: 暂停物理(仅 viewer 内部)
  - ESC: 关闭窗口

同时支持键盘控制机械臂(焦点需在窗口内):
  A/D  shoulder_pan   左/右
  W/S  shoulder_lift  抬/落   (W=抬起)
  Q/E  elbow_flex     收/伸
  Z/X  wrist_flex     俯/仰
  C/V  wrist_roll     滚转
  O/P  gripper        张开/闭合
  R    复位到初始姿态
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "cache" / "mujoco_menagerie" / "trs_so_arm100"

# MuJoCo actuator 名 <-> LeRobot 真机电机名
ACT2MOTOR = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
    "Jaw": "gripper",
}

HOME = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.0,
    "elbow_flex": 1.6,
    "wrist_flex": -0.9,
    "wrist_roll": 0.0,
    "gripper": 1.2,
}

# 键盘 -> (电机名, 增量方向)
KEYMAP = {
    "a": ("shoulder_pan", -1),
    "d": ("shoulder_pan", +1),
    "w": ("shoulder_lift", +1),
    "s": ("shoulder_lift", -1),
    "q": ("elbow_flex", -1),
    "e": ("elbow_flex", +1),
    "z": ("wrist_flex", -1),
    "x": ("wrist_flex", +1),
    "c": ("wrist_roll", -1),
    "v": ("wrist_roll", +1),
    "o": ("gripper", +1),
    "p": ("gripper", -1),
}

STEP_RAD = 0.03  # 每次按键的关节增量


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)

    # actuator 索引 -> 电机名
    idx2motor = {
        i: ACT2MOTOR[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)] for i in range(model.nu)
    }
    motor2idx = {v: k for k, v in idx2motor.items()}
    ctrl = np.zeros(model.nu)
    for motor, val in HOME.items():
        ctrl[motor2idx[motor]] = val
    data.ctrl[:] = ctrl

    def settle(seconds: float = 0.6) -> None:
        for _ in range(int(seconds / model.opt.timestep)):
            mujoco.mj_step(model, data)

    settle()

    # 键盘状态。注意: MuJoCo 3.14 的 key_callback 是 launch_passive 的构造参数,
    # 且回调在 GLFW 线程里触发, 因此回调只往集合里塞字符, 主循环再消费。
    pressed: set[str] = set()

    def key_callback(keycode: int) -> None:
        # GLFW 字母键的 keycode 就是大写 ASCII (A=65..Z=90), 转小写后直接查表
        ch = chr(keycode).lower() if 65 <= keycode <= 90 else ""
        if ch in KEYMAP or ch == "r":
            pressed.add(ch)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # 视角调成一个能看清整条臂的角度
        viewer.cam.lookat[:] = [0.0, -0.05, 0.18]
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -30.0

        print("窗口已打开。键盘: A/D=肩转 W/S=肩抬 Q/E=肘 Z/X=腕俯 C/V=腕滚 O/P=夹爪 R=复位")
        print("鼠标: 左键拖拽旋转, 右键平移, 滚轮缩放。ESC 或关窗退出。")

        t0 = time.time()
        last = t0
        last_auto = t0
        AUTO_PERIOD = 6.0  # 无操作时每 6 秒自动演示一段扫动

        while viewer.is_running():
            now = time.time()
            dt = now - last
            last = now

            # 应用按键增量
            if pressed:
                for ch in list(pressed):
                    if ch == "r":
                        for motor, val in HOME.items():
                            ctrl[motor2idx[motor]] = val
                    else:
                        motor, sign = KEYMAP[ch]
                        i = motor2idx[motor]
                        lo, hi = model.actuator_ctrlrange[i]
                        ctrl[i] = float(np.clip(ctrl[i] + sign * STEP_RAD, lo, hi))
                pressed.clear()
                last_auto = now

            # 空闲自动演示: 肩部平滑扫动
            elif now - last_auto > 2.0:
                phase = (now - last_auto) % AUTO_PERIOD / AUTO_PERIOD
                ctrl[motor2idx["shoulder_pan"]] = 1.1 * np.sin(2 * np.pi * phase)
                if now - last_auto > 4.0:
                    # 同时演示夹爪开合
                    g = 0.79 - 0.96 * np.sin(2 * np.pi * phase * 2)
                    i = motor2idx["gripper"]
                    lo, hi = model.actuator_ctrlrange[i]
                    ctrl[i] = float(np.clip(g, lo, hi))

            data.ctrl[:] = ctrl

            # 物理按 2ms 子步推进, 与实时同步
            n_sub = max(1, int(dt / model.opt.timestep))
            for _ in range(n_sub):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, 1.0 / 120.0 - (time.time() - now)))


if __name__ == "__main__":
    main()
