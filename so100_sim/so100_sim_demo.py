#!/usr/bin/env python
"""SO-100 MuJoCo 仿真的最小可运行演示。

模型来源: mujoco_menagerie / trs_so_arm100 (Apache-2.0, Google DeepMind 维护)

设计意图:
- 复刻 LeRobot Robot 接口的 get_observation() / send_action() 约定,
  这样同一个测试脚本既可以在真机上跑, 也可以在仿真上跑。
- 真机 SO-100 的 6 个电机与 MuJoCo 的 6 个 actuator 一一对应:
    shoulder_pan  <-> Rotation
    shoulder_lift <-> Pitch
    elbow_flex    <-> Elbow
    wrist_flex    <-> Wrist_Pitch
    wrist_roll    <-> Wrist_Roll
    gripper       <-> Jaw

运行:
    MUJOCO_GL=egl python so100_sim_demo.py
输出:
    out/so100_scripted.mp4   —— 脚本化动作的仿真录像
    out/frame_*.png          —— 关键帧截图
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import mujoco
import numpy as np

from video_utils import write_video_h264

MODEL_DIR = Path(__file__).resolve().parents[1] / "cache" / "mujoco_menagerie" / "trs_so_arm100"
OUT_DIR = Path(__file__).resolve().parent / "out"

# MuJoCo actuator 名(按模型里的顺序) -> LeRobot 真机电机名
ACT2MOTOR = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
    "Jaw": "gripper",
}
MOTORS = list(ACT2MOTOR.values())


class SO100Sim:
    """把 MuJoCo 的 SO-ARM100 包成 LeRobot 风格的 Robot 对象。

    真机上 SO100Follower 提供 get_observation()/send_action();
    这里用同样的方法名, 让上层代码(数据采集/策略推理)无需区分仿真与真机。
    """

    def __init__(self, model_path: Path, width: int = 640, height: int = 480, fps: int = 30):
        self.model = mujoco.MjModel.from_xml_path(str(model_path / "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        # 一个固定的第三人称视角, 看向机械臂中部。
        # 距离与扫动幅度(见下方动作序列 ±0.9 rad)配合, 保证整条臂全程在画面内。
        self.camera.lookat[:] = [0.0, -0.05, 0.16]
        self.camera.distance = 0.92
        self.camera.azimuth = 145.0
        self.camera.elevation = -26.0
        self.ctrl = np.zeros(self.model.nu)  # 位置目标 (rad), 夹爪是关节角
        self.dt = 1.0 / fps
        self.model.opt.timestep = 0.002  # 物理子步 2ms
        self.substeps = int(round(self.dt / self.model.opt.timestep))
        self._reset_pose()

    # ---------- 姿态工具 ----------
    def _reset_pose(self) -> None:
        """到一个可读的初始姿态: 臂抬起, 夹爪张开。"""
        home = {
            "Rotation": 0.0,
            "Pitch": -1.0,
            "Elbow": 1.6,
            "Wrist_Pitch": -0.9,
            "Wrist_Roll": 0.0,
            "Jaw": 1.2,  # 正值为张开
        }
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.ctrl[i] = home[name]
        self.data.ctrl[:] = self.ctrl
        for _ in range(200):  # 让臂安定下来
            mujoco.mj_step(self.model, self.data)

    def joint_positions_rad(self) -> dict[str, float]:
        """返回 {真机电机名: 关节角度(rad)}"""
        out = {}
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_id = self.model.actuator_trnid[i, 0]
            out[ACT2MOTOR[name]] = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        return out

    def joint_positions_deg(self) -> dict[str, float]:
        return {k: float(np.degrees(v)) for k, v in self.joint_positions_rad().items()}

    def gripper_tip_pose(self) -> np.ndarray:
        """末端(gripper 挂点)位置, 策略的笛卡尔观测通常用它。"""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Gripper")
        return self.data.xpos[body_id].copy()

    # ---------- LeRobot Robot 风格接口 ----------
    def get_observation(self) -> dict:
        """与真机约定一致: 关节位置 + 相机图像。"""
        return {
            "joint_positions": self.joint_positions_rad(),
            "eef_pos": self.gripper_tip_pose(),
            "image": self.render(),
        }

    def send_action(self, action: dict[str, float]) -> None:
        """action: {真机电机名: 目标角度(rad)}。未给的关节保持不动。"""
        motor2act = {v: k for k, v in ACT2MOTOR.items()}
        for motor, value in action.items():
            if motor not in motor2act:
                raise KeyError(f"未知电机 '{motor}', 可用: {MOTORS}")
            idx = [
                i
                for i in range(self.model.nu)
                if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == motor2act[motor]
            ][0]
            self.ctrl[idx] = float(np.clip(value, *self.model.actuator_ctrlrange[idx]))
        self.data.ctrl[:] = self.ctrl

    def step(self) -> None:
        """推进一个控制周期 (默认 1/30 s)。"""
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

    def render(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render()

    def close(self) -> None:
        self.renderer.close()


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    sim = SO100Sim(MODEL_DIR)

    fps = 30
    frames: list[np.ndarray] = []
    keyframes = {"start": None, "sweep_mid": None, "grip_closed": None, "end": None}

    # 名字对到 actuator 顺序, 方便直接按名字写脚本
    def pose(**kw) -> None:
        sim.send_action(kw)

    base = {
        "shoulder_lift": -1.0,
        "elbow_flex": 1.6,
        "wrist_flex": -0.9,
        "wrist_roll": 0.0,
    }

    def hold(seconds: float, snapshot: str | None = None) -> None:
        for _ in range(int(seconds * fps)):
            sim.step()
            frames.append(sim.render())
        if snapshot:
            keyframes[snapshot] = frames[-1].copy()

    # ---------- 动作序列 ----------
    # 1. 起始姿态
    pose(shoulder_pan=0.0, gripper=1.4, **base)
    hold(0.5, "start")

    # 2+3. 肩部左右扫动 (shoulder_pan), 中途抓拍
    for t in np.linspace(0, 1, fps, endpoint=False):
        pose(shoulder_pan=lerp(0.0, -0.9, t * t), gripper=1.4, **base)
        sim.step()
        frames.append(sim.render())
    keyframes["sweep_mid"] = frames[-1].copy()
    for t in np.linspace(0, 1, fps, endpoint=False):
        pose(shoulder_pan=lerp(-0.9, 0.9, t), gripper=1.4, **base)
        sim.step()
        frames.append(sim.render())
    for t in np.linspace(0, 1, fps // 2, endpoint=False):
        pose(shoulder_pan=lerp(0.9, 0.0, t), gripper=1.4, **base)
        sim.step()
        frames.append(sim.render())

    # 4. 闭合夹爪
    for t in np.linspace(0, 1, fps // 2, endpoint=False):
        pose(shoulder_pan=0.0, gripper=lerp(1.4, -0.17, t), **base)
        sim.step()
        frames.append(sim.render())
    keyframes["grip_closed"] = frames[-1].copy()
    hold(0.4)

    # 5. 关节联动 - 像在做一个"抓取-抬起"的动作
    picks = [
        # (目标姿态, 秒数)
        ({"shoulder_lift": -1.9, "elbow_flex": 1.1, "wrist_flex": -0.3, "wrist_roll": 0.6}, 1.0),
        ({"shoulder_lift": -0.6, "elbow_flex": 2.1, "wrist_flex": -1.2, "wrist_roll": -0.6}, 1.2),
        ({"shoulder_lift": -1.4, "elbow_flex": 1.4, "wrist_flex": -0.6, "wrist_roll": 0.0}, 1.0),
    ]
    cur = dict(base)
    for target, secs in picks:
        start = dict(cur)
        for t in np.linspace(0, 1, int(secs * fps), endpoint=False):
            # 平滑插值 (三次缓动)
            s = t * t * (3 - 2 * t)
            merged = {k: lerp(start[k], target[k], s) for k in target}
            merged["shoulder_pan"] = 0.0
            merged["gripper"] = -0.17
            pose(**merged)
            sim.step()
            frames.append(sim.render())
        cur = dict(target)

    # 6. 张开夹爪收尾
    for t in np.linspace(0, 1, fps, endpoint=False):
        pose(gripper=lerp(-0.17, 1.4, t), shoulder_pan=0.0, **{k: cur[k] for k in base})
        sim.step()
        frames.append(sim.render())
    hold(0.5, "end")

    # ---------- 保存视频与截图 ----------
    video_path = OUT_DIR / "so100_scripted.mp4"
    write_video_h264(video_path, frames, fps=fps)

    for name, img in keyframes.items():
        if img is not None:
            cv2.imwrite(str(OUT_DIR / f"frame_{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # ---------- 打印接口演示 ----------
    obs = sim.get_observation()
    print("=== get_observation() 结构 (策略的输入) ===")
    print("  keys:", list(obs.keys()))
    print("  joint_positions:", {k: round(v, 3) for k, v in obs["joint_positions"].items()})
    print("  eef_pos (m):", obs["eef_pos"].round(3))
    print("  image:", obs["image"].shape, obs["image"].dtype)
    print()
    print("=== send_action() 用法 (策略的输出) ===")
    print("  sim.send_action({'shoulder_pan': 0.5, 'gripper': 1.0, ...})")
    print()
    print(f"视频: {video_path}  ({len(frames)} 帧, {len(frames)/fps:.1f} 秒)")
    print(f"截图: {OUT_DIR}/frame_*.png")
    sim.close()


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    main()
