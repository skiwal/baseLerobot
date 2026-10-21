#!/usr/bin/env python
"""感知接口: 给定 SO100SleeveEnv, 仅用 RGB-D 相机输入 + 背景差分(感知预处理),
运行 C 感知程序, 返回第一步结果(套筒/杆 6D 位姿)。

注意: 本模块**不读取**套筒/杆的 Ground-Truth 位姿。背景差分所需的"空场景深度"
通过把目标临时移远再渲染得到(等同真实部署时离线拍一张空背景), 随后还原场景状态,
该过程不向抓取策略泄露任何目标位置信息。
"""
from pathlib import Path
import re
import subprocess

import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent            # perception/python
PROJ = HERE.parent                                 # perception/
DATA = PROJ / "data"
BUILD = PROJ / "build" / "perceive"


def perceive_scene(env, cam: str = "base_rgbd", W: int = 640, H: int = 480) -> dict:
    """渲染 RGB-D -> 背景差分 -> C 感知, 返回 {'sleeve':{pos,axis,centroid}, 'rod':{...}}。"""
    DATA.mkdir(parents=True, exist_ok=True)

    # 1) 含目标: RGB + 深度
    rgb, depth = env.render_rgbd(cam)
    depth_mm = np.where(np.isfinite(depth), np.clip(depth * 1000.0, 0, 65535), 0).astype(np.uint16)

    # 2) 背景深度: 把套筒+杆临时移远再渲染, 之后用整段 qpos 快照 + 静态托座变换还原(不读 GT)
    sl = env.sleeve_qadr
    qpos_full = env.data.qpos.copy()
    # set_rod_pose 会改写静态 rest_body(model.body_pos/quat), 仅还原 qpos 不够 -> 一并存/还原
    rest_pos = env.model.body_pos[env.rest_body].copy()
    rest_quat = env.model.body_quat[env.rest_body].copy()
    env.data.qpos[sl:sl + 7] = [10.0, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0]
    env.set_rod_pose(35.0, 0.0, base=np.array([10.0, 10.0, 10.0]))  # 杆移到远处
    mujoco.mj_forward(env.model, env.data)
    _, depth_bg = env.render_rgbd(cam)
    depth_bg_mm = np.where(np.isfinite(depth_bg), np.clip(depth_bg * 1000.0, 0, 65535), 0).astype(np.uint16)
    env.data.qpos[:] = qpos_full                                # 还原自由体(套筒)
    env.model.body_pos[env.rest_body] = rest_pos                 # 还原静态托座/杆
    env.model.body_quat[env.rest_body] = rest_quat
    mujoco.mj_forward(env.model, env.data)

    # 3) 写 raw 交给 C 程序
    def save(path, arr):
        with open(path, "wb") as f:
            if arr.dtype == np.uint16:
                f.write(arr.astype("<u2").tobytes())
            else:
                f.write(arr.astype(np.uint8).tobytes())

    save(DATA / "rgb.bin", rgb)
    save(DATA / "depth.bin", depth_mm)
    save(DATA / "bg.bin", depth_bg_mm)
    # 动态相机外参(pos/right/up/back) + 视场角, 支持移动底盘(相机随臂根体平移+偏航)
    cpos, cright, cup, cback = env.camera_extrinsics(cam)
    cid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    fovy_deg = float(env.model.cam_fovy[cid])
    (DATA / "meta.txt").write_text(
        f"{W} {H} {fovy_deg:.6f}\n"
        + " ".join(f"{v:.6f}" for v in cpos) + "\n"
        + " ".join(f"{v:.6f}" for v in cright) + "\n"
        + " ".join(f"{v:.6f}" for v in cup) + "\n"
        + " ".join(f"{v:.6f}" for v in cback) + "\n")

    if not BUILD.exists():
        raise RuntimeError(f"找不到 C 感知程序 {BUILD}, 请先在 perception/ 里 cmake 构建")
    proc = subprocess.run(
        [str(BUILD), str(DATA / "rgb.bin"), str(DATA / "depth.bin"),
         str(DATA / "bg.bin"), str(DATA / "meta.txt"), str(DATA / "result.bin")],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("C 感知程序失败:\n" + proc.stderr)

    out = {}
    m = re.search(r"SLEEVE pos=\(([^)]+)\) axis=\(([^)]+)\)", proc.stdout)
    mc = re.search(r"SLEEVE_RAW centroid=\(([^)]+)\)", proc.stdout)
    if m:
        out["sleeve"] = {
            "pos": np.array([float(x) for x in m.group(1).split(",")]),
            "axis": np.array([float(x) for x in m.group(2).split(",")]),
            "centroid": (np.array([float(x) for x in mc.group(1).split(",")])
                         if mc else out["sleeve"]["pos"]),
        }
    m = re.search(r"ROD\s+base=\(([^)]+)\) tip=\(([^)]+)\) axis=\(([^)]+)\)", proc.stdout)
    if m:
        out["rod"] = {
            "base": np.array([float(x) for x in m.group(1).split(",")]),
            "tip": np.array([float(x) for x in m.group(2).split(",")]),
            "axis": np.array([float(x) for x in m.group(3).split(",")]),
        }
    return out
