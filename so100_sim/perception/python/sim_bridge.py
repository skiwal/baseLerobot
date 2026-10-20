#!/usr/bin/env python
"""仿真桥接: 用 MuJoCo 渲染根部 RGB-D 相机(base_rgbd)的
   rgb / depth / 背景深度(无套筒无杆), 写成 raw 文件交给 C 感知程序处理,
   再把高亮结果图转成 PNG 存到 so100_sim/out/。

运行:
    MUJOCO_GL=egl python perception/python/sim_bridge.py

依赖: mujoco, numpy, Pillow (均在 lerobot 环境)
"""
from pathlib import Path
import sys, subprocess, struct

import numpy as np
import mujoco

try:
    from PIL import Image
except Exception as e:  # noqa
    raise SystemExit("需要 Pillow: pip install pillow") from e

# 让 import 到 so100_sleeve_env (本文件在 so100_sim/perception/python/ -> 上两层是 so100_sim/)
ROOT = Path(__file__).resolve().parent.parent.parent  # so100_sim/
sys.path.insert(0, str(ROOT))
from so100_sleeve_env import SO100SleeveEnv, SLEEVE_HOME, ROD_HOME  # noqa: E402

W, H = 640, 480
CAM = "base_rgbd"
# 缩回抬高姿态: 臂不挡视线
RETRACT_Q5 = np.array([0.0, 1.2, 2.2, -1.0, -1.57])

HERE = Path(__file__).resolve().parent          # perception/
DATA = HERE / "data"
OUT = ROOT / "out"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)


def save_raw(path, arr: np.ndarray):
    """arr: uint8 (H,W,3) 或 uint16 (H,W)。小端, 行主序。"""
    with open(path, "wb") as f:
        if arr.dtype == np.uint16:
            f.write(arr.astype("<u2").tobytes())
        else:
            f.write(arr.astype(np.uint8).tobytes())


def main():
    env = SO100SleeveEnv(width=W, height=H, camera=CAM, render=True)

    # 放好套筒+杆, 臂缩回
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env.sleeve_qadr:env.sleeve_qadr + 3] = SLEEVE_HOME
    env.data.qpos[env.sleeve_qadr + 3:env.sleeve_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    env.set_rod_pose(35.0, 0.0)
    env.set_arm(RETRACT_Q5, 1.4)
    mujoco.mj_forward(env.model, env.data)

    # 1) 含目标的 RGB + 深度
    rgb, depth = env.render_rgbd(CAM)            # rgb(H,W,3) uint8; depth(H,W) float32 米
    depth_mm = np.where(np.isfinite(depth), np.clip(depth * 1000.0, 0, 65535), 0).astype(np.uint16)

    # 2) 背景深度: 把套筒/杆移到远处再渲染(臂保持不动 -> 臂在差分中抵消)
    env.data.qpos[env.sleeve_qadr:env.sleeve_qadr + 3] = [10.0, 10.0, 10.0]
    env.set_rod_pose(35.0, 0.0, base=np.array([10.0, 10.0, 10.0]))
    mujoco.mj_forward(env.model, env.data)
    _, depth_bg = env.render_rgbd(CAM)
    depth_bg_mm = np.where(np.isfinite(depth_bg), np.clip(depth_bg * 1000.0, 0, 65535), 0).astype(np.uint16)

    # 写 raw
    save_raw(DATA / "rgb.bin",    rgb)
    save_raw(DATA / "depth.bin",  depth_mm)
    save_raw(DATA / "bg.bin",     depth_bg_mm)
    (DATA / "meta.txt").write_text(f"{W} {H}\n")

    # 3) 调 C 感知程序
    exe = HERE.parent / "build" / "perceive"
    if not exe.exists():
        raise SystemExit(f"找不到 {exe}, 请先 cmake 构建(见 README)")
    proc = subprocess.run([str(exe),
                           str(DATA / "rgb.bin"), str(DATA / "depth.bin"),
                           str(DATA / "bg.bin"), str(DATA / "meta.txt"),
                           str(DATA / "result.bin")],
                          capture_output=True, text=True)
    print("=== C 感知程序输出 ===")
    print(proc.stdout.strip())
    if proc.returncode != 0:
        print("stderr:", proc.stderr.strip())
        raise SystemExit("C 感知程序执行失败")

    # 4) 转 PNG
    buf = (DATA / "result.bin").read_bytes()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
    Image.fromarray(arr).save(OUT / "perception_result.png")
    print(f"\n已保存高亮效果图: {OUT / 'perception_result.png'}")

    # 顺便也存一张原图对比
    Image.fromarray(rgb).save(OUT / "perception_raw.png")
    print(f"原始图(对比):     {OUT / 'perception_raw.png'}")


if __name__ == "__main__":
    main()
