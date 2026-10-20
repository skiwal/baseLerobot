#!/usr/bin/env python
"""从根部 RGB-D 相机(base_rgbd)采样一帧, 校验套筒/杆是否可见, 并保存 RGB + 深度图。

用法:
    /home/ubuntu/anaconda3/envs/lerobot/bin/python so100_sim/capture_base_cam.py

输出(so100_sim/out/):
    base_rgbd_rgb.png        RGB
    base_rgbd_depth.png      归一化灰度深度
    base_rgbd_depth.npy      原始深度(米)

可见性判定用 MuJoCo 分割渲染(geom 是否出现在画面), 不依赖投影/深度一致性。
臂用"缩回抬高"姿态, 模拟"机械臂不伸出去"时相机看工作区。
"""
import math
from pathlib import Path

import numpy as np
import mujoco

from so100_sleeve_env import SO100SleeveEnv, SLEEVE_HOME, ROD_HOME

W, H = 640, 480
CAM = "base_rgbd"

# 缩回抬高姿态([Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll]), 让臂不挡相机视线
RETRACT_Q5 = np.array([0.0, 1.2, 2.2, -1.0, -1.57])


def project(model, cam_name, point, W, H):
    """把世界点投影到相机画面 (u, v), 并返回是否在相机前方。"""
    cid = model.camera(cam_name).id
    cam = model.camera(cid)
    pos = cam.pos
    R = cam.mat0.reshape(3, 3)       # 静态相机-世界旋转, 列 = 相机各轴(右/上/后)
    fovy = float(cam.fovy.item())    # 度数
    d = np.asarray(point, float) - pos
    x_c = R[:, 0] @ d               # 右
    y_c = R[:, 1] @ d               # 上
    z_c = R[:, 2] @ d               # 后(相机看向 -z)
    in_front = z_c < 0
    fx = (W / 2) / math.tan(math.radians(fovy) / 2)
    fy = (H / 2) / math.tan(math.radians(fovy) / 2)
    u = W / 2 + fx * (x_c / -z_c)
    v = H / 2 - fy * (y_c / -z_c)
    return u, v, in_front


def render_seg(env, cam):
    r = env.renderer
    r.enable_segmentation_rendering()
    r.update_scene(env.data, cam)
    seg = r.render()
    r.disable_segmentation_rendering()
    return seg


def main():
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)

    env = SO100SleeveEnv(width=W, height=H, camera=CAM, render=True)

    # 臂缩回(抬高让开)姿态: 放好套筒与杆, 臂抬高不挡相机视线
    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[env.sleeve_qadr:env.sleeve_qadr + 3] = SLEEVE_HOME
    env.data.qpos[env.sleeve_qadr + 3:env.sleeve_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    env.set_rod_pose(35.0, 0.0)
    env.set_arm(RETRACT_Q5, 1.4)
    mujoco.mj_forward(env.model, env.data)

    # 1) RGB-D
    rgb, depth = env.render_rgbd(CAM)

    # 2) 分割渲染 -> 判定套筒/杆是否出现在画面
    seg = render_seg(env, CAM)
    seg_ids = set(seg.ravel().tolist())
    sleeve_ids = {g + 1 for g in env.sleeve_geoms}  # 分割值 = geom_id + 1
    rod_ids = {g + 1 for g in env.rod_geoms}

    for name, ids in (("套筒", sleeve_ids), ("杆座", rod_ids)):
        n = len(ids & seg_ids)
        # 也在画面内投影一次, 仅作参考
        u, v, front = project(env.model, CAM, SLEEVE_HOME if name == "套筒" else ROD_HOME, W, H)
        in_frame = (0 <= u <= W) and (0 <= v <= H) and front
        print(f"  {name}: 出现在画面(分割)={n > 0} 投影在框内={in_frame} 像素({u:.0f},{v:.0f})")

    # 3) 保存
    try:
        from PIL import Image
        Image.fromarray(rgb).save(out / "base_rgbd_rgb.png")
    except Exception as e:                       # noqa: BLE001
        print("  保存 RGB 失败:", e)

    d = depth.copy()
    valid = np.isfinite(d) & (d < 2.0)
    if valid.any():
        lo, hi = np.percentile(d[valid], 1), np.percentile(d[valid], 99)
        dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        dn = np.zeros_like(d)
    dn[~valid] = 0
    gray = (dn * 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(gray).save(out / "base_rgbd_depth.png")
    except Exception as e:                       # noqa: BLE001
        print("  保存深度失败:", e)
    np.save(out / "base_rgbd_depth.npy", depth)

    if valid.any():
        print(f"  深度: 有效像素 {valid.mean()*100:.0f}%, "
              f"范围 {d[valid].min():.3f}~{d[valid].max():.3f}m, "
              f"中位 {np.median(d[valid]):.3f}m")
    print(f"  已保存: {out/'base_rgbd_rgb.png'}")
    print(f"           {out/'base_rgbd_depth.png'}")
    print(f"           {out/'base_rgbd_depth.npy'}")


if __name__ == "__main__":
    main()
