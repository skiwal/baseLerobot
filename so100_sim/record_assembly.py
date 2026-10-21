#!/usr/bin/env python
"""录制底盘自主装配全过程(closeup 第三人称全景) -> out/assembly.mp4。
用法: python record_assembly.py [start_x start_y start_yaw dock_x dock_y dock_yaw]
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import SO100SleeveEnv, GRIP_FORCE_TARGET
from perception_api import perceive_scene
from cart_assembly import drive_cart_to, grasp_sleeve_base_indep, park_arm
from grasp_perceived import insert_sleeve_perceived


def main():
    a = [float(x) for x in sys.argv[1:]]
    start = tuple(a[0:3]) if len(a) >= 3 else (0.45, 0.0, 0.0)
    dock = tuple(a[3:6]) if len(a) >= 6 else (0.0, 0.0, 0.0)

    env = SO100SleeveEnv(width=640, height=480, camera="closeup", render=True)
    frames = []
    _step, _setarm = env.step, env.set_arm

    def _rec():
        img = env.render_image("closeup")          # RGB uint8 (H,W,3)
        frames.append(img[:, :, ::-1].copy())       # -> BGR for cv2

    def step_rec(action):
        _step(action); _rec()

    def setarm_rec(q5, jaw):
        _setarm(q5, jaw); _rec()

    env.step, env.set_arm = step_rec, setarm_rec

    env.reset(tilt_deg=35.0, azim_deg=0.0)
    env.set_cart_pose(*start); park_arm(env); _rec()
    print(f"[录制] 起点={start} -> 停靠位={dock}")
    drive_cart_to(env, *dock)
    det = perceive_scene(env, "base_rgbd")
    print(f"[录制] 感知: 套筒={'有' if 'sleeve' in det else '无'} 杆={'有' if 'rod' in det else '无'}")
    info = grasp_sleeve_base_indep(env, verbose=True)
    ok = False
    if info["reached"] and info["grip_force"] > GRIP_FORCE_TARGET:
        rinfo = insert_sleeve_perceived(env, info["rod_axis"], info["rod_tip"],
                                        info["rod_base"], verbose=True)
        ok = rinfo["success"]
    print(f"[录制] 抓取力={info['grip_force']:.1f}N  装配={'成功' if ok else '失败'}")

    # 抽稀到 ~30fps 观感: 目标约 900 帧
    if len(frames) > 900:
        idx = np.linspace(0, len(frames) - 1, 900).astype(int)
        frames = [frames[i] for i in idx]
    out = ROOT / "out" / "assembly.mp4"
    out.parent.mkdir(exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"[录制] 已保存 {out}  ({len(frames)} 帧, {w}x{h})")


if __name__ == "__main__":
    main()
