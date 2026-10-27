#!/usr/bin/env python
"""P5b 归因: 各 azim 下 C 感知的杆 base/axis/tip 误差, 区分
azim=90 的 50-130mm 插飞是 (a) 感知杆轴方向错 -> 预插位指错, 还是 (b) 插入几何/运动学不随杆姿自适应。
用法: python diag_axis.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import park_arm, drive_cart_to, ASSEMBLY_DOCK
import cart_transport as CT
from perception_api import perceive_scene

print("azim | base_err  axis_angle_err(deg)  tip_err  detected")
for az in [0.0, 45.0, 90.0, 135.0, 180.0, 270.0]:
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    try:
        rng = np.random.default_rng(int(35 * 10 + az))
        env.reset(tilt_deg=35.0, azim_deg=az, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env); drive_cart_to(env, *ASSEMBLY_DOCK)
        det = perceive_scene(env, "base_rgbd", expected_tilt_deg=35.0, expected_azim_deg=az)
        if "rod" not in det:
            print(f"{az:>4.0f} | (未检出)"); continue
        pb = np.asarray(det["rod"]["base"], float)
        pa = np.asarray(det["rod"]["axis"], float); pa /= np.linalg.norm(pa)
        pt = np.asarray(det["rod"]["tip"], float)
        gtb = env.rod_base(); gta = env.rod_axis(); gtt = env.rod_tip()
        be = np.linalg.norm(pb - gtb)
        ae = np.degrees(np.arccos(np.clip(abs(float(np.dot(pa, gta))), -1, 1)))
        te = np.linalg.norm(pt - gtt)
        print(f"{az:>4.0f} | {be*1000:6.1f}mm  {ae:6.2f}            {te*1000:6.1f}mm  yes")
    finally:
        env.close()
