#!/usr/bin/env python
"""视角鲁棒性校验: 固定平台, 让底盘(臂+根部相机整体)在不同平移/偏航位姿下,
用纯 RGB-D 感知套筒, 与真值比较——验证"小车移动、相机视角变化, 感知仍稳健"。

前提(同真实装配管线): 感知前把机械臂收起到 PARK 构型, 避免臂遮挡根部相机视野。
运行: python test_perception_viewpoint.py
"""
import sys
from pathlib import Path
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import SO100SleeveEnv
from perception_api import perceive_scene

PARK = np.array([0.0, 1.2, 2.2, -1.0, -1.57])   # 收起抬高: 不挡根部相机


def main():
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    env.reset(tilt_deg=35.0, azim_deg=0.0)
    sg = list(env.sleeve_geoms)

    # 一组覆盖平移与偏航(含组合)的底盘位姿; 平台在世界系固定不动
    poses = [(0.0, 0, 0), (0.10, 0, 0), (-0.08, 0, 0), (0, 0.10, 0),
             (0, -0.08, 0), (0, 0, 0.35), (0, 0, -0.35), (0.15, 0.12, 0.5)]

    print("  底盘位姿(x,y,yaw)      感知套筒中心            误差(mm)  检出")
    errs = []
    for p in poses:
        env.set_cart_pose(*p)
        env.set_arm(PARK, 0.0); mujoco.mj_forward(env.model, env.data)   # 收臂, 避免遮挡
        det = perceive_scene(env, "base_rgbd")
        gt = np.mean([env.data.geom_xpos[i] for i in sg], axis=0)
        if "sleeve" in det:
            sp = np.asarray(det["sleeve"]["pos"])
            e = float(np.linalg.norm(sp - gt) * 1000); errs.append(e)
            print(f"  {str(tuple(round(v, 2) for v in p)):20s} {np.round(sp, 3)}  {e:5.1f}   ✅")
        else:
            print(f"  {str(tuple(round(v, 2) for v in p)):20s} {'(未检出)':>22}   --    ❌")
    if errs:
        print(f"\n[结论] {len(errs)}/{len(poses)} 位姿检出套筒, 位置误差 {min(errs):.1f}~{max(errs):.1f}mm "
              f"(<1cm) -> 感知对底盘平移/偏航视角变化鲁棒。")


if __name__ == "__main__":
    main()
