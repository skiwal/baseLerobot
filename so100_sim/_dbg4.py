import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"perception"/"python"))
from so100_sleeve_env import SO100SleeveEnv
from perception_api import perceive_scene
import mujoco
from cart_assembly import park_arm, PICKUP_REL_XY

env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
# 在若干已知套筒位置, 从"典型停靠位姿"感知, 量 感知-真实 偏差
tests = [(0.06,-0.22),(0.10,-0.22),(0.08,-0.24),(0.12,-0.24),(0.05,-0.25),(0.13,-0.20)]
print("已知套筒(x,y)      停靠位           感知(x,y)        偏差dx,dy (mm)")
for (sx,sy) in tests:
    dock=(sx-PICKUP_REL_XY[0], sy-PICKUP_REL_XY[1], 0.0)
    env.reset(tilt_deg=35.0, azim_deg=0.0, sleeve_xy=(sx,sy))
    env.set_cart_pose(*dock); park_arm(env)
    for _ in range(20): env.step(env.arm_qpos())
    det=perceive_scene(env,"base_rgbd")
    true=env.sleeve_pose()[0]
    if "sleeve" in det:
        sp=np.asarray(det["sleeve"]["pos"])
        b=(sp-true)*1000
        print(f"({sx:.2f},{sy:.2f})   ({dock[0]:+.3f},{dock[1]:+.3f})  ({sp[0]:.3f},{sp[1]:.3f})  dx={b[0]:+5.1f} dy={b[1]:+5.1f}")
    else:
        print(f"({sx:.2f},{sy:.2f})   未检出")
