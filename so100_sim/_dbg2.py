import sys, numpy as np, mujoco
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"perception"/"python"))
from so100_sleeve_env import SO100SleeveEnv
from perception_api import perceive_scene
from cart_assembly import park_arm, drive_cart_to, PICKUP_REL_XY

env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
rng = np.random.default_rng(0)
env.reset(tilt_deg=35.0, azim_deg=0.0, rng=rng, sleeve_xy_range=((0.03,0.21),(-0.30,-0.17)))
true_sp = env.sleeve_pose()[0]   # 仅调试打印真值
print("随机放置后 真实套筒位 =", np.round(true_sp,4))

# 模拟 run_once: 起点远 -> 勘测位(0,0,0) -> 感知
env.set_cart_pose(0.45,0,0); park_arm(env)
drive_cart_to(env, 0,0,0)
det = perceive_scene(env, "base_rgbd")
sp = np.asarray(det["sleeve"]["pos"]) if "sleeve" in det else None
print("勘测位(0,0,0) 感知套筒 =", np.round(sp,4) if sp is not None else "未检出")
if sp is not None:
    print("  感知-真实 误差 =", np.round(sp-true_sp,4), " |.|=", np.round(np.linalg.norm(sp-true_sp)*1000,1),"mm")
    print("  抓取停靠位 =", np.round([sp[0]-PICKUP_REL_XY[0], sp[1]-PICKUP_REL_XY[1]],4))

# 开到抓取停靠位后再感知一次(抓取函数内部会这么做)
dock = (float(sp[0]-PICKUP_REL_XY[0]), float(sp[1]-PICKUP_REL_XY[1]), 0.0)
drive_cart_to(env, *dock)
det2 = perceive_scene(env, "base_rgbd")
sp2 = np.asarray(det2["sleeve"]["pos"]) if "sleeve" in det2 else None
print("抓取停靠位 感知套筒 =", np.round(sp2,4) if sp2 is not None else "未检出")
print("  真实套筒位(未动)  =", np.round(env.sleeve_pose()[0],4))
print("  套筒相对停靠位    =", np.round(env.sleeve_pose()[0][:2]-np.array(dock[:2]),4), "(目标≈[0.12,-0.235])")
