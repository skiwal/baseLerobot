import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"perception"/"python"))
from so100_sleeve_env import SO100SleeveEnv
from perception_api import perceive_scene
from cart_assembly import drive_cart_to, park_arm, PICKUP_REL_XY, ASSEMBLY_DOCK

seed = int(sys.argv[1]) if len(sys.argv)>1 else 3
env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
rng = np.random.default_rng(seed)
env.reset(tilt_deg=35.0, azim_deg=0.0, rng=rng, sleeve_xy_range=((0.03,0.14),(-0.255,-0.195)))
print("seed",seed,"reset后 真实套筒 =", np.round(env.sleeve_pose()[0],4))

env.set_cart_pose(0.45,0,0); park_arm(env)
drive_cart_to(env, *ASSEMBLY_DOCK)
det = perceive_scene(env,"base_rgbd"); sp=np.asarray(det["sleeve"]["pos"])
print("  勘测感知 =", np.round(sp,4), " 真实 =", np.round(env.sleeve_pose()[0],4))
dock=(float(sp[0]-PICKUP_REL_XY[0]), float(sp[1]-PICKUP_REL_XY[1]),0.0)
print("  算得停靠位 =", np.round(dock,3))
drive_cart_to(env, *dock)
tp=env.sleeve_pose()[0]
print("  到停靠位后 真实套筒 =", np.round(tp,4), " 相对停靠位 =", np.round(tp[:2]-np.array(dock[:2]),4), "(目标[0.12,-0.235])")
det2=perceive_scene(env,"base_rgbd")
if "sleeve" in det2:
    print("  停靠位新感知 =", np.round(np.asarray(det2["sleeve"]["pos"]),4))
else:
    print("  停靠位未检出套筒")
