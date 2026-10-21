import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"perception"/"python"))
from so100_sleeve_env import SO100SleeveEnv, Z_AXIS, SLEEVE_LEN
from perception_api import perceive_scene
from cart_assembly import drive_cart_to, drive_cart_holding, park_arm, PICKUP_REL_XY, ASSEMBLY_DOCK
from grasp_perceived import grasp_sleeve_perceived
import cart_transport as CT

seed = int(sys.argv[1]) if len(sys.argv)>1 else 0
env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
env.reset(tilt_deg=35.0, azim_deg=0.0, rng=np.random.default_rng(seed), sleeve_xy_range=CT.SLEEVE_RANGE)
env.set_cart_pose(0.45,0,0); park_arm(env)

# 复刻 run_once 的伺服部分
drive_cart_to(env, *ASSEMBLY_DOCK)
sp = CT._sleeve_xy_corrected(env); dock = sp[:2]-PICKUP_REL_XY
for it in range(3):
    drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
    sp2 = CT._sleeve_xy_corrected(env); nd = sp2[:2]-PICKUP_REL_XY
    if np.linalg.norm(nd-dock)<0.003: dock=nd; break
    dock=nd
drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
print("最终停靠位", np.round(dock,4), " 真实套筒", np.round(env.sleeve_pose()[0],4),
      " 相对", np.round(env.sleeve_pose()[0][:2]-np.array(dock[:2]),4))

info = grasp_sleeve_perceived(env, verbose=False)
print("抬起后 arm", np.round(env.arm_qpos()[:5],4))
print("PROVEN_LIFT_SEED 参考 [0.5697,-2.1175,2.7828,-0.6256,-1.5702]")
held = env.grasp_point()
print("抬起后 held套筒(grasp_point)", np.round(held,4))

drive_cart_holding(env, *ASSEMBLY_DOCK)
print("搬到(0,0,0)后 cart", np.round(env.cart_pose,3), " held套筒", np.round(env.grasp_point(),4))
det = perceive_scene(env,"base_rgbd")
rod=det["rod"]; rt=np.asarray(rod["tip"]); ra=np.asarray(rod["axis"]); ra/=np.linalg.norm(ra)
pre = rt + ra*(SLEEVE_LEN/2+0.01)
print("感知杆 tip", np.round(rt,4), " 预插点pre", np.round(pre,4))
print("held套筒 到 pre 距离", np.round(np.linalg.norm(env.grasp_point()-pre)*1000,1),"mm")
