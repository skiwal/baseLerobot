import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"perception"/"python"))
from so100_sleeve_env import SO100SleeveEnv, Z_AXIS
from cart_assembly import drive_cart_to, drive_cart_holding, park_arm, PICKUP_REL_XY, ASSEMBLY_DOCK
from grasp_perceived import grasp_sleeve_perceived
import cart_transport as CT

for seed in [1,2]:
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    env.reset(tilt_deg=35.0, azim_deg=0.0, rng=np.random.default_rng(seed), sleeve_xy_range=CT.SLEEVE_RANGE)
    env.set_cart_pose(0.45,0,0); park_arm(env)
    drive_cart_to(env, *ASSEMBLY_DOCK)
    sp = CT._sleeve_xy_corrected(env); dock = sp[:2]-PICKUP_REL_XY
    for it in range(3):
        drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
        sp2 = CT._sleeve_xy_corrected(env); nd = sp2[:2]-PICKUP_REL_XY
        if np.linalg.norm(nd-dock)<0.003: dock=nd; break
        dock=nd
    drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
    tp=env.sleeve_pose()[0]; rel=tp[:2]-np.array(dock[:2])
    print(f"seed{seed} 停靠位{np.round(dock,4)} 真实套筒{np.round(tp,4)} 相对{np.round(rel,4)} (目标[0.12,-0.235]) 偏差{np.round((rel-PICKUP_REL_XY)*1000,1)}mm")
    grasp_sleeve_perceived(env, verbose=False)
    sa = env.jaw_rot() @ env.grasp_axis_local
    print(f"   抬起后 held套筒轴{np.round(sa,3)} 与竖直Z夹角{np.degrees(np.arccos(min(1,abs(sa@Z_AXIS)))):.1f}° held位置{np.round(env.grasp_point(),4)}")
    env.close()
