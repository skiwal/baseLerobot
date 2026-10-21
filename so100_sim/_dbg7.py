import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"perception"/"python"))
from so100_sleeve_env import SO100SleeveEnv, Z_AXIS
from cart_assembly import drive_cart_to, park_arm, PICKUP_REL_XY, ASSEMBLY_DOCK
from grasp_perceived import grasp_sleeve_perceived
import cart_transport as CT

for seed in [int(x) for x in sys.argv[1:]]:
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    env.reset(tilt_deg=35.0, azim_deg=0.0, rng=np.random.default_rng(seed), sleeve_xy_range=CT.SLEEVE_RANGE)
    env.set_cart_pose(0.45,0,0); park_arm(env)
    drive_cart_to(env, *ASSEMBLY_DOCK)
    sp = CT._sleeve_xy_corrected(env); dock = sp[:2]-PICKUP_REL_XY
    for it in range(3):
        drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
        sp2 = CT._sleeve_xy_corrected(env)
        if sp2 is None: break
        nd = sp2[:2]-PICKUP_REL_XY
        if np.linalg.norm(nd-dock)<0.003: dock=nd; break
        dock=nd
    drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
    tp=env.sleeve_pose()[0]; rel=tp[:2]-np.array(dock[:2]); err=(rel-PICKUP_REL_XY)*1000
    q = env.data.qpos[env.sleeve_qadr+3:env.sleeve_qadr+7]
    zrot = np.degrees(2*np.arctan2(q[3], q[0]))
    pos_before = env.sleeve_pose()[0].copy()
    info = grasp_sleeve_perceived(env, verbose=False)
    pos_after = env.sleeve_pose()[0].copy()
    moved = np.linalg.norm(pos_after-pos_before)*1000
    print(f"seed{seed}: 偏差{np.round(err,1)}mm 绕Z={zrot:+.1f}deg 闭爪力={info['grip_force']:.1f}N 抓取中套筒移动={moved:.0f}mm")
    env.close()
