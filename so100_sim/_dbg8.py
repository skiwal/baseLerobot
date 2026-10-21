import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"perception"/"python"))
from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import drive_cart_to, park_arm, ASSEMBLY_DOCK
import cart_transport as CT

def zrot(env):
    q = env.data.qpos[env.sleeve_qadr+3:env.sleeve_qadr+7]
    ax = env.data.xmat[env.sleeve_body].reshape(3,3)[:,2]
    return np.degrees(2*np.arctan2(q[3],q[0])), np.degrees(np.arccos(min(1,abs(ax[2]))))

for seed in [10, 13, 0, 3]:
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    env.reset(tilt_deg=35.0, azim_deg=0.0, rng=np.random.default_rng(seed), sleeve_xy_range=CT.SLEEVE_RANGE)
    z0,t0 = zrot(env); p0=env.sleeve_pose()[0]
    print(f"seed{seed} reset后: 绕Z={z0:+.1f}deg 轴倾={t0:.1f}deg pos={np.round(p0,3)}")
    env.set_cart_pose(0.45,0,0); park_arm(env)
    for _ in range(20): env.step(env.arm_qpos())
    z1,t1 = zrot(env)
    print(f"   移到起点+park后: 绕Z={z1:+.1f}deg 轴倾={t1:.1f}deg")
    drive_cart_to(env, *ASSEMBLY_DOCK)
    z2,t2 = zrot(env)
    print(f"   导航到勘测位后: 绕Z={z2:+.1f}deg 轴倾={t2:.1f}deg")
    env.close()
