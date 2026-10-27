#!/usr/bin/env python
"""P5a 感知失效归因 + 勘测位姿搜索。
对每个 (tilt,azim) 把杆真实 tip/base 投影进 base 相机视锥, 并运行 perceive_scene,
区分 (a) 杆在视锥外(-> 改勘测小车位姿可解) vs (b) 视野内但 C 感知失效(-> 需感知算法改动)。
可在不同 survey_dock 下重测, 找能覆盖所有杆姿的勘测位姿。

用法: python diag_cam.py [tilts逗号] [azims逗号] [survey_x,y,yaw逗号|auto]
auto = 自动扫一组候选勘测位姿
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


def project(cam_pos, right, up, back, P):
    rel = np.asarray(P, float) - cam_pos
    depth = -float(rel @ back)          # 相机看向 -back
    return depth, float(rel @ right), float(rel @ up)


def check_pose(env, tilt, azim, survey_dock):
    drive_cart_to(env, *survey_dock)
    cam_pos, r, u, b = env.camera_extrinsics("base_rgbd")
    cid = env.cam_rgbd_id
    fovy = float(env.model.cam_fovy[cid])
    W, H = 640, 480
    tan_v = np.tan(np.radians(fovy / 2.0)); tan_h = (W / H) * tan_v
    rt = env.rod_tip(); rb = env.rod_base()
    fov_any = False
    for P in (rt, rb):
        depth, xc, yc = project(cam_pos, r, u, b, P)
        inv = depth > 0.01 and abs(xc / depth) < tan_h and abs(yc / depth) < tan_v
        fov_any = fov_any or inv
    det = perceive_scene(env, "base_rgbd", expected_tilt_deg=tilt, expected_azim_deg=azim)
    has = "rod" in det
    perr = float(np.linalg.norm(det["rod"]["base"] - rb)) if has else None
    return fov_any, has, perr


def main():
    tilts = [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "20,35,50").split(",")]
    azims = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,90,180,270").split(",")]
    sd = sys.argv[3] if len(sys.argv) > 3 else "auto"
    if sd == "auto":
        docks = [(0.0, 0.0, 0.0), (0.0, -0.05, 0.0), (0.0, -0.1, 0.0),
                 (-0.05, -0.05, 0.0), (0.05, -0.05, 0.0),
                 (0.0, 0.0, 0.3), (0.0, -0.05, 0.3), (0.0, -0.1, 0.3)]
    else:
        xx = [float(x) for x in sd.split(",")]
        docks = [(xx[0], xx[1], xx[2] if len(xx) > 2 else 0.0)]
    for t in tilts:
        for a in azims:
            env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
            try:
                rng = np.random.default_rng(int(t * 10 + a))
                env.reset(tilt_deg=t, azim_deg=a, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
                env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
                cells = []
                for d in docks:
                    fov, has, perr = check_pose(env, t, a, d)
                    cells.append(f"{tuple(round(x,2) for x in d)}:fov={int(fov)}rod={int(has)}"
                                 + (f"e{perr*1000:.0f}" if perr is not None else ""))
                print(f"tilt={t:>4.0f} azim={a:>4.0f} | " + "  ".join(cells))
            except Exception as e:
                print(f"tilt={t:>4.0f} azim={a:>4.0f} | EXC {type(e).__name__}: {e}")
            finally:
                env.close()


if __name__ == "__main__":
    main()
