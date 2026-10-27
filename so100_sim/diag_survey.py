#!/usr/bin/env python
"""P5a 通用勘测位姿搜索: 在候选小车 dock 网格上, 对每个 dock 跑全部 (tilt,azim) 杆姿,
找"全检出 + 最大感知误差最小"的通用 SURVEY_DOCK, 用于替代 ASSEMBLY_DOCK 做杆勘测
(套筒感知仍在 ASSEMBLY_DOCK, 互不影响)。

用法: python diag_survey.py
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import park_arm, drive_cart_to
import cart_transport as CT
from perception_api import perceive_scene

TILTS = [20.0, 35.0, 50.0]
AZIMS = [0.0, 90.0, 180.0, 270.0]
XS = [-0.03, 0.0]
YS = [-0.16, -0.12, -0.08]
YAWS = [0.2, 0.4]


def eval_dock(env, t, a, dock):
    drive_cart_to(env, *dock)
    det = perceive_scene(env, "base_rgbd", expected_tilt_deg=t, expected_azim_deg=a)
    if "rod" not in det:
        return None
    return float(np.linalg.norm(det["rod"]["base"] - env.rod_base()))


def main():
    # 预生成每杆姿的 rng 种子(固定, 隔离变量)
    seed_of = {t: {a: int(t * 10 + a) for a in AZIMS} for t in TILTS}
    results = []
    npose = len(TILTS) * len(AZIMS)
    for x in XS:
        for y in YS:
            for yaw in YAWS:
                dock = (x, y, yaw)
                cov = 0; maxerr = 0.0; errs = []
                ok_all = True
                for t in TILTS:
                    for a in AZIMS:
                        env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
                        try:
                            env.reset(tilt_deg=t, azim_deg=a, rng=np.random.default_rng(seed_of[t][a]),
                                      sleeve_xy_range=CT.SLEEVE_RANGE)
                            env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
                            e = eval_dock(env, t, a, dock)
                        except Exception:
                            e = None
                        finally:
                            env.close()
                        if e is None:
                            ok_all = False
                        else:
                            cov += 1; maxerr = max(maxerr, e); errs.append((t, a, e))
                results.append((dock, cov, maxerr, ok_all))
                print(f"dock={dock}: cov={cov}/{npose} maxerr={maxerr*1000:.0f}mm all={ok_all}")
    print("\n=== 全检出(ok_all)候选, 按 maxerr 升序 ===")
    cands = sorted([r for r in results if r[3]], key=lambda r: r[2])
    for dock, cov, maxerr, _ in cands[:8]:
        print(f"  {dock}  maxerr={maxerr*1000:.0f}mm")
    if not cands:
        print("  无全检出候选, 需扩大网格")


if __name__ == "__main__":
    main()
