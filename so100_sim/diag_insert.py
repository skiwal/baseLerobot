#!/usr/bin/env python
"""P5b: 插入站位 yaw 随杆 azim 自适应的实证搜索。
对每个 (tilt,azim) 扫一组插入站位 yaw, 用 run_once(insert_dock=(0,0,yaw)) 跑 N 种子,
报告成功率与失败 GT 横向(insert_real 的 GT 指标)。
目的: 确认 "yaw ≈ azim_rad + 0.19" 假设, 并找出各 azim 最优 yaw。

用法: python diag_insert.py [tilt] [azims逗号] [yaw_start:yaw_end:step] [nseed]
"""
import sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import park_arm
import cart_transport as CT


def run_one(args):
    tilt, azim, yaw, seed = args
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    try:
        rng = np.random.default_rng(seed)
        env.reset(tilt_deg=tilt, azim_deg=azim, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
        ok, info = CT.run_once(env, verbose=False, insert_dock=(0.0, 0.0, yaw),
                               tilt_deg=tilt, azim_deg=azim)
        if ok:
            return (tilt, azim, yaw, seed, True, 0.0, "")
        if isinstance(info, dict):
            lat = info.get("gt_lateral_mm", 0.0)
            return (tilt, azim, yaw, seed, False, float(lat),
                    f"insert_real gt_ov={info.get('gt_overlap_mm')} gt_lat={lat:.0f}")
        return (tilt, azim, yaw, seed, False, 0.0, str(info)[:60])
    except Exception as e:
        return (tilt, azim, yaw, seed, False, 0.0, f"EXC:{type(e).__name__}")
    finally:
        env.close()


def main():
    tilt = float(sys.argv[1]) if len(sys.argv) > 1 else 35.0
    azims = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "90").split(",")]
    yr = (sys.argv[3] if len(sys.argv) > 3 else "-1.5:2.6:0.3").split(":")
    ys = float(yr[0]); ye = float(yr[1]); ystep = float(yr[2])
    yaws = [round(y, 2) for y in np.arange(ys, ye + 1e-9, ystep)]
    nseed = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    W = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    jobs = [(tilt, a, y, s) for a in azims for y in yaws for s in range(nseed)]
    res = []
    with Pool(W) as p:
        for r in p.imap_unordered(run_one, jobs):
            res.append(r)
    for a in azims:
        print(f"\n=== tilt={tilt:.0f} azim={a:.0f} (yaw 扫描, 每 yaw {nseed} 种子) ===")
        for y in yaws:
            rs = [r for r in res if r[1] == a and abs(r[2] - y) < 1e-6]
            n = len(rs); ok = sum(1 for r in rs if r[4])
            lats = [r[5] for r in rs if not r[4] and r[5] > 0]
            avg_lat = (sum(lats) / len(lats)) if lats else 0.0
            mark = " <===" if ok == n and n > 0 else ""
            print(f"  yaw={y:+.2f}: {100*ok/n:>3.0f}% (n{n}) 失败GT横向均={avg_lat:.0f}mm{mark}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[{time.time()-t0:.0f}s] done")
