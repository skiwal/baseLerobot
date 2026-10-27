#!/usr/bin/env python
"""P4 泛化审计: 扫细杆倾角(tilt) × 方位(azim), 量化脚本控制器在固定 35°/0° 之外的真实成功率。

为什么需要: 当前 100% 是在固定 tilt=35/azim=0 + 窄中心区套筒摆放下调出来的; 多个标定常数
(ROD_BASE_AXIAL_CALIB / INSERTION_DOCK 偏航 / PICKUP_REL_XY / 感知兜底标称) 都针对 35/0 校准。
本脚本把实际 tilt/azim 透传给感知兜底与 env.reset, 扫出真实成功率曲面, 暴露过拟合。

用法: python diag_gen.py [tilts逗号] [azims逗号] [nseed=24] [workers=8]
例:   python diag_gen.py 20,35,50 0,90,180,270 24 8
结果写入 out/diag_gen.txt 并打印汇总。
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


def run_seed_ta(args):
    seed, tilt, azim = args
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    ok, cat, detail = False, "exception", ""
    try:
        rng = np.random.default_rng(seed)
        env.reset(tilt_deg=tilt, azim_deg=azim, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
        ok, info = CT.run_once(env, verbose=False, tilt_deg=tilt, azim_deg=azim)
        if ok:
            cat = "success"
        elif isinstance(info, str):
            if "抓取" in info:
                cat = "grasp"
            elif "未感知" in info or "细杆" in info:
                cat = "perc_missing"
            else:
                cat = "other"
            detail = info[:120]
        elif isinstance(info, dict):
            cat = "insert_perc" if info.get("gt_success") else "insert_real"
            detail = (f"gt_ov={info.get('gt_overlap_mm')} perc_ov={info.get('insert_overlap_mm')} "
                      f"gt_lat={info.get('gt_lateral_mm')} gt_align={info.get('gt_align_deg')}")
    except Exception as e:
        cat = "exc"; detail = f"{type(e).__name__}: {e}"[:160]
    finally:
        env.close()
    return (seed, tilt, azim, ok, cat, detail)


def main():
    tilts = [float(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "20,35,50").split(",")]
    azims = [float(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,90,180,270").split(",")]
    nseed = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    W = int(sys.argv[4]) if len(sys.argv) > 4 else 8

    out = ROOT / "out" / "diag_gen.txt"
    out.parent.mkdir(exist_ok=True)
    t0 = time.time()
    jobs = [(s, t, a) for t in tilts for a in azims for s in range(nseed)]
    results = []
    with open(out, "w") as f:
        f.write(f"P4 gen sweep tilts={tilts} azims={azims} nseed={nseed} W={W}\n"); f.flush()
        with Pool(W) as p:
            for r in p.imap_unordered(run_seed_ta, jobs):
                results.append(r)
                f.write(f"tilt={r[1]:.0f} azim={r[2]:.0f} seed={r[0]}: {r[4]} "
                        f"{'OK' if r[3] else 'FAIL'} {r[5]}\n"); f.flush()
        # 汇总
        cats = ("success", "grasp", "insert_real", "insert_perc", "perc_missing", "other", "exc")
        print(f"\n{'tilt\\azim':>9} | " + " | ".join(f"a={a:>6}" for a in azims) + " | 行汇总")
        for t in tilts:
            row = []
            for a in azims:
                rs = [r for r in results if r[1] == t and r[2] == a]
                n = len(rs); okc = sum(1 for r in rs if r[3])
                row.append(f"{100*okc/n:>3.0f}%(n{n})" if n else "  -  ")
            rs_all = [r for r in results if r[1] == t]
            na = len(rs_all); oka = sum(1 for r in rs_all if r[3])
            print(f"{t:>9} | " + " | ".join(row) + f" | {100*oka/na:>3.0f}%(n{na})")
        # 列汇总
        col_summary = []
        for a in azims:
            rs = [r for r in results if r[2] == a]
            n = len(rs); okc = sum(1 for r in rs if r[3])
            col_summary.append(f"a={a}:{100*okc/n:.0f}%(n{n})")
        print("列汇总: " + "  ".join(col_summary))
        print(f"\n全局: {sum(1 for r in results if r[3])}/{len(results)} "
              f"= {100*sum(1 for r in results if r[3])/len(results):.0f}%")
        # 失败类别分布
        from collections import Counter
        cc = Counter(r[4] for r in results if not r[3])
        print("失败类别分布(非成功): " + ", ".join(f"{k}={v}" for k, v in cc.items()))
        # 每格失败类别细分
        print("\n--- 每格失败细分 (tilt,azim: 类别=次数) ---")
        for t in tilts:
            for a in azims:
                rs = [r for r in results if r[1] == t and r[2] == a and not r[3]]
                if not rs:
                    continue
                sub = Counter(r[4] for r in rs)
                print(f"  tilt={t:.0f} azim={a:.0f}: " +
                      ", ".join(f"{k}={v}" for k, v in sub.items()))
        print(f"\n[{time.time()-t0:.0f}s] 明细见 {out}")


if __name__ == "__main__":
    main()
