#!/usr/bin/env python
"""并行批跑"随机套筒 抓取-搬运-插装", 统计成功率。
用法: python batch_transport.py [N=100] [workers=8]
结果写入 out/batch_results.txt。
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

RESULTS = ROOT / "out" / "batch_results.txt"


def run_seed(seed):
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    ok, why = False, ""
    try:
        env.reset(tilt_deg=35.0, azim_deg=0.0, rng=np.random.default_rng(seed),
                  sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
        ok, info = CT.run_once(env, verbose=False)
        if not ok:
            why = str(info)
    except Exception as e:
        why = f"EXC {type(e).__name__}: {e}"
    finally:
        env.close()
    return seed, bool(ok), why


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.time()
    done, okc, fails = 0, 0, []
    with open(RESULTS, "w") as f:
        f.write(f"start N={N} W={W} t={t0:.0f}\n"); f.flush()
        with Pool(W) as p:
            for seed, ok, why in p.imap_unordered(run_seed, range(N)):
                done += 1; okc += int(ok)
                f.write(f"seed {seed}: {'OK' if ok else 'FAIL'}  {why}\n"); f.flush()
                if not ok:
                    fails.append(seed)
                if done % 10 == 0:
                    f.write(f"-- progress {done}/{N}  ok={okc} ({100*okc/done:.0f}%)  {time.time()-t0:.0f}s\n"); f.flush()
        rate = 100 * okc / N
        f.write(f"\nSUMMARY: {okc}/{N} = {rate:.0f}% success in {time.time()-t0:.0f}s\n")
        f.write(f"FAILS: {fails}\n")
        print(f"SUMMARY: {okc}/{N} = {rate:.0f}%  ({time.time()-t0:.0f}s)")
