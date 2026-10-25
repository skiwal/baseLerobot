#!/usr/bin/env python
"""P2.5 接近几何重摆: 在"插入站位"上扫小车偏航/平移, 看能否收付深几何楔死(重试救不回的硬核)。

原理: 杆在世界系固定, 同一世界系插入目标(pre/推入方向)在不同小车站位下会解出不同关节构型,
推入期沿杆轴的漂移/楔死特性随之改变。只改"夹持后开往的装配位"(run_once(insert_dock=)),
感知勘测位保持已验证的 (0,0,0), 避免把杆移出视野。

用法: MUJOCO_GL=egl python diag_approach.py
输出: 每种子 x 每站位 的成功/GT横向, 及每站位对"硬核种子"的收复计数。
"""
import sys, time, io, contextlib
from pathlib import Path
import numpy as np
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "perception" / "python"))
from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import park_arm
import cart_transport as CT

# 硬核楔死种子(重试救不回) —— 扫站位看能否收复
SEEDS = [5, 18, 59, 84, 97, 1, 98, 26]
# 站位扫描: 更细偏航网格找最佳单点
DOCKS = [
    (0.0, 0.0, 0.10), (0.0, 0.0, 0.13), (0.0, 0.0, 0.16),
    (0.0, 0.0, 0.19), (0.0, 0.0, 0.22), (0.0, 0.0, 0.25),
    (0.0, 0.0, 0.28), (0.0, 0.0, 0.31), (0.0, 0.0, 0.34),
]


def run_one(args):
    seed, dock = args
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    res = {"seed": seed, "dock": dock, "ok": False, "gt_succ": None,
           "gt_lat": None, "gt_ov": None, "msg": ""}
    try:
        rng = np.random.default_rng(seed)
        env.reset(tilt_deg=35.0, azim_deg=0.0, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
        ok, info = CT.run_once(env, verbose=False, insert_dock=dock)
        res["ok"] = bool(ok)
        if isinstance(info, dict):
            res["gt_succ"] = info.get("gt_success")
            res["gt_lat"] = info.get("gt_lateral_mm")
            res["gt_ov"] = info.get("gt_overlap_mm")
        else:
            res["msg"] = str(info)
    except Exception as e:
        res["msg"] = f"EXC {type(e).__name__}: {e}"
    finally:
        env.close()
    return res


def summarize(results):
    hard = set(SEEDS)
    print(f"\n{'dock(x,yaw)':>16} | {'硬核收复':>8} | {'总成功':>6} | 逐硬核种子")
    for d in DOCKS:
        rs = [r for r in results if tuple(r["dock"]) == tuple(d)]
        hard_hit = [r["seed"] for r in rs if r["seed"] in hard and r["ok"]]
        tot = sum(1 for r in rs if r["ok"])
        label = f"({d[0]:.2f},y{d[2]:+.2f})"
        hstr = ",".join(str(s) for s in sorted(hard_hit)) or "-"
        print(f"{label:>16} | {len(hard_hit):>3}/{len(hard):<3}   | {tot:>3}/{len(SEEDS)}  | {hstr}")
    print(f"\n--- 逐硬核种子成功站位 ---")
    for s in sorted(hard):
        got = [d for d in DOCKS if (lambda r: r and r["ok"])(
            next((x for x in results if x["seed"] == s and tuple(x["dock"]) == tuple(d)), None))]
        print(f"  seed {s:>3}: {[f'(x{d[0]:.2f},yaw{d[2]:+.2f})' for d in got] or '全失败'}")


if __name__ == "__main__":
    out = ROOT / "out" / "diag_approach.txt"
    out.parent.mkdir(exist_ok=True)
    t0 = time.time()
    jobs = [(s, d) for s in SEEDS for d in DOCKS]
    W = min(8, len(jobs))
    results = []
    with open(out, "w") as f:
        f.write(f"start jobs={len(jobs)} t={t0:.0f}\n"); f.flush()
        with Pool(W) as p:
            for r in p.imap_unordered(run_one, jobs):
                results.append(r)
                f.write(f"seed {r['seed']} dock={tuple(r['dock'])} ok={r['ok']} "
                        f"gt_succ={r['gt_succ']} gt_lat={r['gt_lat']} gt_ov={r['gt_ov']} {r['msg']}\n")
                f.flush()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summarize(results)
    print(buf.getvalue())
    out.write(buf.getvalue())
    print(f"[{time.time()-t0:.0f}s] 详细见 {out}")
