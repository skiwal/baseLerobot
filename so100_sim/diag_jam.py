#!/usr/bin/env python
"""P2 软核卡阻二级诊断: 对已知插入失败种子, 抓取 push_in_perceived 的逐步轨迹
(重叠 ov / 横向偏心 |e| / 指令推进量 advanced / 停滞计数), 定位卡阻根因:

  * max|e| 超过 LATERAL_MAX(6mm) -> 横向修正饱和 -> 楔死(应放宽 LATERAL_MAX/GAIN 或加再对准)
  * |e| 始终小但 ov 停滞在 <30mm -> 非横向问题(几何/停滞阈值/预插偏心)
  * ov 接近 34mm 目标但被 stalled>40 误杀 -> 停滞阈值过紧

用法: MUJOCO_GL=egl python diag_jam.py [seeds逗号列表]
默认: 已知软核卡阻 10 种子 + 4 成功对照
"""
import sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "perception" / "python"))
from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import park_arm, drive_cart_to
import cart_transport as CT
import grasp_perceived as GP

LATERAL_MAX = 0.006   # 与 so100_sleeve_env.LATERAL_MAX 一致


def run_seed_diag(seed):
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    res = {"seed": seed, "outcome": "exception", "jammed": False, "final_ov_mm": None,
           "final_e_mm": None, "max_e_mm": None, "final_adv_mm": None, "align_deg": None,
           "gt_success": None, "gt_ov_mm": None, "gt_lat_mm": None, "gt_align_deg": None,
           "n_steps": 0, "msg": ""}
    try:
        rng = np.random.default_rng(seed)
        env.reset(tilt_deg=35.0, azim_deg=0.0, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
        drive_cart_to(env, *CT.ASSEMBLY_DOCK)

        GP._PUSH_REC = []                       # 打开逐步记录器
        ok, info = CT.run_once(env, verbose=False)
        traj = GP._PUSH_REC; GP._PUSH_REC = None

        if isinstance(info, dict):
            res["jammed"] = bool(info.get("jammed", False))
            res["outcome"] = "success" if ok else "insert_fail"
            res["final_ov_mm"] = info.get("insert_overlap_mm")
            res["align_deg"] = info.get("align_axis_deg")
            res["final_e_mm"] = info.get("final_lateral_mm")
            res["gt_success"] = info.get("gt_success")
            res["gt_ov_mm"] = info.get("gt_overlap_mm")
            res["gt_lat_mm"] = info.get("gt_lateral_mm")
            res["gt_align_deg"] = info.get("gt_align_deg")
        else:
            res["outcome"] = "grasp_fail"; res["msg"] = str(info)

        if traj:
            arr = np.array(traj)                # (n,5): ov, |e|, advanced, stalled, jammed
            res["n_steps"] = len(arr)
            res["final_e_mm"] = res["final_e_mm"] or float(arr[-1, 1] * 1000)
            res["max_e_mm"] = float(np.max(arr[:, 1]) * 1000)
            res["final_adv_mm"] = float(arr[-1, 2] * 1000)
    except Exception as e:
        res["msg"] = f"EXC {type(e).__name__}: {e}"
    finally:
        env.close()
    return res


def summarize(results, f):
    results = sorted(results, key=lambda r: r["seed"])
    jams = [r for r in results if r["outcome"] == "insert_fail"]
    succ = [r for r in results if r["outcome"] == "success"]
    print(f"\n================ P2 软核卡阻二级诊断 (N={len(results)}) ================", file=f)
    print(f"成功 {len(succ)} / 插入失败 {len(jams)}", file=f)
    print(f"\n--- 插入失败逐种子 (感知ov / 最终e / 峰值e / 推进 / GT成功 / GT重叠 / GT横向) ---", file=f)
    print(f"  {'seed':>5} {'ov':>6} {'f_e':>6} {'max_e':>7} {'adv':>6} {'GT?':>4} {'gtov':>6} {'gtlat':>6}  note", file=f)
    for r in jams:
        gt = r["gt_success"]
        # 区分: GT物理已坐入(感知度量失败) vs 真实物理卡死
        if gt:
            note = "⚠感知度量失败(GT已成功)"
        elif (r["max_e_mm"] or 0) > LATERAL_MAX * 1000:
            note = "横向饱和楔死"
        else:
            note = "非横向/几何?"
        print(f"  {r['seed']:>5} {(r['final_ov_mm'] or 0):6.1f} {(r['final_e_mm'] or 0):6.1f} "
              f"{(r['max_e_mm'] or 0):7.1f} {(r['final_adv_mm'] or 0):6.1f} "
              f"{('Y' if gt else ('N' if gt is False else '?')):>4} "
              f"{(r['gt_ov_mm'] or 0):6.1f} {(r['gt_lat_mm'] or 0):6.1f}  {note}", file=f)
    if jams:
        me = np.array([r["max_e_mm"] or 0 for r in jams])
        ov = np.array([r["final_ov_mm"] or 0 for r in jams])
        gts = [r for r in jams if r["gt_success"]]
        wedge = [r for r in jams if not r["gt_success"] and (r["max_e_mm"] or 0) > LATERAL_MAX * 1000]
        print(f"\n  峰值|e|  : 中位={np.median(me):.1f}mm p90={np.percentile(me,90):.1f}mm "
              f"超{LATERAL_MAX*1000:.0f}mm上限={int(np.sum(me > LATERAL_MAX*1000))}/{len(jams)}", file=f)
        print(f"  感知ov   : 中位={np.median(ov):.1f}mm", file=f)
        print(f"  GT物理已成功但感知判失败(度量失真): {len(gts)}/{len(jams)} -> {[r['seed'] for r in gts]}", file=f)
        print(f"  GT物理也失败且横向饱和(真楔死):     {len(wedge)}/{len(jams)} -> {[r['seed'] for r in wedge]}", file=f)
    print(f"\n--- 成功对照 ---", file=f)
    for r in succ[:8]:
        print(f"  seed {r['seed']}: 感知ov={r['final_ov_mm'] or 0:.1f}mm f_e={r['final_e_mm'] or 0:.1f}mm "
              f"max_e={r['max_e_mm'] or 0:.1f}mm GTov={r['gt_ov_mm'] or 0:.1f}mm GTlat={r['gt_lat_mm'] or 0:.1f}mm", file=f)
    print(f"\n[结论] 见上方分类: 感知度量失真 vs 真实横向楔死 -> 分别对症修复", file=f)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seeds = [int(x) for x in sys.argv[1].split(",")]
    else:
        # 已知软核卡阻 10 种子(P1 前诊断) + 4 成功对照
        seeds = [1, 5, 18, 26, 55, 58, 59, 84, 97, 98, 0, 3, 7, 10]
    W = min(8, len(seeds))
    t0 = time.time()
    with open(ROOT / "out" / "diag_jam.txt", "w") as f:
        f.write(f"start seeds={seeds} t={t0:.0f}\n"); f.flush()
        with Pool(W) as p:
            for r in p.imap_unordered(run_seed_diag, seeds):
                f.write(f"seed {r['seed']}: {r['outcome']} ov={r['final_ov_mm']} "
                        f"final_e={r['final_e_mm']} max_e={r['max_e_mm']} adv={r['final_adv_mm']} "
                        f"jammed={r['jammed']} gt={r['gt_success']} gtov={r['gt_ov_mm']} "
                        f"gtlat={r['gt_lat_mm']} gtalign={r['gt_align_deg']} {r['msg']}\n"); f.flush()
                f.flush()
        # 重新读取汇总
        rows = []
        for line in open(ROOT / "out" / "diag_jam.txt"):
            if not line.startswith("seed "):
                continue
            parts = line.split()
            sd = int(parts[1].rstrip(":"))
            def g(k):
                for x in parts:
                    if x.startswith(k):
                        v = x[len(k):]
                        return None if v == "None" else float(v)
                return None
            jtok = [x for x in parts if x.startswith("jammed=")]
            jammed = (jtok[0] == "jammed=True") if jtok else False
            gtok = [x for x in parts if x.startswith("gt=")]
            gsucc = (gtok[0] == "gt=True") if gtok else None
            rows.append({"seed": sd, "outcome": parts[2], "final_ov_mm": g("ov="),
                         "final_e_mm": g("final_e="), "max_e_mm": g("max_e="),
                         "final_adv_mm": g("adv="), "jammed": jammed,
                         "gt_success": gsucc, "gt_ov_mm": g("gtov="),
                         "gt_lat_mm": g("gtlat="), "gt_align_deg": g("gtalign="), "msg": ""})
        summarize(rows, f)
    print(f"\n详细: {ROOT/'out'/'diag_jam.txt'}  ({time.time()-t0:.0f}s)")
