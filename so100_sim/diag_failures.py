#!/usr/bin/env python
"""失败归因诊断(0 号任务): 在 81% 脚本流水线上跑 N 个种子, 对每个回合记录:

  1) 感知杆轴误差(度)   = arccos(|感知杆轴 @ GT杆轴|)   -> 指向"感知噪声导致对准失败"
  2) 感知套筒位置误差(mm) = ||感知套筒中心 - GT套筒中心|| -> 指向"抓取/位姿估计失败"
  3) 全程 IK 解最小关节余量(rad) = min(限位上界-q, q-限位下界) -> 指向"几何极限/肘部撞限位"

然后对失败回合做归因拆分:
  * perception-dominated: 杆轴误差显著高于全体 p90, 且关节余量正常
  * geometry-dominated  : 关节余量显著低于全体 p10, 且杆轴误差正常
  * both / neither(软核) : 两者都高 / 两者都正常 -> 卡阻等软失败

用法: MUJOCO_GL=egl python diag_failures.py [N=100] [W=8]
结果同时写 out/diag_failures.txt。
"""
import sys, time, math
from pathlib import Path
import numpy as np
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import SO100SleeveEnv, ARM_JOINTS
from perception_api import perceive_scene
from cart_assembly import park_arm, drive_cart_to
import cart_transport as CT

OUT = ROOT / "out" / "diag_failures.txt"


def run_seed_diag(seed):
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    res = {"seed": seed, "outcome": "exception", "rod_err_deg": None,
           "rod_err_tb_deg": None, "sleeve_err_mm": None, "min_margin": None,
           "jammed": False, "msg": ""}
    try:
        rng = np.random.default_rng(seed)
        env.reset(tilt_deg=35.0, azim_deg=0.0, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
        drive_cart_to(env, *CT.ASSEMBLY_DOCK)   # 移到勘测位, 相机才看得到套筒/杆(与 run_once 一致)

        # ---- GT (必须在 run_once 覆盖 env.sleeve_pose 之前取) ----
        gt_rod_axis = env.rod_axis().copy()
        gt_sleeve_pos = env.sleeve_pose()[0].copy()

        # ---- 感知(复测一次用于量误差; run_once 内部会再测, 不影响控制) ----
        det = perceive_scene(env, "base_rgbd")
        if "rod" not in det or "sleeve" not in det:
            res["msg"] = "感知缺失"; return res
        perc_rod = np.asarray(det["rod"]["axis"], float); perc_rod /= np.linalg.norm(perc_rod)
        perc_sleeve = np.asarray(det["sleeve"]["pos"], float)
        res["rod_err_deg"] = float(np.degrees(np.arccos(np.clip(abs(perc_rod @ gt_rod_axis), 0, 1))))
        res["sleeve_err_mm"] = float(np.linalg.norm(perc_sleeve - gt_sleeve_pos) * 1000.0)
        # 杆轴"由 tip-base 反算"的误差(验证 tip-base 是否比 C 拟合轴更稳, 用于定修复方案)
        if "tip" in det["rod"] and "base" in det["rod"]:
            tb = np.asarray(det["rod"]["tip"], float) - np.asarray(det["rod"]["base"], float)
            nn = np.linalg.norm(tb)
            if nn > 1e-6:
                tb /= nn
                res["rod_err_tb_deg"] = float(np.degrees(np.arccos(np.clip(abs(tb @ gt_rod_axis), 0, 1))))

        # ---- 挂 IK 记录器: 量每次逆解离关节限位多近 ----
        ranges = np.array([env.jnt_range[n] for n in ARM_JOINTS])   # (5,2)
        margins = []
        orig_ik = env.pose_ik
        def rec_ik(*a, **k):
            sol = orig_ik(*a, **k)
            q = np.asarray(sol, float)
            m = min(float(np.min(ranges[:, 1] - q)), float(np.min(q - ranges[:, 0])))
            margins.append(m)
            return sol
        env.pose_ik = rec_ik

        ok, info = CT.run_once(env, verbose=False)
        env.pose_ik = orig_ik
        res["min_margin"] = float(min(margins)) if margins else float("inf")

        if isinstance(info, dict):                 # 走完 insertion
            res["jammed"] = bool(info.get("jammed", False))
            res["outcome"] = "success" if ok else "insert_fail"
        else:                                      # grasp 阶段就失败
            res["outcome"] = "grasp_fail"
            res["msg"] = str(info)
    except Exception as e:
        res["msg"] = f"EXC {type(e).__name__}: {e}"
    finally:
        env.close()
    return res


def summarize(results, f):
    results = sorted(results, key=lambda r: r["seed"])
    N = len(results)
    succ = [r for r in results if r["outcome"] == "success"]
    fails = [r for r in results if r["outcome"] != "success"]
    grasp_fails = [r for r in fails if r["outcome"] == "grasp_fail"]
    insert_fails = [r for r in fails if r["outcome"] == "insert_fail"]

    rod = np.array([r["rod_err_deg"] for r in results if r["rod_err_deg"] is not None])
    marg = np.array([r["min_margin"] for r in results if r["min_margin"] is not None])
    rod_p90 = float(np.percentile(rod, 90)) if len(rod) else 0.0
    marg_p10 = float(np.percentile(marg, 10)) if len(marg) else 0.0
    # 阈值: 杆轴误差 > p90 视为"偏高"; 关节余量 < p10 视为"撞限位"
    ROD_HI = max(rod_p90, 4.0)        # 至少 4° 才算感知异常(基线有 ~2.3° 确定性深度误差)
    MARG_LO = min(marg_p10, 0.06)      # 至少 0.06rad(~3.4°) 才算接近限位

    def attrib(r):
        re = r["rod_err_deg"] or 0.0
        mm = r["min_margin"]
        if mm is None:
            return "unknown"
        hi_rod = re > ROD_HI
        lo_marg = mm < MARG_LO
        if hi_rod and not lo_marg: return "perception"
        if lo_marg and not hi_rod: return "geometry"
        if hi_rod and lo_marg:     return "both"
        return "neither(soft)"

    cats = {}
    for r in insert_fails:
        c = attrib(r); cats[c] = cats.get(c, 0) + 1

    print(f"\n================ 失败归因诊断 (N={N}) ================", file=f)
    print(f"成功率: {len(succ)}/{N} = {100*len(succ)/N:.0f}%", file=f)
    print(f"失败: {len(fails)}  (抓取 {len(grasp_fails)} / 插入 {len(insert_fails)})", file=f)
    print(f"全体分布: 杆轴误差 p50={np.median(rod):.2f}° p90={rod_p90:.2f}° | "
          f"最小关节余量 p50={np.median(marg):.3f} p10={marg_p10:.3f}", file=f)
    print(f"归因阈值: 杆轴误差>{ROD_HI:.1f}° 视为感知异常; 关节余量<{MARG_LO:.3f}rad 视为撞限位", file=f)
    print(f"\n--- 插入失败({len(insert_fails)}) 归因拆分 ---", file=f)
    for c in ("perception", "geometry", "both", "neither(soft)", "unknown"):
        if cats.get(c):
            print(f"  {c:16s}: {cats[c]}", file=f)
    print(f"\n--- 逐失败明细 ---", file=f)
    print(f"  {'seed':>5} {'type':12s} {'rod_err':>7} {'sleeve_err':>11} {'min_margin':>10} {'jammed':>6}  msg", file=f)
    for r in sorted(fails, key=lambda x: x["seed"]):
        print(f"  {r['seed']:>5} {r['outcome']:12s} "
              f"{(r['rod_err_deg'] or 0):7.2f} "
              f"{(r['sleeve_err_mm'] or 0):11.1f} "
              f"{(r['min_margin'] if r['min_margin'] is not None else float('nan')):10.3f} "
              f"{('Y' if r['jammed'] else '-'):>6}  {r['msg'][:40]}", file=f)
    # 终端也打印核心结论
    print(f"\n[结论] 成功率 {100*len(succ)/N:.0f}% | 插入失败 {len(insert_fails)}: "
          f"感知主导 {cats.get('perception',0)} / 几何主导 {cats.get('geometry',0)} / "
          f"两者 {cats.get('both',0)} / 软核 {cats.get('neither(soft)',0)}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = ([int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(N)))
    N = len(seeds)
    t0 = time.time()
    with open(OUT, "w") as f:
        f.write(f"start N={N} W={W} t={t0:.0f}\n"); f.flush()
        results = []
        with Pool(W) as p:
            for r in p.imap_unordered(run_seed_diag, seeds):
                results.append(r)
                f.write(f"seed {r['seed']}: {r['outcome']} "
                        f"rod_err={r['rod_err_deg']} rod_err_tb={r['rod_err_tb_deg']} "
                        f"sleeve_err={r['sleeve_err_mm']} "
                        f"min_margin={r['min_margin']} jammed={r['jammed']} {r['msg']}\n")
                f.flush()
        summarize(results, f)
    print(f"\n详细结果: {OUT}  ({time.time()-t0:.0f}s)")
