#!/usr/bin/env python
"""IDQL 评测: 同一起点种子集合上对比 RL 策略 / 基座零残差 / 完整脚本(gain=1.0)。

两种条件各测一遍(公平对比, 同种子):
  * 干净(p_perturb=0, 无坏起点): 基础能力;
  * 受扰(p_perturb=0.03, 坏起点 0.3): 恢复能力(M1 go/no-go 判据)。

用法:
    MUJOCO_GL=egl python so100_sim/eval_idql_insert.py --ckpt .../last.pt --episodes 20
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from idql_model import IDQLModel
from rl_insert_env import InsertPushEnv, ACT_DIM
from grasp_perceived import push_in_perceived


def _key(info):
    return ("success" if info["success"] else "crooked" if info["crooked"]
            else "jammed" if info["jammed"] else "timeout")


def _tally(st, info):
    st[_key(info)] += 1
    if info.get("perturbed"):
        st["perturbed"] += 1
        st["recovered"] += int(bool(info["success"]))


def _new_stats():
    return {"success": 0, "crooked": 0, "jammed": 0, "timeout": 0,
            "perturbed": 0, "recovered": 0, "overlap": []}


def run_rl(env: InsertPushEnv, model: IDQLModel, obs_mean, obs_std,
           seed0: int, n: int, best_of_k: int = 1) -> dict:
    st = _new_stats()
    dev = next(model.parameters()).device
    for ep in range(n):
        try:
            obs, _ = env.reset(seed=seed0 + ep)
        except RuntimeError:
            continue
        o = ((torch.from_numpy(obs).float().to(dev) - obs_mean) / obs_std)[None]
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            with torch.no_grad():
                if best_of_k > 1:
                    a = model.sample_action_best_of_k(o, k=best_of_k)[0].cpu().numpy()
                else:
                    a = model.sample_action(o)[0].cpu().numpy()
            obs, r, terminated, truncated, info = env.step(a)
            o = ((torch.from_numpy(obs).float().to(dev) - obs_mean) / obs_std)[None]
        _tally(st, info)
        st["overlap"].append(info["overlap_mm"])
    return st


def run_zero(env: InsertPushEnv, seed0: int, n: int) -> dict:
    st = _new_stats()
    for ep in range(n):
        try:
            env.reset(seed=seed0 + ep)
        except RuntimeError:
            continue
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
        _tally(st, info)
        st["overlap"].append(info["overlap_mm"])
    return st


def run_scripted(env: InsertPushEnv, seed0: int, n: int) -> dict:
    """完整脚本推入(gain=1.0 的 push_in_perceived), 从 env 对准后的状态起。"""
    st = _new_stats()
    for ep in range(n):
        try:
            env.reset(seed=seed0 + ep)
        except RuntimeError:
            continue
        sim = env.sim
        cur = sim.arm_qpos().astype(float)
        sol_prev = cur[:5].copy()
        rinfo = push_in_perceived(sim, env.rod_axis, env.rod_base, cur, sol_prev,
                                  verbose=False)
        st["success" if rinfo["success"] else "jammed"] += 1
        st["overlap"].append(rinfo["insert_overlap_mm"])
    return st


def report(tag: str, st: dict):
    n = max(1, sum(st[k] for k in ("success", "crooked", "jammed", "timeout")))
    rec = (f" 恢复率 {st['recovered']}/{st['perturbed']}"
           f"({st['recovered']/max(1,st['perturbed']):.0%})" if st["perturbed"] else "")
    ov = f" 均重叠 {np.mean(st['overlap']):.1f}mm" if st["overlap"] else ""
    print(f"[{tag}] 成功 {st['success']}/{n} ({st['success']/n:.0%}) "
          f"歪插 {st['crooked']} 卡阻 {st['jammed']} 超时 {st['timeout']}{rec}{ov}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=500000, help="评测种子基(与训练期评测种子不同)")
    ap.add_argument("--skip-scripted", action="store_true")
    ap.add_argument("--best-of-k", type=int, default=8,
                    help="推理期 Q 筛选的候选数(1=单次采样; 8=扩散多模态提议+Q 选优)")
    ap.add_argument("--p-perturb", type=float, default=0.015,
                    help="受扰条件的扰动率(与采集默认 0.01 匹配; 过大属分布外)")
    ap.add_argument("--p-bad-start", type=float, default=0.25)
    ap.add_argument("--rl-only", action="store_true", help="只评 RL(快速扫 checkpoint)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model = IDQLModel(obs_dim=31, act_dim=ACT_DIM,
                      discount=ckpt["config"]["discount"],
                      expectile_tau=ckpt["config"]["expectile_tau"],
                      awr_beta=ckpt["config"]["awr_beta"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    obs_mean = ckpt["obs_mean"].to(device)
    obs_std = ckpt["obs_std"].to(device)
    print(f"加载 checkpoint: {args.ckpt} (step {ckpt.get('step')})")

    env_clean = InsertPushEnv(render=True, p_perturb=0.0, p_bad_start=0.0)
    env_pert = InsertPushEnv(render=True, p_perturb=args.p_perturb,
                             p_bad_start=args.p_bad_start)

    for tag, env in (("干净", env_clean), ("受扰", env_pert)):
        print(f"\n===== 条件: {tag} (seed0={args.seed0}, best_of_k={args.best_of_k}) =====", flush=True)
        report("RL 策略", run_rl(env, model, obs_mean, obs_std, args.seed0,
                                args.episodes, best_of_k=args.best_of_k))
        if args.rl_only:
            continue
        report("基座零残差", run_zero(env, args.seed0, args.episodes))
        if args.skip_scripted:
            continue
        if tag == "受扰":
            # 脚本直接调 push_in_perceived(不经 env.step), 吃不到中程扰动(只吃到坏起点),
            # 受扰条件下对比不公平 -> 跳过(恢复率对比只看 RL vs 基座)。
            print("[完整脚本] 受扰条件跳过(脚本不经 env.step, 不吃中程扰动, 对比不公平)")
            continue
        report("完整脚本(gain=1.0)", run_scripted(env, args.seed0, args.episodes))

    env_clean.close()
    env_pert.close()


if __name__ == "__main__":
    main()
