#!/usr/bin/env python
"""IDQL 训练(M1 离线阶段): 从 LeRobotDataset 推入子阶段转移对训练 残差扩散策略。

流程: 读 parquet 转移对 -> 观测标准化 -> IDQL(Q/V/扩散actor 三损失) ->
      周期性在仿真里评测策略(与基座零残差对比) -> checkpoint。

用法:
    MUJOCO_GL=egl python so100_sim/train_idql.py --steps 400000 --batch 256
    (采集跑完后训练; 训练中用 GPU, 采集用 CPU 仿真, 可同时跑)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from idql_model import IDQLModel
from rl_insert_env import InsertPushEnv, OBS_DIM, ACT_DIM

DEFAULT_ROOT = Path("/home/ubuntu/code/baseLerobot/datasets/so100_insert_react")
DEFAULT_OUT = Path("/home/ubuntu/code/baseLerobot/outputs/idql_insert")


def load_transitions(root: Path, shards: list[int]):
    """读各分片 parquet, 拼成 (s, a, r, s', done) 转移对(全局 episode id 防撞)。

    注意: 必须在采集完成后调用(并发写入时 parquet 读不了)。
    """
    dfs = []
    for rank in shards:
        files = sorted((root / f"w{rank}" / "data").glob("**/*.parquet"))
        if not files:
            print(f"[数据] 跳过缺失分片: w{rank}")
            continue
        df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        df["gid"] = rank * 1_000_000 + df["episode_index"]
        dfs.append(df)
    if not dfs:
        raise SystemExit(f"未找到任何数据分片于 {root}")
    df = pd.concat(dfs, ignore_index=True).sort_values(["gid", "frame_index"]).reset_index(drop=True)

    obs = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    act = np.stack(df["action"].to_numpy()).astype(np.float32)
    rew = df["reward"].to_numpy().astype(np.float32)
    done = df["done"].to_numpy().astype(np.float32)
    gid = df["gid"].to_numpy()

    # s' = 同集下一帧 obs; 回合边界处用 dummy(被 done 掩蔽)
    next_obs = np.roll(obs, -1, axis=0)
    same_next = np.zeros(len(df), dtype=bool)
    same_next[:-1] = gid[1:] == gid[:-1]
    next_obs[~same_next] = obs[~same_next]

    print(f"[数据] {len(df)} 帧 / {df['gid'].nunique()} 回合 | "
          f"均奖励 {rew.mean():.1f} [{rew.min():.0f}, {rew.max():.0f}] | "
          f"done 率 {done.mean():.1%}")
    return obs, act, rew, next_obs, done


def evaluate_policy(env: InsertPushEnv, model: IDQLModel, obs_mean, obs_std,
                    n_ep: int, seed0: int, tag: str = "", device: str = "cpu",
                    best_of_k: int = 1) -> dict:
    """在仿真里跑 n_ep 回合(策略采样, 无探索噪声), 返回统计。"""
    stats = {"success": 0, "crooked": 0, "jammed": 0, "timeout": 0, "overlap": [],
             "steps": [], "perturbed": 0, "recovered": 0}
    for ep in range(n_ep):
        try:
            obs, _ = env.reset(seed=seed0 + ep)
        except RuntimeError:
            continue
        o = ((torch.from_numpy(obs).float().to(device) - obs_mean) / obs_std)[None]
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            with torch.no_grad():
                if best_of_k > 1:
                    a = model.sample_action_best_of_k(o, k=best_of_k)[0].cpu().numpy()
                else:
                    a = model.sample_action(o)[0].cpu().numpy()
            obs, r, terminated, truncated, info = env.step(a)
            o = ((torch.from_numpy(obs).float().to(device) - obs_mean) / obs_std)[None]
        key = ("success" if info["success"] else "crooked" if info["crooked"]
               else "jammed" if info["jammed"] else "timeout")
        stats[key] += 1
        stats["overlap"].append(info["overlap_mm"])
        stats["steps"].append(info["steps"])
        if info.get("perturbed"):
            stats["perturbed"] += 1
            stats["recovered"] += int(bool(info["success"]))
    n = max(1, sum(stats[k] for k in ("success", "crooked", "jammed", "timeout")))
    out = {k: stats[k] for k in ("success", "crooked", "jammed", "timeout")}
    out["n"] = n
    out["mean_overlap"] = float(np.mean(stats["overlap"])) if stats["overlap"] else 0.0
    rec = (f" 恢复率 {stats['recovered']}/{stats['perturbed']}"
           f"({stats['recovered']/max(1,stats['perturbed']):.0%})" if stats["perturbed"] else "")
    print(f"[评测{tag}] 成功 {out['success']}/{n} ({out['success']/n:.0%}) "
          f"歪插 {out['crooked']} 卡阻 {out['jammed']} 超时 {out['timeout']} "
          f"均重叠 {out['mean_overlap']:.1f}mm{rec}", flush=True)
    return out


def evaluate_base(env: InsertPushEnv, n_ep: int, seed0: int) -> dict:
    """基座零残差基线(同种子, 公平对比)。"""
    stats = {"success": 0, "crooked": 0, "jammed": 0, "timeout": 0,
             "perturbed": 0, "recovered": 0}
    for ep in range(n_ep):
        try:
            env.reset(seed=seed0 + ep)
        except RuntimeError:
            continue
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            _, _, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
        key = ("success" if info["success"] else "crooked" if info["crooked"]
               else "jammed" if info["jammed"] else "timeout")
        stats[key] += 1
        if info.get("perturbed"):
            stats["perturbed"] += 1
            stats["recovered"] += int(bool(info["success"]))
    n = max(1, sum(stats[k] for k in ("success", "crooked", "jammed", "timeout")))
    rec = (f" 恢复率 {stats['recovered']}/{stats['perturbed']}"
           f"({stats['recovered']/max(1,stats['perturbed']):.0%})" if stats["perturbed"] else "")
    print(f"[评测·基座零残差] 成功 {stats['success']}/{n} ({stats['success']/n:.0%}){rec}", flush=True)
    stats["n"] = n
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--shards", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--discount", type=float, default=0.99)
    ap.add_argument("--expectile-tau", type=float, default=0.7)
    ap.add_argument("--awr-beta", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--best-of-k", type=int, default=4,
                    help="训练期评测的 Q 筛选候选数(推理期扩散提议+Q 选优)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-scale", type=float, default=0.01,
                    help="奖励缩放(稳定 Q/V 数值尺度; 优势权重对尺度不敏感)")
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    obs, act, rew, next_obs, done = load_transitions(Path(args.data_root), args.shards)
    rew = rew * args.reward_scale          # 奖励缩放(成功大奖 100 -> 1.0 量级)
    # 观测标准化(均值/方差)
    obs_mean = torch.from_numpy(obs.mean(axis=0)).to(device)
    obs_std = torch.from_numpy(obs.std(axis=0) + 1e-6).to(device)
    # 全部搬到 GPU(660k x 31/5 float32 ≈ 90MB, 可行)
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(act).to(device)
    rew_t = torch.from_numpy(rew).to(device)
    next_t = torch.from_numpy(next_obs).to(device)
    done_t = torch.from_numpy(done).to(device)
    N = len(obs_t)

    model = IDQLModel(obs_dim=OBS_DIM, act_dim=ACT_DIM,
                      discount=args.discount, expectile_tau=args.expectile_tau,
                      awr_beta=args.awr_beta).to(device)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)
        print(f"[恢复] {args.resume} @ step {start_step}")
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = InsertPushEnv(render=True)      # 仅用于周期性评测
    rng = np.random.default_rng(args.seed + 1)

    # 基线(同代评测种子, 公平对比)
    print("=== 训练前基线 ===", flush=True)
    evaluate_base(env, args.eval_episodes, seed0=90000)

    t0 = time.time()
    for step in range(start_step, args.steps):
        idx = torch.randint(0, N, (args.batch,), device=device)
        b_obs = (obs_t[idx] - obs_mean) / obs_std
        b_next = (next_t[idx] - obs_mean) / obs_std
        b_act, b_rew, b_done = act_t[idx], rew_t[idx], done_t[idx]

        losses = model.compute_losses(b_obs, b_act, b_rew, b_next, b_done)
        total = losses["q_loss"] + losses["v_loss"] + losses["actor_loss"]
        optim.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optim.step()
        model.update_targets()

        if (step + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"[step {step+1}/{args.steps}] q={losses['q_loss']:.2f} v={losses['v_loss']:.3f} "
                  f"actor={losses['actor_loss']:.4f}(raw {losses['raw_diffusion_loss']:.4f}) "
                  f"Q~{losses['q_mean']:.1f} V~{losses['v_mean']:.1f} | {el:.0f}s", flush=True)

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            evaluate_policy(env, model, obs_mean, obs_std,
                            args.eval_episodes, seed0=90000, tag=f"@{step+1}",
                            device=device, best_of_k=args.best_of_k)
            evaluate_base(env, args.eval_episodes, seed0=90000)
            ckpt = {"model": model.state_dict(), "step": step + 1,
                    "obs_mean": obs_mean.cpu(), "obs_std": obs_std.cpu(),
                    "config": {"discount": args.discount, "expectile_tau": args.expectile_tau,
                               "awr_beta": args.awr_beta}}
            torch.save(ckpt, out_dir / f"ckpt_{step+1}.pt")
            torch.save(ckpt, out_dir / "last.pt")

    env.close()
    print(f"训练完成, checkpoint 于 {out_dir}/last.pt (总用时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
