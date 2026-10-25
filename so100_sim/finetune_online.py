#!/usr/bin/env python
"""M2 在线微调 v2: IDQL offline -> online(恢复导向重设计)。

v1 失败教训(400 回合, 受扰成功率 17%->13% 反而略降): 采集用单样本策略(受扰成功率仅 ~20%),
在线 buffer 被失败轨迹主导, AWR 只学会"避开坏动作", 恢复正例太稀疏(~40 条)。

v2 三处核心改动:
  1. 采集用 **Q 筛选策略**(best_of_k=8, 受扰成功率 ~50-70%): buffer 正例密度 x3,
     恢复轨迹密集, AWR 有东西可学;
  2. **在线权重 50%->80%**: 在线信号不被 41 万离线数据淹没(留 20% 离线锚定防遗忘);
  3. **受扰起点课程**: 受扰采集环境坏起点率 0.25->0.5, 一半受扰回合从"已偏"状态起,
     专产恢复数据; 干净/受扰回合交替保基础技能不退化。

用法:
    MUJOCO_GL=egl python so100_sim/finetune_online.py \
        --init .../ckpt_75000.pt --episodes 300 --online-ratio 0.8
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from idql_model import IDQLModel
from rl_insert_env import InsertPushEnv, ACT_DIM, OBS_DIM
from train_idql import load_transitions, evaluate_policy, evaluate_base
from collect_rl_data import reactive_action

DEFAULT_OUT = Path("/home/ubuntu/code/baseLerobot/outputs/idql_online2")
ONLINE_CAP = 150_000          # 在线缓冲上限


class ReplayBuffer:
    """GPU 回放缓冲: 预分配离线 + 在线区, 在线 FIFO。"""

    def __init__(self, obs, act, rew, next_obs, done, cap_online: int, device: str):
        self.device = device
        self.n_off = len(obs)
        self.cap_on = cap_online
        self.obs = torch.cat([torch.from_numpy(obs).to(device),
                              torch.zeros(cap_online, obs.shape[1], device=device)])
        self.act = torch.cat([torch.from_numpy(act).to(device),
                              torch.zeros(cap_online, act.shape[1], device=device)])
        self.rew = torch.cat([torch.from_numpy(rew).to(device),
                              torch.zeros(cap_online, device=device)])
        self.next_obs = torch.cat([torch.from_numpy(next_obs).to(device),
                                   torch.zeros(cap_online, next_obs.shape[1], device=device)])
        self.done = torch.cat([torch.from_numpy(done).to(device),
                               torch.zeros(cap_online, device=device)])
        self.n_on = 0
        self.on_ptr = 0

    def add(self, s, a, r, s2, d):
        i = self.n_off + self.on_ptr
        self.obs[i] = torch.from_numpy(np.asarray(s, dtype=np.float32)).to(self.device)
        self.act[i] = torch.from_numpy(np.asarray(a, dtype=np.float32)).to(self.device)
        self.rew[i] = float(r)
        self.next_obs[i] = torch.from_numpy(np.asarray(s2, dtype=np.float32)).to(self.device)
        self.done[i] = float(d)
        self.on_ptr = (self.on_ptr + 1) % self.cap_on
        self.n_on = min(self.n_on + 1, self.cap_on)

    def sample(self, batch: int, rng: np.random.Generator, online_ratio: float = 0.8):
        """按 online_ratio 混合采样(在线不足时按比例回退离线)。"""
        n_on_want = int(round(batch * online_ratio))
        n_on = min(n_on_want, self.n_on)
        n_off = batch - n_on
        idx_on = (self.n_off + rng.integers(0, self.n_on, size=n_on)) if n_on > 0 else np.array([], dtype=int)
        idx_off = rng.integers(0, self.n_off, size=n_off)
        idx = np.concatenate([idx_on, idx_off]) if n_on > 0 else idx_off
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.next_obs[idx], self.done[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, required=True, help="离线 checkpoint(如 ckpt_75000.pt)")
    ap.add_argument("--data-root", type=str,
                    default="/home/ubuntu/code/baseLerobot/datasets/so100_insert_react")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--episodes", type=int, default=300, help="在线回合数")
    ap.add_argument("--grad-steps", type=int, default=30, help="每在线回合的梯度步数")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--online-ratio", type=float, default=0.8, help="批内在线数据比例")
    ap.add_argument("--eps-explore", type=float, default=0.10, help="反应式探索概率")
    ap.add_argument("--best-of-k-collect", type=int, default=8, help="采集期 Q 筛选候选数")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-episodes", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-scale", type=float, default=0.01)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 7)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 数据 + 模型 ---
    obs, act, rew, next_obs, done = load_transitions(Path(args.data_root), [0, 1, 2, 3])
    rew = rew * args.reward_scale
    obs_mean = torch.from_numpy(obs.mean(axis=0)).to(device)
    obs_std = torch.from_numpy(obs.std(axis=0) + 1e-6).to(device)
    buf = ReplayBuffer(obs, act, rew, next_obs, done, ONLINE_CAP, device)

    ckpt = torch.load(args.init, map_location=device)
    model = IDQLModel(obs_dim=OBS_DIM, act_dim=ACT_DIM,
                      discount=ckpt["config"]["discount"],
                      expectile_tau=ckpt["config"]["expectile_tau"],
                      awr_beta=ckpt["config"]["awr_beta"]).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[初始化] {args.init} (offline step {ckpt.get('step')})", flush=True)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- 环境: 采集(干净 + 受扰-坏起点课程) + 评测(受扰) ---
    env_collect_clean = InsertPushEnv(render=True, p_perturb=0.0, p_bad_start=0.0)
    env_collect_pert = InsertPushEnv(render=True, p_perturb=0.015, p_bad_start=0.5)
    env_eval = InsertPushEnv(render=True, p_perturb=0.015, p_bad_start=0.25)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def norm(o):
        return ((torch.from_numpy(o).float().to(device) - obs_mean) / obs_std)[None]

    best_score = -1.0
    t0 = time.time()
    n_success = n_total = 0
    for ep in range(args.episodes):
        perturbed = (ep % 2 == 1)                    # 干净/受扰交替
        env = env_collect_pert if perturbed else env_collect_clean
        ep_seed = 800000 + ep
        try:
            obs_e, _ = env.reset(seed=ep_seed)
        except RuntimeError:
            continue
        # 探索: eps 概率换反应式探索保多样性, 否则用 Q 筛选策略采集(正例密集)
        explore = rng.random() < args.eps_explore
        k_lat = float(rng.choice([0.5, 0.7, 0.9]))
        k_axis = float(rng.choice([0.0, 0.3, 0.6]))
        axial_bias = float(rng.choice([0.0, 0.3]))
        sigma = float(rng.uniform(0.10, 0.25))
        terminated = truncated = False
        info = {}
        o = norm(obs_e)
        while not (terminated or truncated):
            with torch.no_grad():
                if explore:
                    a = reactive_action(obs_e, k_lat, k_axis, axial_bias, sigma, rng)
                else:
                    a = model.sample_action_best_of_k(o, k=args.best_of_k_collect)[0].cpu().numpy()
            obs_next, r, terminated, truncated, info = env.step(a)
            buf.add(obs_e, a, r * args.reward_scale, obs_next,
                    float(terminated or truncated))
            obs_e = obs_next
            o = norm(obs_e)
        n_total += 1
        n_success += int(bool(info["success"]))

        # --- 梯度步(80/20 在线/离线混合) ---
        for _ in range(args.grad_steps):
            b_obs, b_act, b_rew, b_next, b_done = buf.sample(
                args.batch, rng, online_ratio=args.online_ratio)
            b_obs = (b_obs - obs_mean) / obs_std
            b_next = (b_next - obs_mean) / obs_std
            losses = model.compute_losses(b_obs, b_act, b_rew, b_next, b_done)
            total = losses["q_loss"] + losses["v_loss"] + losses["actor_loss"]
            optim.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()
            model.update_targets()

        if (ep + 1) % 10 == 0:
            print(f"[在线 ep {ep+1}/{args.episodes}] {'受扰' if perturbed else '干净'} "
                  f"{'探索' if explore else 'Q筛选'} -> {info['success'] and '成功' or info['crooked'] and '歪插' or '卡阻/超时'} "
                  f"(重叠 {info['overlap_mm']:.0f}mm) | 在线成功率 {n_success}/{n_total} "
                  f"| 缓冲 {buf.n_on} | {time.time()-t0:.0f}s", flush=True)

        # --- 周期性评测(受扰条件, 与基座同种子) ---
        if (ep + 1) % args.eval_every == 0 or ep + 1 == args.episodes:
            st_rl = evaluate_policy(env_eval, model, obs_mean, obs_std,
                                    args.eval_episodes, seed0=90000,
                                    tag=f"·在线{ep+1}", device=device, best_of_k=8)
            evaluate_base(env_eval, args.eval_episodes, seed0=90000)
            score = st_rl["success"] / max(1, st_rl["n"])
            if score >= best_score:
                best_score = score
                torch.save({"model": model.state_dict(), "step": ckpt.get("step", 0),
                            "online_ep": ep + 1,
                            "obs_mean": obs_mean.cpu(), "obs_std": obs_std.cpu(),
                            "config": ckpt["config"]},
                           out_dir / "best.pt")
                print(f"[保存] best.pt (评测成功率 {score:.0%})", flush=True)
            torch.save({"model": model.state_dict(), "step": ckpt.get("step", 0),
                        "online_ep": ep + 1,
                        "obs_mean": obs_mean.cpu(), "obs_std": obs_std.cpu(),
                        "config": ckpt["config"]},
                       out_dir / "last.pt")

    for e in (env_collect_clean, env_collect_pert, env_eval):
        e.close()
    print(f"在线微调 v2 完成: 采集成功率 {n_success}/{n_total}, best 评测 {best_score:.0%}, "
          f"产物 {out_dir}/best.pt (总用时 {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
