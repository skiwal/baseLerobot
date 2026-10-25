#!/usr/bin/env python
"""RL 数据采集: 推入子阶段, 脚本基座 + 探索噪声 + 扰动注入 -> LeRobotDataset(IDQL 离线阶段)。

约束(与全局一致): 观测/动作只用真机器可得到的信号; 不记录任何仿真平台内部数据
(接触力/GT位姿/GT成功标志都不进数据集)。

设计要点:
  * 动作多样性(v2 反应式残差, 修正 v1 纯噪声的缺陷): 纯零均值噪声下"动作质量"与
    episode 结果无关(成功靠当集噪声小), AWR 权重近均匀 -> 策略只学到噪声分布。
    v2 改为**反应式残差**: 横向 -k*e + 轴纠 -k_a*tilt + 轴向偏置, 每回合从参数网格
    随机取 (k, k_a, axial_bias, sigma), 让数据包含不同质量的定心/轴纠/进给行为,
    AWR 才能按优势提取最优组合;
  * 扰动注入: env 内置(p_perturb 横向/轴倾斜、p_bad_start 坏起点), 制造需要恢复的坏状态;
  * 失败轨迹全部保留: IDQL 从坏数据里学"什么是坏的";
  * 数据集布局: 帧 = (observation.state[31], action[5], reward[1], done[1]),
    s' 取同集下一帧的 obs(IDQL 标准布局; 重叠/偏心/轴角都可由 obs 推导, 不另存)。

用法:
    MUJOCO_GL=egl python so100_sim/collect_rl_data.py --episodes 200 --workers 4
    产物: <out-root>/<repo-id>_w<rank>/  (LeRobotDataset v3, 纯本地)
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rl_insert_env import (InsertPushEnv, OBS_DIM, ACT_DIM,
                           DELTA_LIMITS, LATERAL_MAX)

DEFAULT_ROOT = Path("/home/ubuntu/code/baseLerobot/datasets/so100_insert_deep")
DEFAULT_REPO = "local/so100_insert_deep"
TASK_TEXT = "insert the sleeve onto the rod"

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (OBS_DIM,),
        "names": None,
    },
    "action": {
        "dtype": "float32",
        "shape": (ACT_DIM,),
        "names": ["d_axial", "d_lat1", "d_lat2", "d_tilt1", "d_tilt2"],
    },
    "reward": {"dtype": "float32", "shape": (1,), "names": None},
    "done": {"dtype": "bool", "shape": (1,), "names": None},
}


def reactive_action(obs: np.ndarray, k_lat: float, k_axis: float,
                    axial_bias: float, sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """反应式残差动作(杆体系, 归一化到 [-1,1])。

    分量(与 rl_insert_env 的动作语义一致):
      [0] 轴向:  常量偏置(额外进给, 归一化单位 = DELTA_LIMITS[0] m);
      [1,2] 横向: -k_lat * e_rf / LATERAL_MAX  (定心, e_rf=obs[17:20]);
      [3,4] 轴倾: -k_axis * tilt / DELTA_LIMITS[3](轴纠, tilt=obs[20:22] rad)。
    再加 N(0, sigma) 探索噪声。
    """
    e_rf = obs[17:20]                       # 横向偏心(杆系, m; 轴向分量≈0)
    tilt = obs[20:22]                       # 轴倾误差(rad)
    a = np.zeros(ACT_DIM, dtype=np.float32)
    a[0] = axial_bias
    a[1] = np.clip(-k_lat * e_rf[0] / LATERAL_MAX, -1.0, 1.0)
    a[2] = np.clip(-k_lat * e_rf[1] / LATERAL_MAX, -1.0, 1.0)
    a[3] = np.clip(-k_axis * tilt[0] / DELTA_LIMITS[3], -1.0, 1.0)
    a[4] = np.clip(-k_axis * tilt[1] / DELTA_LIMITS[3], -1.0, 1.0)
    if sigma > 0:
        a = a + rng.normal(0.0, sigma, ACT_DIM).astype(np.float32)
    return np.clip(a, -1.0, 1.0)


def collect_worker(rank: int, episodes: int, seed: int, root: Path,
                   repo_id: str, p_perturb: float, p_bad_start: float,
                   overwrite: bool):
    """单个采集进程: 自建环境 + 自建数据集分片(env 必须在子进程内创建, 勿在 fork 前建)。"""
    shard_repo = f"{repo_id}_w{rank}"
    shard_root = root / f"w{rank}"
    if shard_root.exists():
        if not overwrite:
            print(f"[w{rank}] 分片目录已存在, 跳过: {shard_root}")
            return
        shutil.rmtree(shard_root)

    env = InsertPushEnv(render=True, p_perturb=p_perturb, p_bad_start=p_bad_start)
    rng = np.random.default_rng(seed + rank * 1_000_000)

    ds = LeRobotDataset.create(
        repo_id=shard_repo,
        fps=30,
        features=FEATURES,
        root=shard_root,
        robot_type="so100_sim",
        use_videos=False,
        image_writer_threads=0,
    )

    t0 = time.time()
    n_saved, n_reset_fail, successes, lengths, returns = 0, 0, [], [], []
    for ep in range(episodes):
        ep_seed = seed + rank * 1_000_000 + ep
        try:
            obs, reset_info = env.reset(seed=ep_seed)
        except RuntimeError:
            n_reset_fail += 1          # 抓取/对准重试耗尽(随机位姿的几何/感知极限)
            continue
        # 本回合行为参数(网格随机): 总横向增益/轴纠增益/轴向偏置/噪声强度
        # k_lat 是"残差承担的增益", 基座另有 0.3, 故总增益 ≈ 0.3 + k_lat
        # v3 攻卡阻: 轴向偏置加档 —— 含回退(-0.2)和更激进进给(0.2/0.4),
        # 让 AWR 能在"快推/慢推/回退"里按优势挑(v1/v2 只有 0/0.3 两档, 轴向学得保守)
        k_lat = float(rng.choice([0.3, 0.5, 0.7, 0.9]))
        k_axis = float(rng.choice([0.0, 0.3, 0.6]))
        axial_bias = float(rng.choice([-0.2, 0.0, 0.2, 0.4]))
        sigma = float(rng.uniform(0.10, 0.25))
        terminated = truncated = False
        total_r, n_steps = 0.0, 0
        while not (terminated or truncated):
            action = reactive_action(obs, k_lat, k_axis, axial_bias, sigma, rng)
            obs_next, r, terminated, truncated, step_info = env.step(action)
            ds.add_frame({
                "observation.state": np.ascontiguousarray(obs, dtype=np.float32),
                "action": np.ascontiguousarray(action, dtype=np.float32),
                "reward": np.array([r], dtype=np.float32),
                "done": np.array([terminated or truncated], dtype=bool),
                "task": TASK_TEXT,
            })
            obs = obs_next
            total_r += r
            n_steps += 1
        ds.save_episode()
        n_saved += 1
        successes.append(bool(step_info["success"]))
        lengths.append(n_steps)
        returns.append(total_r)
        if n_saved % 10 == 0 or n_saved == 1:
            print(f"[w{rank}] 已存 {n_saved}/{episodes} (重置失败 {n_reset_fail}) "
                  f"成功率 {np.mean(successes)*100:.0f}% 均长 {np.mean(lengths):.0f} 步 "
                  f"用时 {time.time()-t0:.0f}s", flush=True)

    ds.finalize()
    env.close()
    print(f"[w{rank}] 完成: {n_saved} 集 / {episodes} 尝试, 重置失败 {n_reset_fail}, "
          f"推入成功率 {np.mean(successes)*100:.1f}%, 均长 {np.mean(lengths):.0f} 步, "
          f"均回报 {np.mean(returns):.1f}, 目录 {shard_root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200, help="总回合数(按 worker 平分)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--repo-id", type=str, default=DEFAULT_REPO)
    ap.add_argument("--p-perturb", type=float, default=0.01)
    ap.add_argument("--p-bad-start", type=float, default=0.25)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    per_worker = max(1, args.episodes // max(1, args.workers))
    print(f"采集 {args.episodes} 回合 ({args.workers} 进程 x {per_worker}) -> {root}")

    if args.workers <= 1:
        collect_worker(0, per_worker, args.seed, root, args.repo_id,
                       args.p_perturb, args.p_bad_start, args.overwrite)
        return

    procs = []
    for rank in range(args.workers):
        p = mp.Process(target=collect_worker,
                       args=(rank, per_worker, args.seed, root, args.repo_id,
                             args.p_perturb, args.p_bad_start, args.overwrite))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
