#!/usr/bin/env python
"""采集脚本化抓取演示, 写入本地 LeRobotDataset (不上传任何东西)。

用法:
    MUJOCO_GL=egl python collect_demos.py --episodes 50

产物:
    /home/ubuntu/code/baseLerobot/datasets/so100_pick/   (LeRobotDataset v3)
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so100_pick_env import ALL_JOINTS, FPS, SO100PickEnv, run_scripted_demo

DATASET_ROOT = Path("/home/ubuntu/code/baseLerobot/datasets/so100_pick")
REPO_ID = "local/so100_pick"  # 本地命名, 不指向任何 Hub 仓库
IMG_SIZE = 224
TASK_TEXT = "pick up the cube"

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ALL_JOINTS,  # Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw (rad)
    },
    "observation.images.side": {
        "dtype": "video",
        "shape": (IMG_SIZE, IMG_SIZE, 3),
        "names": ["height", "width", "channel"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ALL_JOINTS,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的数据集目录")
    args = ap.parse_args()

    if DATASET_ROOT.exists():
        if not args.overwrite:
            raise SystemExit(
                f"数据集目录已存在: {DATASET_ROOT}\n"
                f"如需重新采集请加 --overwrite (会删除旧数据)"
            )
        print(f"删除旧数据集: {DATASET_ROOT}")
        shutil.rmtree(DATASET_ROOT)

    env = SO100PickEnv(width=IMG_SIZE, height=IMG_SIZE, camera="side")
    rng = np.random.default_rng(args.seed)

    print(f"创建数据集 {REPO_ID} -> {DATASET_ROOT}")
    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=DATASET_ROOT,
        robot_type="so100_sim",
        use_videos=True,
        # 流式编码: 帧直接进视频编码器, 不写临时 PNG.
        # 既快, 也避开本机沙箱对批量删除(清理临时图片目录)的拦截。
        streaming_encoding=True,
        encoder_queue_maxsize=60,
        image_writer_processes=0,
        image_writer_threads=0,
    )

    t0 = time.time()
    successes = []
    lengths = []
    n_saved = 0
    n_failed = 0
    for ep in range(args.episodes):
        env.reset(rng=rng)
        demo = run_scripted_demo(env, collect_frames=True)
        if not demo["success"]:
            # 失败的演示不进数据集 (会污染模仿学习)
            n_failed += 1
            successes.append(False)
            continue
        n = len(demo["actions"])
        for t in range(n):
            ds.add_frame(
                {
                    "observation.state": demo["states"][t],
                    "observation.images.side": demo["frames"][t],
                    "action": demo["actions"][t],
                    "task": TASK_TEXT,
                }
            )
        ds.save_episode()
        successes.append(True)
        lengths.append(n)
        n_saved += 1
        if n_saved % 5 == 0 or n_saved == 1:
            print(f"  [已存 {n_saved}/{args.episodes} | 尝试 {ep+1}] "
                  f"成功率 {np.mean(successes)*100:.0f}% 本集帧数={n} 用时={time.time()-t0:.0f}s")

    ds.finalize()
    env.close()

    print(f"\n完成: 尝试 {args.episodes} 集, 成功 {n_saved} 集, 失败丢弃 {n_failed} 集")
    if lengths:
        print(f"平均每集 {np.mean(lengths):.0f} 帧, 数据集总帧数 {sum(lengths)}")
    print(f"脚本演示成功率: {np.mean(successes)*100:.1f}%")
    print(f"数据集目录: {DATASET_ROOT}")
    print("（纯本地存储, 未上传到 HuggingFace）")


if __name__ == "__main__":
    main()
