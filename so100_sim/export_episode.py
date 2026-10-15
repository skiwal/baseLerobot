#!/usr/bin/env python
"""把 LeRobotDataset 里的 episode 导出为可播放的 H.264 MP4。

背景: 数据集内部视频用 AV1 编码 (压缩率高, 训练管线用 torchcodec 读取),
      但 AV1 在部分播放器(QuickTime/旧版 Windows 播放器)上打不开。
      本工具把指定 episode 导出为全平台兼容的 H.264。

用法:
    # 导出第 0 集
    python export_episode.py --episode 0

    # 导出前 3 集
    python export_episode.py --episode 0 1 2

    # 导出全部
    python export_episode.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from video_utils import write_video_h264

DATASET_ROOT = "/home/ubuntu/code/baseLerobot/datasets/so100_pick"
REPO_ID = "local/so100_pick"
VIDEO_KEY = "observation.images.side"
OUT_DIR = Path(__file__).resolve().parent / "out" / "dataset_episodes"


def export_episode(ds: LeRobotDataset, ep: int, out_dir: Path) -> Path:
    meta = ds.meta.episodes[ep]
    start, end = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    frames = []
    for idx in range(start, end):
        item = ds[idx]
        img = item[VIDEO_KEY].numpy()  # (3, H, W) float [0,1]
        frames.append((img.transpose(1, 2, 0) * 255).astype(np.uint8))
    out = out_dir / f"episode_{ep:03d}.mp4"
    write_video_h264(out, frames, fps=ds.fps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DATASET_ROOT)
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--episode", type=int, nargs="*", default=[0])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = range(ds.num_episodes) if args.all else args.episode
    for ep in episodes:
        out = export_episode(ds, ep, out_dir)
        print(f"已导出 episode {ep} -> {out}")

    print(f"\n共 {len(list(episodes))} 集, 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
