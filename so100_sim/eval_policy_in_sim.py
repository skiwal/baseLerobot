#!/usr/bin/env python
"""在 SO-100 MuJoCo 仿真里评估训练好的 ACT 策略。

用法:
    MUJOCO_GL=egl python eval_policy_in_sim.py \
        --checkpoint /home/ubuntu/code/baseLerobot/outputs/train/so100_pick_act/checkpoints/last/pretrained_model \
        --episodes 20 --record 5

产物:
    out/eval_<name>/episode_*.mp4   录像
    out/eval_<name>/summary_grid.png 成功/失败关键帧拼图
    终端打印成功率等指标
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy

from so100_pick_env import FPS, SO100PickEnv, SUCCESS_Z, PEDESTAL_TOP
from video_utils import write_video_h264

OUT_ROOT = Path(__file__).resolve().parent / "out"


def to_policy_obs(obs: dict) -> dict:
    """env 观测 -> 策略输入 (未 batch, 预处理器会加 batch 维)。"""
    state = torch.from_numpy(np.ascontiguousarray(obs["joint_positions"]))
    img = torch.from_numpy(np.ascontiguousarray(obs["image"].transpose(2, 0, 1))).float() / 255.0
    return {"observation.state": state, "observation.images.side": img}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="本地 checkpoint 目录 (含 config.json/model.safetensors)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=150, help="每集最多步数 (演示需要 87 步)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--record", type=int, default=5, help="录几集视频")
    ap.add_argument("--video-size", type=int, default=448)
    ap.add_argument("--tag", default=None, help="输出子目录名")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    assert ckpt.exists(), f"checkpoint 不存在: {ckpt}"
    tag = args.tag or ckpt.parent.name  # e.g. 001000 或 last
    out_dir = OUT_ROOT / f"eval_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"加载策略: {ckpt}")
    policy = ACTPolicy.from_pretrained(str(ckpt))
    policy.eval()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=str(ckpt))
    print(f"策略已就绪 (device={device}), chunk={policy.config.chunk_size}, n_action_steps={policy.config.n_action_steps}")

    env = SO100PickEnv(width=224, height=224, camera="side", render=True)
    video_renderer = mujoco.Renderer(env.model, args.video_size, args.video_size)
    rng = np.random.default_rng(args.seed)

    results = []
    lifts = []
    steps_to_success = []
    t0 = time.time()

    for ep in range(args.episodes):
        env.reset(rng=rng)
        policy.reset()
        cube_start = env.cube_pos().copy()

        frames = []
        success = False
        max_cube_z = 0.0
        for step in range(args.max_steps):
            if ep < args.record:
                video_renderer.update_scene(env.data, camera="side")
                frames.append(video_renderer.render())

            obs = to_policy_obs(env.get_obs(render=True))
            with torch.inference_mode():
                action = policy.select_action(preprocessor(obs))
            action = postprocessor(action).squeeze(0).cpu().numpy()

            env.step(action)
            max_cube_z = max(max_cube_z, env.cube_pos()[2])

            if env.is_success():
                success = True
                break
            if not env.cube_on_table():
                break  # 方块被撞飞, 提前结束

        results.append(success)
        lifts.append(max_cube_z)
        if success:
            steps_to_success.append(step + 1)

        # 录像
        if ep < args.record:
            for _ in range(10):  # 末尾多停 10 帧
                video_renderer.update_scene(env.data, camera="side")
                frames.append(video_renderer.render())
            vpath = out_dir / f"episode_{ep:02d}_{'OK' if success else 'FAIL'}.mp4"
            write_video_h264(vpath, frames, fps=FPS)

        print(f"  ep {ep:02d}: {'✅' if success else '❌'} "
              f"最高抬升 z={max_cube_z:.3f} (阈值{SUCCESS_Z}) 步数={step+1} "
              f"方块起点=({cube_start[0]:+.3f},{cube_start[1]:+.3f})")

    # 汇总
    n = len(results)
    succ = int(np.sum(results))
    print(f"\n===== 评估结果 ({tag}) =====")
    print(f"成功率: {succ}/{n} = {succ / n * 100:.1f}%")
    print(f"平均最高抬升: {np.mean(lifts):.3f} m (方块初始 0.100)")
    if steps_to_success:
        print(f"成功集平均步数: {np.mean(steps_to_success):.0f} / {args.max_steps}")
    print(f"总用时: {time.time() - t0:.0f}s")

    # 结果拼图 (录像集的关键帧)
    if args.record > 0:
        thumbs = []
        for ep in range(min(args.record, 3)):
            vpath = list(out_dir.glob(f"episode_{ep:02d}_*.mp4"))
            if not vpath:
                continue
            cap = cv2.VideoCapture(str(vpath[0]))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            picks = [0, n_frames // 3, n_frames * 2 // 3, n_frames - 1]
            row = []
            for p in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, p)
                ok, img = cap.read()
                if ok:
                    row.append(cv2.resize(img, (224, 224)))
            cap.release()
            if row:
                thumbs.append(np.concatenate(row, axis=1))
        if thumbs:
            grid = np.concatenate(thumbs, axis=0)
            cv2.imwrite(str(out_dir / "summary_grid.png"), grid)
            print(f"拼图: {out_dir / 'summary_grid.png'}")

    env.close()


if __name__ == "__main__":
    main()
