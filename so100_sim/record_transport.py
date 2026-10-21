#!/usr/bin/env python
"""录制"套筒随机放置 -> 抓取 -> 搬运 -> 插装"全过程(closeup 全景) -> out/transport.mp4。
用法: python record_transport.py [seed]
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import park_arm
import cart_transport as CT


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    env = SO100SleeveEnv(width=640, height=480, camera="closeup", render=True)
    frames = []
    _step, _setarm = env.step, env.set_arm

    def _rec():
        frames.append(env.render_image("closeup")[:, :, ::-1].copy())

    def step_rec(a):
        _step(a); _rec()

    def setarm_rec(q, j):
        _setarm(q, j); _rec()

    env.step, env.set_arm = step_rec, setarm_rec

    env.reset(tilt_deg=35.0, azim_deg=0.0, rng=np.random.default_rng(seed),
              sleeve_xy_range=CT.SLEEVE_RANGE)
    env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env); _rec()
    print(f"[录制] seed={seed} 起点(0.45,0,0), 套筒由环境随机放置")
    ok, info = CT.run_once(env)
    print(f"[录制] 结果: {'成功' if ok else '失败'} {info if ok else ''}")

    if len(frames) > 900:
        idx = np.linspace(0, len(frames) - 1, 900).astype(int)
        frames = [frames[i] for i in idx]
    out = ROOT / "out" / "transport.mp4"
    out.parent.mkdir(exist_ok=True)
    h, w = frames[0].shape[:2]
    # cv2 只能写 mp4v(MPEG-4 Part2, 很多播放器不支持) -> 转 H.264/yuv420p(全兼容)
    tmp = out.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    import subprocess, imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", "-movflags", "+faststart", str(out)], check=True)
    tmp.unlink()
    print(f"[录制] 已保存 {out} (H.264, {len(frames)} 帧)")


if __name__ == "__main__":
    main()
