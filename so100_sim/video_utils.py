#!/usr/bin/env python
"""视频写入工具：统一用 H.264 (libx264) + yuv420p 编码 MP4。

为什么不用 cv2.VideoWriter：
    OpenCV 默认的 "mp4v" fourcc 是 MPEG-4 Part 2 编码，Windows 播放器 /
    macOS QuickTime 默认不支持，会报"文件损坏"。必须用 H.264。

本工具与 LeRobot 自身的实现保持一致（lerobot.utils.io_utils.write_video），
同样基于 PyAV + libx264 + yuv420p —— 这是兼容性最好的组合。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_video_h264(
    video_path: str | Path,
    frames: list[np.ndarray],
    fps: int = 30,
    crf: int = 23,
) -> None:
    """把 RGB HWC uint8 帧序列写为 H.264 MP4（yuv420p，全平台可播放）。

    Args:
        video_path: 输出路径。
        frames: RGB uint8 (H, W, 3) 帧列表。
        fps: 帧率。
        crf: x264 质量参数，越小越清晰（18~28 常用，默认 23）。
    """
    import av

    if not frames:
        raise ValueError("frames 为空，无法写入视频")

    video_path = str(video_path)
    height, width = frames[0].shape[:2]
    if height % 2 or width % 2:
        # yuv420p 要求偶数尺寸
        height -= height % 2
        width -= width % 2

    # movflags=+faststart: 把 moov 索引放到文件头, 浏览器/流式播放器也能即点即播
    with av.open(video_path, mode="w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        for arr in frames:
            if (arr.shape[0], arr.shape[1]) != (height, width):
                arr = arr[:height, :width]
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def remux_faststart(src: str | Path, dst: str | Path | None = None) -> Path:
    """无损重封装: 只把 moov 索引移到文件头(faststart), 不重新编码。

    dst 为 None 时原地替换。
    """
    import av

    src = Path(src)
    in_place = dst is None
    dst = src.with_name(src.stem + ".__fs_tmp__.mp4") if in_place else Path(dst)

    with av.open(str(src)) as inc:
        with av.open(str(dst), mode="w", options={"movflags": "+faststart"}) as outc:
            in_stream = inc.streams.video[0]
            out_stream = outc.add_stream_from_template(in_stream)
            for packet in inc.demux(in_stream):
                if packet.dts is None:  # 跳过多余的空包
                    continue
                packet.stream = out_stream
                outc.mux(packet)

    if in_place:
        dst.replace(src)
        return src
    return dst


def reencode_to_h264(src: str | Path, dst: str | Path | None = None, fps: int | None = None) -> Path:
    """把已有视频（任意编码，如 mp4v/AV1）重编码为 H.264。

    dst 为 None 时原地替换（先写临时文件再改名）。
    fps 为 None 时沿用源视频帧率。
    """
    import av

    src = Path(src)
    in_place = dst is None
    dst = src.with_name(src.stem + ".__h264_tmp__.mp4") if in_place else Path(dst)

    with av.open(str(src)) as container:
        stream = container.streams.video[0]
        out_fps = fps if fps is not None else (int(round(float(stream.average_rate))) if stream.average_rate else 30)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]

    write_video_h264(dst, frames, fps=out_fps)
    if in_place:
        dst.replace(src)  # 原子替换回原文件名
        return src
    return dst


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        out = reencode_to_h264(p)
        print(f"重编码完成: {out}")
