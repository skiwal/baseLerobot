#!/usr/bin/env python
"""P3 抓取诊断: 复现 _pickup 的停靠伺服后直接调 grasp_sleeve_perceived, 打印抓取验证细节
(reached/blocked/实际jaw角/下探pad中点z/套筒顶z), 定位抓取失败种子(如 57)的失败环节。

用法: MUJOCO_GL=egl python diag_grasp.py 57 0 3
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "perception" / "python"))
from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import park_arm, drive_cart_to, ASSEMBLY_DOCK, PICKUP_REL_XY
import cart_transport as CT
from grasp_perceived import grasp_sleeve_perceived, grip_p, JAW_OPEN


def servo_and_grasp(env, verbose):
    sp = CT._sleeve_xy_corrected(env)
    if sp is None:
        return None, "未感知套筒"
    dock = sp[:2] - PICKUP_REL_XY
    for _ in range(3):
        drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
        sp2 = CT._sleeve_xy_corrected(env)
        if sp2 is None:
            break
        nd = sp2[:2] - PICKUP_REL_XY
        if np.linalg.norm(nd - dock) < 0.003:
            dock = nd; break
        dock = nd
    drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
    try:
        info = grasp_sleeve_perceived(env, verbose=verbose)
    except RuntimeError as e:
        return None, f"抓取异常: {e}"
    return info, None


def diag(seed, max_retry=3):
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    results = []
    try:
        rng = np.random.default_rng(seed)
        env.reset(tilt_deg=35.0, azim_deg=0.0, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
        drive_cart_to(env, *ASSEMBLY_DOCK)
        for attempt in range(max_retry):
            info, err = servo_and_grasp(env, verbose=(attempt == 0))
            if info is None:
                results.append((attempt, False, False, None, err)); break
            ja = float(env.data.qpos[env.qadr["Jaw"]])
            jg = float(env.jaw_grip)
            reached = info["reached"]; blocked = (ja - jg) > 0.05
            results.append((attempt, reached, blocked, info.get("approach_mid_z"), None))
            if reached and blocked:
                break
            # 未夹稳 -> 松爪重试
            cur = env.arm_qpos().astype(float)
            sc = env.grasp_point() + env.jaw_rot() @ env.grasp_off_local
            grip_p(env, sc, JAW_OPEN, 0.6, cur, cur[:5])
    except Exception as e:
        results.append(("EXC", str(e)))
    finally:
        env.close()
    return results


if __name__ == "__main__":
    seeds = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [57, 0, 3]
    for s in seeds:
        rs = diag(s)
        last = rs[-1]
        ok = any(r[1] and r[2] for r in rs if isinstance(r[0], int))
        print(f"seed {s}: 成功={ok} 历程=" +
              ", ".join(f"#{r[0]} reached={r[1]} blocked={r[2]} padz={r[3]}" for r in rs if isinstance(r[0], int)))
