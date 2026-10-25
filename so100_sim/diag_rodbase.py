#!/usr/bin/env python
"""量化 rod_base 感知误差结构: 对一批种子分解感知 base/tip 相对 GT 的轴向/垂直轴误差,
确认 seed 97 类"重叠度量失真"来自轴向偏移(需标定)还是垂直偏移(投影到标称线即可修)。

用法: MUJOCO_GL=egl python diag_rodbase.py
"""
import sys, time
from pathlib import Path
import numpy as np
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "perception" / "python"))
from so100_sleeve_env import SO100SleeveEnv
from cart_assembly import park_arm, drive_cart_to, ASSEMBLY_DOCK
from perception_api import perceive_scene
import cart_transport as CT


def run_seed(seed):
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    r = {"seed": seed, "axis_err": None, "b_ax": None, "b_perp": None,
         "t_ax": None, "t_perp": None, "sanity": None, "msg": ""}
    try:
        rng = np.random.default_rng(seed)
        env.reset(tilt_deg=35.0, azim_deg=0.0, rng=rng, sleeve_xy_range=CT.SLEEVE_RANGE)
        env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
        drive_cart_to(env, *ASSEMBLY_DOCK)
        det = perceive_scene(env, "base_rgbd", expected_tilt_deg=35.0, expected_azim_deg=0.0)
        if "rod" not in det:
            r["msg"] = "no rod"; return r
        rod = det["rod"]
        pa = np.asarray(rod["axis"], float); pa /= np.linalg.norm(pa)
        pb = np.asarray(rod["base"], float); pt = np.asarray(rod["tip"], float)
        r["sanity"] = rod.get("sanity_ok", None)
        # GT
        ga = env.rod_axis(); gb = env.rod_base(); gt = env.rod_tip()
        r["axis_err"] = float(np.degrees(np.arccos(np.clip(abs(pa @ ga), 0, 1))))
        # 误差分解(沿 GT 轴)
        eb = pb - gb; et = pt - gt
        r["b_ax"] = float(eb @ ga * 1000)
        r["b_perp"] = float(np.linalg.norm(eb - (eb @ ga) * ga) * 1000)
        r["t_ax"] = float(et @ ga * 1000)
        r["t_perp"] = float(np.linalg.norm(et - (et @ ga) * ga) * 1000)
    except Exception as e:
        r["msg"] = f"EXC {type(e).__name__}: {e}"
    finally:
        env.close()
    return r


def main():
    seeds = [0, 3, 7, 10, 12, 20, 26, 30, 54, 57, 71, 91, 94, 97, 98, 5, 18, 59, 84]
    t0 = time.time()
    with open(ROOT / "out" / "diag_rodbase.txt", "w") as f:
        f.write(f"seed axis_err  b_ax  b_perp  t_ax  t_perp  sanity  (mm, 沿GT轴)\n")
        with Pool(8) as p:
            for r in p.imap_unordered(run_seed, seeds):
                f.write(f"{r['seed']:>4} {r['axis_err']:6.2f} {r['b_ax'] or 0:6.1f} "
                        f"{r['b_perp'] or 0:6.1f} {r['t_ax'] or 0:6.1f} {r['t_perp'] or 0:6.1f} "
                        f"{str(r['sanity']):>5}  {r['msg']}\n")
                f.flush()
    print(f"[{time.time()-t0:.0f}s] 见 out/diag_rodbase.txt")


if __name__ == "__main__":
    main()
