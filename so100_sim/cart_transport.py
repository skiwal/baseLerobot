#!/usr/bin/env python
"""套筒随机放置 + 抓取后搬运再插装(视角/位置鲁棒的移动装配)。

约束(与全局一致): 策略输入只有 RGB-D 图像 + 机械臂/底盘自身数据(关节、夹爪、底盘里程计),
不读任何仿真真值(不读 env.sleeve_pose/rod_* 的 GT, 不用 geom_xpos)。套筒由"环境"随机放置
(竖姿不变、位置变), 策略靠感知找到它并决定停靠位。

流程: 远处 -> 开到勘测/装配位感知套筒 -> 由"感知套筒位 - 舒适抓取相对位姿"算抓取停靠位 ->
开去抓起并抬起 -> 夹持开往装配停靠位(细杆固定于此, 套筒随底盘搬到 proven 插入位姿) ->
在装配位感知细杆 -> 插入。

用法: python cart_transport.py [seed]
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import SO100SleeveEnv, Z_AXIS, GRIP_FORCE_TARGET
from perception_api import perceive_scene
from cart_assembly import (drive_cart_to, drive_cart_holding, park_arm,
                           PICKUP_REL_XY, ASSEMBLY_DOCK)
from grasp_perceived import (grasp_sleeve_perceived, insert_sleeve_perceived, goto_p)

# 套筒随机放置范围(世界系 x,y): 台面盒为 x∈[-0.165,0.165], y∈[-0.28,-0.16](顶面 z=0.08);
# 中央有托座(x≈0)与细杆座(x=-0.06)。取右侧空旷、离边缘留~1.5cm 余量的稳定区, 且与杆分开
# (保证抓起后需开一段路运到杆处)。这是"环境/任务设定"侧随机化; 策略不读该真值, 只用 RGB-D 感知。
SLEEVE_RANGE = ((0.03, 0.14), (-0.255, -0.195))


# 感知系统偏差标定: 实测(多套筒位、多停靠位一致) perceived = true + (dx≈+1.5mm, dy≈-5mm);
# 故 true_xy ≈ perceived_xy + 该校正。用于停靠位计算/视觉伺服, 补偿 RGB-D 中心估计的系统偏差。
PERCEPTION_CORR = np.array([-0.0015, 0.005])


def _sleeve_xy_corrected(env):
    """感知套筒中心并做系统偏差校正 -> 世界系 (x,y,z)。"""
    det = perceive_scene(env, "base_rgbd")
    if "sleeve" not in det:
        return None
    sp = np.asarray(det["sleeve"]["pos"], float)
    sp[:2] = sp[:2] + PERCEPTION_CORR
    return sp


def run_once(env, verbose=True):
    # 1) 开到勘测/装配位(臂收起), 相机视野覆盖台面 -> 感知套筒(带偏差校正) + 细杆。
    #    细杆固定不动, 就在此"无遮挡"时一次性测准(抓持套筒后臂/套筒会悬在杆附近遮挡看杆的视线)。
    drive_cart_to(env, *ASSEMBLY_DOCK)
    det = perceive_scene(env, "base_rgbd")
    if "sleeve" not in det or "rod" not in det:
        return False, "勘测位未感知到套筒/细杆"
    sp = np.asarray(det["sleeve"]["pos"], float); sp[:2] = sp[:2] + PERCEPTION_CORR
    rod = det["rod"]
    rod_axis = np.asarray(rod["axis"], float); rod_axis /= np.linalg.norm(rod_axis)
    rod_tip = np.asarray(rod["tip"], float)
    rod_base = np.asarray(rod["base"], float)

    # 2) 视觉伺服校正停靠位: dock = 校正后套筒位 - 舒适抓取相对位姿; 到位后重感知重算, 迭代收敛,
    #    让套筒真正落到 PICKUP_REL(补偿感知偏差 -> 固定构型抓取得以对准)。
    dock = sp[:2] - PICKUP_REL_XY
    for it in range(3):
        drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
        sp2 = _sleeve_xy_corrected(env)
        if sp2 is None:
            break
        nd = sp2[:2] - PICKUP_REL_XY
        if verbose:
            print(f"[运输] 停靠校正 it{it}: 停靠位 {np.round(dock, 3)} -> {np.round(nd, 3)}")
        if np.linalg.norm(nd - dock) < 0.003:
            dock = nd
            break
        dock = nd
    drive_cart_to(env, float(dock[0]), float(dock[1]), 0.0)
    if verbose:
        print(f"[运输] 抓取停靠位 {np.round(dock, 3)} (套筒被送到舒适抓取相对位姿)")

    # 3) 用 proven 固定构型抓取 + 抬起(套筒已在 proven 相对位姿, PROVEN_*_SEED 与底盘绝对位置无关)
    try:
        info = grasp_sleeve_perceived(env, verbose=verbose)
    except RuntimeError as e:
        return False, f"抓取异常: {e}"
    # 夹持力合理区间: 过低=没夹住, 过高(数千 N)=夹偏卡到棱角/桌座
    if not (info["reached"] and 10.0 < info["grip_force"] < 600.0):
        return False, f"抓取未稳(夹持力={info['grip_force']:.1f}N)"

    # 4) 夹持开往装配停靠位(细杆固定); 套筒随底盘搬到 proven 插入位姿
    if verbose:
        print(f"[运输] 夹持套筒(力={info['grip_force']:.1f}N)开往装配位 {ASSEMBLY_DOCK} ...")
    drive_cart_holding(env, *ASSEMBLY_DOCK)

    # 4b) 预纠竖直: 用夹爪本体实测套筒轴, 若抓取/搬运中倾斜则原地转回竖直。
    #     proven 插入的段1(竖直平移到 pre)假设起始轴=Z; 套筒预倾斜会使其转不动杆轴 -> 先扶正。
    cur = env.arm_qpos().astype(float); sol_prev = cur[:5].copy()
    sc = env.grasp_point() + env.jaw_rot() @ env.grasp_off_local
    sa = env.jaw_rot() @ env.grasp_axis_local
    ang = float(np.degrees(np.arccos(np.clip(abs(sa @ Z_AXIS), 0, 1))))
    if ang > 3.0:
        goto_p(env, sc, sa, sc, Z_AXIS, 0.8, env.jaw_grip, cur, sol_prev,
               mode="pose", pre_offset=0.0)
        ang2 = float(np.degrees(np.arccos(np.clip(abs((env.jaw_rot() @ env.grasp_axis_local) @ Z_AXIS), 0, 1))))
        if verbose:
            print(f"[运输] 套筒轴预纠竖直: {ang:.1f}° -> {ang2:.1f}°")

    # 5+6) 插入: 用勘测位测准的杆位姿(杆固定不变; 不在此处重感知, 避免悬停套筒/臂遮挡看杆)
    rinfo = insert_sleeve_perceived(env, rod_axis, rod_tip, rod_base, verbose=verbose)
    return rinfo["success"], rinfo


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = np.random.default_rng(seed)
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)

    # 环境随机化: 套筒竖姿不变、位置在范围内随机; 细杆固定。策略只感知不读真值。
    env.reset(tilt_deg=35.0, azim_deg=0.0, rng=rng, sleeve_xy_range=SLEEVE_RANGE)

    # 起点: 距离平台一段距离(小车从远处出发)
    env.set_cart_pose(0.45, 0.0, 0.0); park_arm(env)
    print(f"[导航] 起点={np.round(env.cart_pose, 3)} (套筒由环境随机放置, 策略仅感知)")

    ok, info = run_once(env)
    if ok:
        print(f"[结果] ✅ 随机位 抓取-搬运-插装 成功: 重叠 {info['insert_overlap_mm']:.1f}mm, "
              f"轴夹角 {info['align_axis_deg']:.1f}°")
    else:
        print(f"[结果] ❌ 失败: {info}")


if __name__ == "__main__":
    main()
