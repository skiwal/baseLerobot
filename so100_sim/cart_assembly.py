#!/usr/bin/env python
"""底盘(小车)自主装配: 远处出发 -> 自身里程计导航到平台前停靠位 -> 偏航朝向 ->
RGB-D 感知确认目标 -> 自行决定开始装配; 失败重试拾取, 不行则重定位, 仍失败判负。

约束(与全局一致): 任务感知只用 RGB-D; 小车"自己在哪"用自身里程计(等同轮编码器,
即 env.cart_pose 由底盘指令直接记录)。机械臂+根部相机整体随底盘平移+偏航(见 env.set_cart_pose),
IK 用 MuJoCo 正运动学自动适配世界系目标, 故抓取/插装天然底盘无关。

底层控制/插装直接复用 grasp_perceived 的已验证控制器(control_step / goto_p /
goto_joint / insert_sleeve_perceived), 仅把"抓取"改造成相对感知套筒中心的 world-IK(不再用固定关节种子)。
"""
import sys
from pathlib import Path
import math
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import (
    SO100SleeveEnv, Z_AXIS, X_AXIS, JAW_OPEN, PRE_GRASP_OFFSET, LIFT_DZ,
    JAW_FORCE_LIMIT, JAW_INSERT_LIMIT, GRIP_FORCE_TARGET, FPS,
    HOVER_BACK, HOVER_DZ, IK_RNG, ROD_LEN,
)
from grasp_perceived import (
    control_step, goto_p, goto_joint, insert_sleeve_perceived,
)
from perception_api import perceive_scene

PARK_QPOS = np.array([0.0, 1.2, 2.2, -1.0, -1.57])   # 缩回抬高: 随底盘平移时不挡视线/不拖地


def park_arm(env):
    env.set_arm(PARK_QPOS, JAW_OPEN)
    mujoco.mj_forward(env.model, env.data)


def drive_cart_to(env, tx, ty, tyaw, step_m=0.01, dt_steps=8):
    """里程计导航: 把底盘从当前位姿平滑开到 (tx,ty,tyaw), 期间臂保持缩回构型随之平移。"""
    park_arm(env)
    cx, cy, cyaw = env.cart_pose
    dist = max(abs(tx - cx), abs(ty - cy))
    n = max(1, int(dist / step_m))
    for i in range(1, n + 1):
        a = i / n
        x = cx + (tx - cx) * a
        y = cy + (ty - cy) * a
        yaw = cyaw + (tyaw - cyaw) * a
        env.set_cart_pose(x, y, yaw)
        for _ in range(dt_steps):
            env.step(env.arm_qpos())        # 保持臂构型(随底盘刚性平移)
    env.set_cart_pose(tx, ty, tyaw)
    for _ in range(30):
        env.step(env.arm_qpos())


def grasp_sleeve_base_indep(env, hover_qpos=None, verbose=True):
    """底盘无关抓取: 全程 world-IK 到"套筒感知中心 + 竖直偏移"的目标(不依赖固定关节种子),
    直下/抬起都用相对套筒中心的 world 目标, 因此从任意可达底盘位姿都能抓。"""
    det = perceive_scene(env, "base_rgbd")
    if "sleeve" not in det:
        raise RuntimeError("感知未检出套筒")
    sp = det["sleeve"]["pos"].astype(float)
    sa = det["sleeve"]["axis"].astype(float)
    if "rod" not in det:
        raise RuntimeError("感知未检出细杆(插装需要杆位姿)")
    rod = det["rod"]
    rod_axis = rod["axis"].astype(float); rod_axis /= np.linalg.norm(rod_axis)
    rod_tip = rod["tip"].astype(float)
    rod_base = rod["base"].astype(float)
    env.sleeve_pose = lambda: (sp.copy(), sa.copy())

    # 悬停偏移随底盘 yaw 旋转: 偏航后"后上方"是相对底盘的, 必须用底盘坐标系的偏移(否则偏航时落点偏掉)
    yaw = float(env.cart_pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    hoff = np.array([-HOVER_BACK * s, HOVER_BACK * c, HOVER_DZ])

    # 悬停构型: world-IK 到套筒上方。种子用中性缩回构型(非 (0,0,0) 专属的 GRASP_SEED),
    # 保证在任何底盘位姿都落到"自然、安全"分支(避免 GRASP_SEED 在非原位姿把末端甩到偏心/碰撞分支)。
    if hover_qpos is None:
        hover = sp + hoff
        sol_h = env.ik_restart(hover, Z_AXIS, PARK_QPOS, axis_local=Z_AXIS, mode="pose",
                               n_rand=24, max_step=0.05, n_iter=300, clamp_ref=X_AXIS, rng=IK_RNG)
        hover_qpos = np.concatenate([sol_h, [JAW_OPEN]])
    env.set_arm(hover_qpos[:5], float(hover_qpos[5]))
    mujoco.mj_forward(env.model, env.data)
    for _ in range(30):
        env.step(hover_qpos)
    cur = env.arm_qpos().astype(float); sol_prev = cur[:5].copy()

    # 直下到套筒(底盘无关): 从悬停(已在套筒正上方、安全分支)起, 笛卡尔直线小步 IK 直下到 sp,
    # 种子连续自悬停 + clamp_ref 锁滚转(不跳偏心分支); 路径竖直、不先抬臂扫掠, 避免 cramped 位姿下
    # 前臂扫到套筒把它弹飞。等价于 proven 的 set_arm(PROVEN_GRASP_SEED)+goto_p 直下, 但底盘无关。
    hover_c = sp + hoff
    sol = sol_prev.copy()                                  # 当前=悬停构型(已安全在套筒上方)
    N = 50
    for i in range(1, N + 1):
        a = i / N
        tgt = hover_c * (1.0 - a) + sp * a
        sol = env.pose_ik(tgt, Z_AXIS, q_seed=sol, axis_local=env.grasp_axis_local,
                          n_iter=80, max_step=0.02, clamp_ref=X_AXIS)
        env.set_arm(sol, JAW_OPEN); mujoco.mj_forward(env.model, env.data)
    for _ in range(25):
        env.step(np.concatenate([sol, [JAW_OPEN]]))
    cur = env.arm_qpos().astype(float)
    pf = env.data.geom_xpos[env.pad_f]; pm = env.data.geom_xpos[env.pad_m]
    approach_mid = (pf + pm) / 2.0
    if verbose:
        print(f"[接近] pad中点={np.round(approach_mid, 3)} 目标sp={np.round(sp, 3)} "
              f"抓取点误差={np.linalg.norm(env.grasp_point() - sp) * 1000:.1f}mm")

    # 闭爪(保持构型, 仅收 jaw)
    n_close = int(0.8 * FPS)
    for k in range(n_close):
        j = JAW_OPEN + (env.jaw_grip - JAW_OPEN) * (k / (n_close - 1))
        tgt = cur.copy(); tgt[5] = j
        env.step(tgt)
    cur[5] = env.jaw_grip
    if verbose:
        print(f"[闭爪] 夹持力={env.grip_force():.2f}N")

    env.measure_grasp()
    env.sleeve_pose = lambda: (env.grasp_point().copy(), Z_AXIS.copy())

    # 抬起: world-IK 到 sp + Z*LIFT_DZ(底盘无关)
    lift_target = sp + Z_AXIS * LIFT_DZ
    sol_l = env.pose_ik(lift_target, Z_AXIS, q_seed=cur[:5], axis_local=env.grasp_axis_local,
                        n_iter=300, max_step=0.05)
    goto_joint(env, sol_l, 0.8, env.jaw_grip, cur, sol_prev)
    for _ in range(30):
        env.step(cur)

    gf = float(env.grip_force())
    reached = float(approach_mid[2]) < sp[2] + 0.02
    return {"target_pos": sp, "grip_force": gf, "approach_mid_z": float(approach_mid[2]),
            "reached": reached, "rod_axis": rod_axis, "rod_tip": rod_tip, "rod_base": rod_base}


def main():
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    env.reset(tilt_deg=35.0, azim_deg=0.0)

    # 起点/停靠位可由命令行覆盖: python cart_assembly.py start_x start_y start_yaw dock_x dock_y dock_yaw
    a = [float(x) for x in sys.argv[1:]]
    start = tuple(a[0:3]) if len(a) >= 3 else (0.45, 0.0, 0.0)
    dock = tuple(a[3:6]) if len(a) >= 6 else (0.0, 0.0, 0.0)

    # 起点: 距离平台一段距离(小车从远处出发)
    env.set_cart_pose(*start)
    park_arm(env)
    print(f"[导航] 起点底盘位姿={np.round(env.cart_pose, 3)}  (平台在世界系固定)")

    # 停靠位: 平台前的"合适工作位姿"——套筒落在 SO-100 臂可达包络内、根部相机能看清套筒+杆。
    # (默认 (0,0,0); 平台在世界系固定, 底盘需开到该合适位姿才能无碰撞抓取, 可用命令行覆盖)
    print(f"[导航] 里程计开向停靠位 {dock} ...")
    drive_cart_to(env, *dock)
    print(f"[导航] 已到达停靠位 {np.round(env.cart_pose, 3)}")

    # 决策: 用 RGB-D 感知确认平台(套筒+细杆)在视野内且可达 -> 自行决定开始装配
    det = perceive_scene(env, "base_rgbd")
    if "sleeve" in det and "rod" in det:
        print(f"[决策] 感知到套筒({np.round(det['sleeve']['pos'],3)})与细杆, 决定开始装配")
    else:
        print(f"[决策] 未感知到完整目标(平台可能出视野/遮挡), 重定位后再决定")
        drive_cart_to(env, 0.0, 0.05, 0.0)
        det = perceive_scene(env, "base_rgbd")
        if "sleeve" not in det or "rod" not in det:
            print("[结果] ❌ 重定位后仍无法感知目标")
            return

    # 决策后: 装配 + 失败重试(重定位回原始停靠位再尝试, 不行则判负)
    for attempt in range(4):
        try:
            info = grasp_sleeve_base_indep(env, verbose=True)
        except RuntimeError as e:
            print(f"[重试{attempt}] 感知/抓取异常: {e}; 重定位后重试")
            drive_cart_to(env, *dock)
            continue
        ok_g = info["reached"] and info["grip_force"] > GRIP_FORCE_TARGET
        if not ok_g:
            print(f"[重试{attempt}] 抓取未稳(下到套筒={info['reached']}, "
                  f"夹持力{info['grip_force']:.1f}N); 重定位后重试")
            drive_cart_to(env, *dock)
            continue
        rinfo = insert_sleeve_perceived(env, info["rod_axis"], info["rod_tip"], info["rod_base"], verbose=True)
        if rinfo["success"]:
            print(f"[结果] ✅ 小车自主装配成功 (停靠位={np.round(env.cart_pose,3)}, "
                  f"重叠 {rinfo['insert_overlap_mm']:.1f}mm, 轴夹角 {rinfo['align_axis_deg']:.1f}°)")
            return
        else:
            print(f"[重试{attempt}] 插装失败; 重定位后重试")
            drive_cart_to(env, *dock)
            continue
    print("[结果] ❌ 多次尝试仍失败")


if __name__ == "__main__":
    main()
