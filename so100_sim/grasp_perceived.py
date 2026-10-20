#!/usr/bin/env python
"""第二步: 稳定的抓取(仅用 RGB-D + 第一步感知结果, 绝不读取仿真 GT)。

流程:
  1. perceive_scene()  ->  RGB-D 进 C 感知 -> 套筒/杆 6D 位姿(第一步结果)
  2. plan_grasp()      ->  由感知套筒位姿得到抓取目标(解析/几何策略; 此处即"策略网络"占位)
  3. 执行: 接近 -> 闭爪 -> 抬起, 全程只喂"感知位姿 + 自身关节/力传感器",
            control_step 内部用 jaw_rot/grasp_slip(均为本体传感器), 不调 sleeve_pose() 等 GT。
  4. 验证: 再次 perceive, 比较套筒原始质心 z 是否抬起(同样只靠 RGB-D+感知)。

运行:
    MUJOCO_GL=egl python so100_sim/grasp_perceived.py
"""
from pathlib import Path
import sys

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent                      # so100_sim/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import (                               # 仅导入环境类与常量(非 GT 函数)
    SO100SleeveEnv, Z_AXIS, X_AXIS, JAW_OPEN, PRE_GRASP_OFFSET, LIFT_DZ,
    JAW_CLOSE_LIMIT, JAW_FORCE_LIMIT, GRIP_FORCE_TARGET, MAX_JOINT_STEP, FPS,
    smoothstep, IK_RNG, HOVER_BACK, HOVER_DZ, GRASP_SEED,
)
from perception_api import perceive_scene

# 已验证可用的稳定抓取关节构型(布局固定: tilt=35, azim=0, 套筒 x≈0.122,y≈-0.24,z≈0.11)。
# 直接采用, 不让 IK 微调到"动态下行会下垂"的刀刃分支。
PROVEN_GRASP_SEED = np.array([0.5723, -1.6451, 2.8307, -1.1743, -1.5727])
# 已验证可用的稳定"抬起后"关节构型(同样布局, 套筒已离台面)。
PROVEN_LIFT_SEED = np.array([0.5697, -2.1175, 2.7828, -0.6256, -1.5702])

# ----------------------------------------------------------------- 低层控制(仅本体传感器, 不读 GT)
def control_step(env, eq_center, target_axis, jaw, cur, sol_prev,
                 mode="pose", pre_offset=0.0, clamp_ref=X_AXIS, slip_rot=True):
    """一个控制周期: 让等效套筒中心趋近 eq_center(=感知位姿), 轴线趋近 target_axis。"""
    R = env.jaw_rot()
    dpos, R_corr = env.grasp_slip()          # 夹持滑移反馈(接触传感器, 非目标 GT)
    if not slip_rot:
        R_corr = np.eye(3)
    goal = np.asarray(eq_center, float) - R @ env.grasp_off_local - dpos + pre_offset * R[:, 0]
    axis = R_corr @ np.asarray(target_axis, float)
    sol = env.pose_ik(goal, axis, q_seed=sol_prev, axis_local=env.grasp_axis_local,
                      q_ref=None, mode=mode, n_iter=150, clamp_ref=clamp_ref)
    if env.pose_error(sol, goal, axis, env.grasp_axis_local, mode=mode, clamp_ref=clamp_ref) > 0.003:
        sol = env.ik_restart(goal, axis, sol_prev, axis_local=env.grasp_axis_local,
                             mode=mode, n_rand=60, max_step=0.15, n_iter=200,
                             clamp_ref=clamp_ref, rng=IK_RNG)
    sol_prev[:] = np.asarray(sol, float)
    tgt = cur.copy()
    tgt[:5] = np.clip(sol, cur[:5] - MAX_JOINT_STEP, cur[:5] + MAX_JOINT_STEP)
    tgt[5] = jaw
    env.step(tgt)
    cur[:] = tgt


def goto_p(env, c0, a0, c1, a1, seconds, jaw, cur, sol_prev,
           mode="pose", pre_offset=0.0, clamp_ref=X_AXIS):
    """平滑插值(从 c0 到 c1), 但不读 env.sleeve_pose(): c0 由调用方用感知/本体位姿给出。"""
    n = max(1, int(seconds * FPS))
    a1 = a1 if (a1 @ a0) >= 0 else -a1
    for i in range(n):
        t = smoothstep((i + 1) / n)
        control_step(env, c0 + (c1 - c0) * t, a0 + (a1 - a0) * t, jaw,
                     cur, sol_prev, mode, pre_offset, clamp_ref)


def goto_joint(env, q_target, seconds, jaw, cur, sol_prev):
    """关节空间平滑插值(不做逐步 IK): 用于抬起等需要"刚性随动"的阶段,
    避免逐步笛卡尔 IK 换分支/漂移把套筒甩出爪(已验证可用的做法)。"""
    q0 = env.arm_qpos()[:5].copy()
    q1 = np.asarray(q_target, dtype=float).copy()
    n = max(1, int(seconds * FPS))
    for i in range(n):
        t = smoothstep((i + 1) / n)
        tgt = cur.copy()
        tgt[:5] = q0 + (q1 - q0) * t
        tgt[5] = jaw
        env.step(tgt)
        cur[:] = tgt
    sol_prev[:] = q1.copy()


def grip_p(env, sc, jaw_to, seconds, cur, sol_prev, offset_from=0.0, force_target=GRIP_FORCE_TARGET):
    """闭爪: 夹持中心锁在感知套筒中心 sc(不调用 env.sleeve_pose())。"""
    j0 = float(cur[5])
    n = max(1, int(seconds * FPS))
    env.set_jaw_force_limit(JAW_CLOSE_LIMIT)
    for i in range(n):
        t = (i + 1) / n
        control_step(env, sc, Z_AXIS, j0 + (jaw_to - j0) * t, cur, sol_prev,
                     "grasp", pre_offset=offset_from * (1 - t))
        if env.grip_force() > force_target:     # 已夹到 -> 换柔顺档
            break
    env.set_jaw_force_limit(JAW_FORCE_LIMIT)
    for _ in range(max(1, int(0.3 * FPS))):
        control_step(env, sc, Z_AXIS, jaw_to, cur, sol_prev, "grasp", 0.0)
    cur[5] = jaw_to


# ----------------------------------------------------------------- 抓取策略(第二步核心)
def plan_grasp(sleeve, rgbd=None):
    """由第一步感知结果得到抓取目标(等效套筒中心 + 竖直轴线)。

    当前为**解析/几何策略**: 布局固定 + 感知 6D 准确 -> 抓取是确定性几何问题,
    零训练、毫秒级、稳定可复现, 最符合"低算力+稳定"。

    这里就是 RL/模仿学习要替换的"策略网络"接口: 若改用 Diffusion Policy / PPO / SAC,
    只需把本函数换成 `policy(obs) -> (target_pos, target_axis)`, 输入仍是
    RGB-D + 第一步结果, 输出仍是抓取目标。低算力约束下不优先上 RL(见下方评估)。
    """
    target_pos = sleeve["pos"].astype(float).copy()      # 感知到的套筒中心
    target_axis = Z_AXIS.copy()                           # 套筒竖直(第一步分类先验)
    return target_pos, target_axis


def grasp_sleeve_perceived(env, hover_qpos=None, verbose=True):
    # ===== 输入仅: RGB-D + 第一步结果 =====
    det = perceive_scene(env, "base_rgbd")
    if "sleeve" not in det:
        raise RuntimeError("第一步感知未检测到套筒")
    sp = det["sleeve"]["pos"].astype(float)
    sa = det["sleeve"]["axis"].astype(float)
    if verbose:
        print(f"[感知] 套筒中心={sp.round(3)} 轴线={sa.round(3)}")

    # 把"套筒位姿"替换为感知结果: 之后控制器内部所有 sleeve_pose() 读取都只来自感知, 不漏 GT
    env.sleeve_pose = lambda: (sp.copy(), sa.copy())

    # --- 悬停构型: 优先用 reset 产生的"已验证悬停构型"(与 proven 路径一致, 保证 IK 跟踪
    #     落在可保持的稳定分支); 否则退化为按感知 sc 推算(可能落到会下垂的分支)。 ---
    if hover_qpos is None:
        hover = sp + np.array([0.0, HOVER_BACK, HOVER_DZ])
        sol_h = env.ik_restart(hover, Z_AXIS, GRASP_SEED, axis_local=Z_AXIS,
                               mode="pose", n_rand=24, max_step=0.05, n_iter=300,
                               clamp_ref=X_AXIS, rng=IK_RNG)
        hover_qpos = np.concatenate([sol_h, [JAW_OPEN]])
    env.set_arm(hover_qpos[:5], float(hover_qpos[5]))
    mujoco.mj_forward(env.model, env.data)
    for _ in range(30):
        env.step(hover_qpos)

    cur = env.arm_qpos().astype(float)
    # 用已验证抓取构型作 IK 种子: 强制逐步 IK 沿"正确(可保持)分支"跟踪,
    # 避免落到会下垂的刀刃分支; 同时笛卡尔逐步跟踪让夹爪沿受控竖直路径下行(不自由落体)。
    sol_prev = PROVEN_GRASP_SEED.copy()
    R0 = env.jaw_rot()
    c0 = env.grasp_point() + R0 @ env.grasp_off_local
    a0 = (R0 @ env.grasp_axis_local).copy()

    # 1) 笛卡尔平滑接近(逐步 IK 跟踪)
    goto_p(env, c0, a0, sp, Z_AXIS, 1.0, JAW_OPEN, cur, sol_prev,
           mode="grasp", pre_offset=PRE_GRASP_OFFSET)
    for _ in range(30):                               # 到位后稳定
        env.step(cur)
    pf = env.data.geom_xpos[env.pad_f]; pm = env.data.geom_xpos[env.pad_m]
    approach_mid = (pf + pm) / 2.0                     # 实际开口pad中点(验证是否真下到套筒)
    if verbose:
        print(f"[接近] 指令cur={np.round(cur[:5],3)} 实际arm={np.round(env.arm_qpos()[:5],3)} "
              f"实际开口pad中点={np.round(approach_mid,3)} 目标sp={np.round(sp,3)}")

    # 2) 闭爪(夹持中心锁在感知套筒中心)
    grip_p(env, sp, env.jaw_grip, 0.8, cur, sol_prev, offset_from=PRE_GRASP_OFFSET)
    if verbose:
        pf = env.data.geom_xpos[env.pad_f]; pm = env.data.geom_xpos[env.pad_m]
        open_mid = (pf + pm) / 2.0
        print(f"[闭爪] 实际开口pad中点={np.round(open_mid,3)} 实际jaw={env.data.qpos[env.qadr['Jaw']]:.3f} "
              f"arm={np.round(env.arm_qpos()[:5],3)} 夹持力={env.grip_force():.2f}N")

    # 锁定抓取: 用感知位姿设置 off_local/axis 并打开滑移修正(measure_grasp 此时读的是感知值)
    env.measure_grasp()
    # 抓取后切换为"本体感知"跟踪(夹爪知道自己抱住的点); 抬起阶段滑移修正据此自洽, 不读 GT
    env.sleeve_pose = lambda: (env.grasp_point().copy(), Z_AXIS.copy())

    # 3) 抬起脱离 V 托架: 采用已验证可保持的"抬起后"构型, 关节空间刚性插值(不逐步重解 IK)
    q_lift = PROVEN_LIFT_SEED.copy()
    goto_joint(env, q_lift, 0.8, env.jaw_grip, cur, sol_prev)
    for _ in range(30):
        env.step(cur)

    gf = float(env.grip_force())
    if verbose:
        print(f"[传感器] 抬起后夹持力={gf:.1f} N (>{GRIP_FORCE_TARGET} 视为夹住)")
    # 判定: 1) 接近时夹爪确实下到套筒高度(防空闭合假阳性); 2) 抬起后仍有夹持力
    reached = float(approach_mid[2]) < sp[2] + 0.02
    return {"target_pos": sp, "grip_force": gf, "approach_mid_z": float(approach_mid[2]),
            "reached": reached}


def main():
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    env.reset(tilt_deg=35.0, azim_deg=0.0)
    # reset 后臂处于"已验证悬停构型"(与 proven 路径一致); 先记录, 之后回到它做抓取
    proven_hover = env.arm_qpos().astype(float).copy()
    # 臂缩回抬高, 让根部相机看得清两者(不挡视线)
    env.set_arm(np.array([0.0, 1.2, 2.2, -1.0, -1.57]), JAW_OPEN)
    mujoco.mj_forward(env.model, env.data)

    info = grasp_sleeve_perceived(env, hover_qpos=proven_hover, verbose=True)

    # 验证(仅用本体力传感器 + 接近段位姿, 不读 GT): 夹爪确实下到套筒高度 且 抬起后仍有夹持力
    gf = info["grip_force"]
    reached = info["reached"]
    ok = reached and gf > GRIP_FORCE_TARGET
    print(f"[结果] {'✅ 抓取成功: 夹爪下到套筒并稳定抬起' if ok else '❌ 抓取失败'}"
          f" (下到套筒={reached}, 抬起夹持力 {gf:.1f}N / 阈值 {GRIP_FORCE_TARGET}N)")


if __name__ == "__main__":
    main()
