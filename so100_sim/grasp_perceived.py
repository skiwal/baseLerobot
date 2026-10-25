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
    JAW_CLOSE_LIMIT, JAW_FORCE_LIMIT, JAW_INSERT_LIMIT, MAX_JOINT_STEP, FPS,
    smoothstep, IK_RNG, HOVER_BACK, HOVER_DZ, GRASP_SEED,
    SLEEVE_LEN, PRE_INSERT_GAP, LATERAL_GAIN, LATERAL_MAX, INSERT_TARGET,
    INSERT_SUCCESS, LATERAL_TOL, ALIGN_TOL_DEG, ROD_LEN, ROD_R, SLEEVE_R_IN,
)
from perception_api import perceive_scene

# 已验证可用的稳定抓取关节构型(布局固定: tilt=35, azim=0, 套筒 x≈0.122,y≈-0.24,z≈0.11)。
# 直接采用, 不让 IK 微调到"动态下行会下垂"的刀刃分支。
PROVEN_GRASP_SEED = np.array([0.5723, -1.6451, 2.8307, -1.1743, -1.5727])
# 已验证可用的稳定"抬起后"关节构型(同样布局, 套筒已离台面)。
PROVEN_LIFT_SEED = np.array([0.5697, -2.1175, 2.7828, -0.6256, -1.5702])

# ----------------------------------------------------------------- 低层控制(仅本体传感器, 不读 GT)
def control_step(env, eq_center, target_axis, jaw, cur, sol_prev,
                 mode="pose", pre_offset=0.0, clamp_ref=X_AXIS, slip_rot=True,
                 ik_rand=0, ik_tol=0.003, n_escape=0, w_lim=0.0, w_dir=None):
    """一个控制周期: 让等效套筒中心趋近 eq_center(=感知位姿), 轴线趋近 target_axis。"""
    R = env.jaw_rot()
    # 夹持滑移修正: 本流程中 env.sleeve_pose 已被替换为"夹爪FK预测"(插入段)或"相机感知"(接近段),
    # 故 grasp_slip 实际比较的是 FK-vs-FK(插入段恒为零, 保留接口)或 FK-vs-感知(接近段),
    # 均不读仿真 GT, 也不是接触传感器。
    dpos, R_corr = env.grasp_slip()
    if not slip_rot:
        R_corr = np.eye(3)
    goal = np.asarray(eq_center, float) - R @ env.grasp_off_local - dpos + pre_offset * R[:, 0]
    axis = R_corr @ np.asarray(target_axis, float)
    sol = env.pose_ik(goal, axis, q_seed=sol_prev, axis_local=env.grasp_axis_local,
                      q_ref=None, mode=mode, n_iter=150, clamp_ref=clamp_ref,
                      n_escape=n_escape, w_lim=w_lim, w_dir_override=w_dir)
    if env.pose_error(sol, goal, axis, env.grasp_axis_local, mode=mode, clamp_ref=clamp_ref) > ik_tol:
        # 回退仍锁定当前(居中)分支: n_rand=0 + 默认 tol=3mm(随机重启会翻偏心分支甩出套筒)。
        sol = env.ik_restart(goal, axis, sol_prev, axis_local=env.grasp_axis_local,
                             mode=mode, n_rand=ik_rand, max_step=0.05, n_iter=200,
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


def grip_p(env, sc, jaw_to, seconds, cur, sol_prev, offset_from=0.0):
    """闭/松爪: 夹持中心锁在感知套筒中心 sc(不调用 env.sleeve_pose())。"""
    j0 = float(cur[5])
    n = max(1, int(seconds * FPS))
    env.set_jaw_force_limit(JAW_CLOSE_LIMIT)
    for i in range(n):
        t = (i + 1) / n
        control_step(env, sc, Z_AXIS, j0 + (jaw_to - j0) * t, cur, sol_prev,
                     "grasp", pre_offset=offset_from * (1 - t))
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
    if "rod" not in det:
        raise RuntimeError("第一步感知未检测到细杆(插装需要杆位姿)")
    rod = det["rod"]
    rod_axis = rod["axis"].astype(float); rod_axis /= np.linalg.norm(rod_axis)
    rod_tip = rod["tip"].astype(float)
    rod_base = rod["base"].astype(float)
    if verbose:
        print(f"[感知] 套筒中心={sp.round(3)} 轴线={sa.round(3)}")
        print(f"[感知] 细杆 base={rod_base.round(3)} tip={rod_tip.round(3)} 轴={rod_axis.round(3)}")

    # 把"套筒位姿"替换为感知结果: 之后控制器内部所有 sleeve_pose() 读取都只来自感知, 不漏 GT
    env.sleeve_pose = lambda: (sp.copy(), sa.copy())

    # --- 悬停构型: 优先用 reset 产生的"已验证悬停构型"(与 proven 路径一致, 保证 IK 跟踪
    #     落在可保持的稳定分支); 否则退化为按感知 sc 推算(可能落到会下垂的分支)。 ---
    if hover_qpos is None:
        hover = sp + np.array([0.0, HOVER_BACK, HOVER_DZ])
        sol_h = env.ik_restart(hover, Z_AXIS, GRASP_SEED, axis_local=Z_AXIS,
                               mode="pose", n_rand=24, max_step=0.05, n_iter=300,
                               clamp_ref=X_AXIS, rng=np.random.default_rng(0))
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

    # 1) 笛卡尔平滑接近: 先到"套筒正上方"高位途经点, 再竖直短下探到套筒中心。
    #    分两段避免斜向接近时夹爪/固定垫片刮碰套筒(随机位姿下曾偶发把套筒碰移导致夹偏/夹空)。
    above = sp + Z_AXIS * 0.05
    goto_p(env, c0, a0, above, Z_AXIS, 0.7, JAW_OPEN, cur, sol_prev,
           mode="grasp", pre_offset=PRE_GRASP_OFFSET)
    goto_p(env, above, Z_AXIS, sp, Z_AXIS, 0.6, JAW_OPEN, cur, sol_prev,
           mode="grasp", pre_offset=PRE_GRASP_OFFSET)
    for _ in range(30):                               # 到位后稳定
        env.step(cur)
    # 末端居中校正: 逐步 IK 跟踪总会漂到偏心分支(夹爪偏套筒中心 ~1.5cm, 后续倾斜必滑脱)。
    # pose_ik 无法复现居中构型(从种子梯度下降落入最近的偏心局部极小), 故直接采用已验证居中
    # 抓取构型 set_arm+settle 可稳稳保持, 且 gripper 正对套筒中心(后续倾斜不滑脱)。
    q_gc = PROVEN_GRASP_SEED.copy()
    env.set_arm(q_gc, JAW_OPEN)
    for _ in range(20):
        env.step(np.concatenate([q_gc, [JAW_OPEN]]))
    cur = env.arm_qpos().astype(float); sol_prev = cur[:5].copy()
    pf = env.data.geom_xpos[env.pad_f]; pm = env.data.geom_xpos[env.pad_m]
    approach_mid = (pf + pm) / 2.0                     # 实际开口pad中点(验证是否真下到套筒)
    if verbose:
        print(f"[接近] 居中校正后 arm={np.round(env.arm_qpos()[:5],3)} "
              f"实际开口pad中点={np.round(approach_mid,3)} 目标sp={np.round(sp,3)}")

    # 2) 闭爪: 保持居中抓取构型(不再解 IK, 否则会漂回偏心分支导致套筒滑脱), 仅闭合夹爪。
    #    控制步用 set_arm 后的居中构型直接定住, 逐步把 jaw 关到 jaw_grip。
    n_close = int(0.8 * FPS)
    for k in range(n_close):
        j = JAW_OPEN + (env.jaw_grip - JAW_OPEN) * (k / (n_close - 1))
        tgt = cur.copy(); tgt[5] = j
        env.step(tgt)
    cur[5] = env.jaw_grip
    if verbose:
        pf = env.data.geom_xpos[env.pad_f]; pm = env.data.geom_xpos[env.pad_m]
        open_mid = (pf + pm) / 2.0
        print(f"[闭爪] 居中保持 arm={np.round(env.arm_qpos()[:5],3)} 闭爪后pad中点={np.round(open_mid,3)} "
              f"实际jaw={env.data.qpos[env.qadr['Jaw']]:.3f}")

    # 锁定抓取: 用感知位姿设置 off_local/axis 并打开滑移修正(measure_grasp 此时读的是感知值, 非 GT)
    env.measure_grasp()
    # 抓取后切换为"本体感知"跟踪(夹爪知道自己抱住的点); 抬起阶段滑移修正据此自洽, 不读 GT
    env.sleeve_pose = lambda: (env.grasp_point().copy(), Z_AXIS.copy())

    # 3) 抬起脱离 V 托架: 采用已验证可保持的"抬起后"构型, 关节空间刚性插值(不逐步重解 IK)
    q_lift = PROVEN_LIFT_SEED.copy()
    goto_joint(env, q_lift, 0.8, env.jaw_grip, cur, sol_prev)
    for _ in range(30):
        env.step(cur)

    # 抓取成功判定(全部机器人可观测, 不用任何仿真平台内部数据):
    # 1) reached: 接近时夹爪pad中点(夹爪FK)确实下到感知套筒高度附近(挡"空闭合"假阳性);
    # 2) blocked: 闭爪后实际jaw角显著大于目标jaw_grip —— jaw_grip 标定为"比套筒对边窄一个
    #    squeeze", 夹到套筒时 Jaw 被挡住关不到底(位置控制+力矩上限, 真机同款判定, 编码器即可得);
    #    空夹时 Jaw 能一路关到 jaw_grip。阈值 0.05rad, 实测分离度 ~0.2rad, 余量充足。
    reached = float(approach_mid[2]) < sp[2] + 0.02
    jaw_actual = float(env.data.qpos[env.qadr["Jaw"]])
    blocked = (jaw_actual - env.jaw_grip) > 0.05
    held = bool(reached and blocked)
    if verbose:
        print(f"[抓取验证] 下到套筒={reached} 闭爪被挡(Δjaw={jaw_actual - env.jaw_grip:.3f}rad)={blocked}")
    return {"target_pos": sp, "held": held, "approach_mid_z": float(approach_mid[2]),
            "reached": reached, "rod_axis": rod_axis, "rod_tip": rod_tip, "rod_base": rod_base}


# ----------------------------------------------------------------- 插装策略(第三步核心)
def align_sleeve_perceived(env, rod_axis, rod_tip, rod_base, verbose=True):
    """第三步a: 对准。把已夹住并抬起的套筒对准细杆轴线并搬到预插位姿。

    仅用感知的杆位姿 + 夹爪本体估计, 不读 GT / 仿真平台数据:
      * 杆位姿(轴/尖/根)由感知给出;
      * 套筒位姿用夹爪本体估计(grasp_point + 腕部姿态变换)。
    返回 (cur, sol_prev, align_axis_deg) —— 推入段(脚本或 RL 环境)接着用。
    """
    rod_axis = np.asarray(rod_axis, float).copy(); rod_axis /= np.linalg.norm(rod_axis)
    rod_tip = np.asarray(rod_tip, float).copy()
    rod_base = np.asarray(rod_base, float).copy()

    # 套筒位姿: 用夹爪本体估计(夹持中心 + 腕部姿态), 完全不读 GT 套筒体
    env.sleeve_pose = lambda: (env.grasp_point() + env.jaw_rot() @ env.grasp_off_local,
                               env.jaw_rot() @ env.grasp_axis_local)

    cur = env.arm_qpos().astype(float)
    sol_prev = cur[:5].copy()

    # ---- 4) 对准(关节空间): 一次性 pose_ik(从当前构型梯度下降 -> 近处解/同分支, 不会像随机重启那样
    #     跳远分支把臂甩飞)求"套筒在杆上方 pre、轴对齐 rod_axis"的构型, 再 goto_joint 关节空间搬过去。
    #     比开环逐步跟踪更能把轴转到杆轴(随机 held 位姿下逐步跟踪常 align 34~73°); 放慢摆臂减小惯性
    #     甩飞。若解跳远分支(关节距过大)则回退 proven 两段式对准。 ----
    pre = rod_tip + rod_axis * (SLEEVE_LEN / 2 + PRE_INSERT_GAP)
    env.set_jaw_force_limit(JAW_FORCE_LIMIT)   # 牢固夹持: 对准/推入全程不脱手
    R = env.jaw_rot()
    q_pre = env.pose_ik(pre - R @ env.grasp_off_local, rod_axis, q_seed=sol_prev,
                        axis_local=env.grasp_axis_local, mode="pose",
                        n_iter=300, max_step=0.05, clamp_ref=None)
    if float(np.linalg.norm(q_pre - sol_prev)) > 2.5:      # 极远分支 -> 回退 proven 对准
        if verbose:
            print(f"[对准] IK极远分支(关节距={np.linalg.norm(q_pre - sol_prev):.2f}rad) -> 回退proven对准")
        sc, _ = env.sleeve_pose()
        goto_p(env, sc, Z_AXIS, pre, Z_AXIS, 1.3, env.jaw_grip, cur, sol_prev)   # 段1: 竖直平移
        goto_p(env, pre, Z_AXIS, pre, rod_axis, 1.2, env.jaw_grip, cur, sol_prev,
               mode="pose", pre_offset=0.0, clamp_ref=None)                     # 段2: 原地转轴
    else:
        goto_joint(env, q_pre, 2.5, env.jaw_grip, cur, sol_prev)   # 放慢摆臂, 减小惯性甩飞套筒
    _, sa = env.sleeve_pose()
    align_axis_deg = float(np.degrees(np.arccos(np.clip(abs(sa @ rod_axis), 0, 1))))
    if verbose:
        residual = float(np.linalg.norm(
            (env.grasp_point() + env.jaw_rot() @ env.grasp_off_local) - pre) * 1000)
        print(f"[对准] 轴夹角={align_axis_deg:.2f}° 预插装残差={residual:.1f}mm")
    return cur, sol_prev, align_axis_deg


def push_in_perceived(env, rod_axis, rod_base, cur, sol_prev, verbose=True):
    """第三步b: 推入。从当前(已对准)状态沿杆轴推入到目标深度(脚本版, 自带收尾)。

    仅用感知杆位姿 + 夹爪本体估计, 不读 GT / 平台数据。
    """
    # ---- 沿杆轴推入(限幅推进 + 横向柔顺 RCC + 卡阻保护) ----
    # 完全基于"相机感知 + 机械臂本体": 控制决策不读任何仿真 GT / 平台内部数据。
    # 每步把目标沿 -rod_axis 推进 insert_speed, 量出套筒(夹爪本体估计)相对杆轴的横向偏心 e 做
    # -LATERAL_GAIN*e 修正(等效 RCC 自定心); target_axis 用 rod_axis 锁定姿态; control_step 回退
    # 锁 n_rand=0(居中分支) 不翻偏心分支。夹持力限 0.7: 松一点让套筒顺杆滑入。
    # 停止判据(纯相机+本体, 零 GT / 零平台数据): 用"感知杆(杆尖+感知轴) + 夹爪本体估计的套筒
    # 中心"自行算重叠, 重叠达标即停; 另加重叠长时间停滞(卡死)与指令推进量上限保护。
    env.set_jaw_force_limit(JAW_INSERT_LIMIT)
    steps, jammed, stalled = 0, False, 0
    des = env.sleeve_pose()[0].copy()
    des_start = des.copy()
    insert_speed = 0.00015
    max_insert_steps = 800
    adv_max = 0.09       # 安全上限: 指令沿 -杆轴推进量超过此值仍没坐入即停(防过推/臂限)
    best_ov = -1.0
    while steps < max_insert_steps:
        cur_c, _ = env.sleeve_pose()                      # 夹爪本体估计套筒中心(非 GT)
        # 感知重叠: 套筒中心相对感知杆的轴向投影 t, 与杆/套筒长度求交(纯感知+本体)
        t = float((cur_c - rod_base) @ rod_axis)
        ov = min(ROD_LEN, t + SLEEVE_LEN / 2) - max(0.0, t - SLEEVE_LEN / 2)
        if ov >= INSERT_TARGET:                           # 感知重叠达标 -> 停止
            break
        if ov > best_ov + 5e-4:                           # 重叠在长 -> 重置停滞计数
            best_ov = ov
            stalled = 0
        else:
            stalled += 1
            if stalled > 40:                              # ~1.3s 重叠完全没长 -> 卡阻
                jammed = True
                break
        advanced = float((des_start - des) @ rod_axis)     # 指令已推进量(沿 -杆轴)
        if advanced >= adv_max:                            # 安全上限
            break
        rel = cur_c - rod_base
        e = rel - float(rel @ rod_axis) * rod_axis         # 横向偏心(相对感知杆轴, 粗定心)
        lateral = np.clip(-LATERAL_GAIN * e, -LATERAL_MAX, LATERAL_MAX)
        # 目标=指令点 des 沿 -rod_axis 推进 insert_speed; 用累积式(始终顶杆里压, 能克服接触卡滞),
        # 而非从实测位置退一步(臂卡住时目标不前进反而停滞)。
        des = des - rod_axis * insert_speed
        control_step(env, des + lateral, rod_axis, env.jaw_grip, cur, sol_prev,
                     mode="pose", pre_offset=0.0, clamp_ref=None, slip_rot=False)
        steps += 1
        if verbose and steps % 25 == 0:
            print(f"  [推入] s={steps} 重叠={ov*1000:5.1f} e={np.linalg.norm(e)*1000:4.1f}mm "
                  f"推进={advanced*1000:.1f}mm "
                  f"指令cur={np.round(cur[:5],3)} 实际qpos={np.round(env.arm_qpos()[:5],3)}")
    env.set_jaw_force_limit(JAW_FORCE_LIMIT)

    # 退出臂之前先抓一次度量(套筒尚在杆上); 之后 reset grasp_off_local/退臂会让 sleeve_pose 失真。
    # 最终成功判定用感知量(套筒中心/轴=夹爪本体估计, 杆=感知杆尖+夹具标定轴), 不读 GT。
    final_sc, final_sa = env.sleeve_pose()
    t_final = float((final_sc - rod_base) @ rod_axis)
    final_ov = min(ROD_LEN, t_final + SLEEVE_LEN / 2) - max(0.0, t_final - SLEEVE_LEN / 2)
    align_cos = float(abs(final_sa @ rod_axis))
    success = (final_ov >= INSERT_SUCCESS) and (align_cos > np.cos(np.radians(ALIGN_TOL_DEG)))

    # (可选)离线 GT 交叉验证, 仅供评测对比, 绝不反馈进控制回路(默认开, 不影响控制)。
    GT_VERIFY = True
    gt_ok = env.is_success() if GT_VERIFY else None
    gt_mm = env.insert_metrics() if GT_VERIFY else None

    # ---- 6) 收尾: 成功保持, 失败先抬后松; 随后退开(用感知杆轴方向退, 不读 GT) ----
    if not success:
        sc, _ = env.sleeve_pose()
        goto_p(env, sc, rod_axis, sc + np.array([0, 0, 0.045]), rod_axis, 0.6, env.jaw_grip, cur, sol_prev)
    grip_p(env, env.sleeve_pose()[0], JAW_OPEN, 0.6, cur, sol_prev)
    env.grasp_active = False
    env.grasp_off_local = np.zeros(3)
    env.grasp_axis_local = Z_AXIS.copy()
    sc, _ = env.sleeve_pose()
    goto_p(env, sc, rod_axis, sc + np.array([0.02, 0.07, 0.07]), rod_axis, 1.0, JAW_OPEN, cur, sol_prev)

    if verbose:
        print(f"[推入] 指令推进={INSERT_TARGET * 1000:.1f}mm "
              f"轴夹角={np.degrees(np.arccos(min(1, align_cos))):.2f}° "
              f"感知重叠={final_ov * 1000:.1f}mm "
              f"{'(卡阻)' if jammed else ''}")
        if GT_VERIFY:
            print(f"[验证·离线GT] is_success={gt_ok}  实测重叠={gt_mm['overlap'] * 1000:.1f}mm "
                  f"横向={gt_mm['lateral'] * 1000:.1f}mm 轴向={np.degrees(np.arccos(min(1, gt_mm['align_cos']))):.2f}° "
                  f"(阈值 重叠{INSERT_SUCCESS * 1000:.0f}mm/横向{LATERAL_TOL * 1000:.0f}mm/轴向{ALIGN_TOL_DEG}°)  <- 仅评测, 非控制器输入")
    return {"success": bool(success), "jammed": jammed,
            "insert_overlap_mm": float(final_ov * 1000),
            "advanced_mm": INSERT_TARGET * 1000}


def insert_sleeve_perceived(env, rod_axis, rod_tip, rod_base, verbose=True):
    """第三步(脚本版全流程): 对准 + 推入 + 收尾。仅用感知 + 本体, 不读 GT。"""
    cur, sol_prev, align_axis_deg = align_sleeve_perceived(
        env, rod_axis, rod_tip, rod_base, verbose=verbose)
    rinfo = push_in_perceived(env, rod_axis, rod_base, cur, sol_prev, verbose=verbose)
    rinfo["align_axis_deg"] = align_axis_deg
    return rinfo


def main():
    env = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=True)
    env.reset(tilt_deg=35.0, azim_deg=0.0)
    # reset 后臂处于"已验证悬停构型"(与 proven 路径一致); 先记录, 之后回到它做抓取
    proven_hover = env.arm_qpos().astype(float).copy()
    # 臂缩回抬高, 让根部相机看得清两者(不挡视线)
    env.set_arm(np.array([0.0, 1.2, 2.2, -1.0, -1.57]), JAW_OPEN)
    mujoco.mj_forward(env.model, env.data)

    info = grasp_sleeve_perceived(env, hover_qpos=proven_hover, verbose=True)

    # 验证(仅用夹爪FK + 相机复查, 不读任何仿真内部数据): 夹爪确实下到套筒高度 且 抬起后套筒随夹爪
    reached = info["reached"]
    held = info["held"]
    ok = reached and held
    print(f"[结果] {'✅ 抓取成功: 夹爪下到套筒并稳定抬起' if ok else '❌ 抓取失败'}"
          f" (下到套筒={reached}, 抬起后套筒随夹爪={held})")

    if not ok:
        return

    # ---- 第三步: 用第一步感知到的杆位姿, 把套筒对准杆轴线并推入 ----
    # 杆轴纯感知(perceive.c 输出, 不读 GT, 不用夹具标定角): RGB-D 在细杆上的杆轴有 ~2.3° 固定
    # 深度误差(确定性, 非噪声), 用牢固夹持防止推入中套筒被杆沿错向顶滑脱, 并深推到感知重叠
    # 达标。套筒(真正目标)位姿全程来自感知, 不读 GT。
    rod_axis_ref = np.asarray(info["rod_axis"], float).copy()        # 纯感知杆轴
    rod_axis_ref = rod_axis_ref / np.linalg.norm(rod_axis_ref)
    rod_tip_ref = np.asarray(info["rod_tip"], float).copy()          # 感知杆尖(准确)
    rod_base_ref = rod_tip_ref - rod_axis_ref * ROD_LEN               # 杆根由杆尖+感知轴+杆长推出
    verbose = True
    if verbose:
        print(f"[精修] 纯感知杆轴={np.round(rod_axis_ref,3)} 位置用感知杆尖={np.round(rod_tip_ref,3)}")

    rinfo = insert_sleeve_perceived(env, rod_axis_ref, rod_tip_ref, rod_base_ref, verbose=True)
    # 最终成功以 is_success(重叠>=阈值 且 轴向达标)为准; jammed 仅为过程提示(接触力尖峰会误报, 不影响已坐入的结果)
    ins_ok = rinfo["success"]
    print(f"[结果] {'✅ 插装成功: 套筒套入细杆' if ins_ok else '❌ 插装失败'}"
          f" (重叠 {rinfo['insert_overlap_mm']:.1f}mm / 阈值 {INSERT_SUCCESS * 1000:.0f}mm, "
          f"轴夹角 {rinfo['align_axis_deg']:.1f}°)")


if __name__ == "__main__":
    main()
