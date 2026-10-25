#!/usr/bin/env python
"""RL 环境: 套筒插装"推入子阶段"(残差强化学习, M1)。

约束(与全局一致): 观测/奖励只使用真机器可以得到的信号 —— 相机感知、关节编码器、
正运动学(FK); 不使用任何仿真平台内部数据(接触力 / 目标 GT 位姿 / GT 成功标志)。

流程:
  reset(): 随机化(杆倾角 20-50° x 方位 0-360° + 套筒摆放) -> 脚本抓取 -> 脚本对准
           -> (可选)坏起点扰动 -> 返回推入段初始观测
  step(a): a∈[-1,1]^5 归一化残差(杆体系: 轴向/横向x2/轴倾x2)
           -> 基座目标(轴向进给 + 0.3 增益横向 RCC) + 残差修正 + (可选)注入扰动
           -> 一个控制周期(30Hz) -> 奖励/终止/截断/信息

基座与残差的权限划分(嵌套设计, 详见 v1 设计):
  * 基座: 沿杆轴 0.15mm/步 进给 + 横向 -0.3*e 修正(脚本是 1.0, 让出权限给残差,
    否则强基座回环会把残差的横向输出完全抵消);
  * 残差: 轴向速率调制(±2mm/步, 可负=回退) + 横向(±3mm) + 轴倾(±2°)。
  数据采集与在线期用同一基座配置, 避免分布偏移。

用法(冒烟测试):
    MUJOCO_GL=egl python so100_sim/rl_insert_env.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent                      # so100_sim/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception" / "python"))

from so100_sleeve_env import (                               # 仅环境类与常量(非 GT 函数)
    SO100SleeveEnv, ALL_JOINTS, Z_AXIS,
    JAW_INSERT_LIMIT, LATERAL_MAX,
    SLEEVE_LEN, ROD_LEN, PRE_INSERT_GAP,
    INSERT_SUCCESS, INSERT_TARGET, ALIGN_TOL_DEG,
)
from grasp_perceived import (
    align_sleeve_perceived, control_step, goto_p, goto_joint,
)
from cart_assembly import drive_cart_to, drive_cart_holding
from cart_transport import _pickup, ASSEMBLY_DOCK, SLEEVE_RANGE
from perception_api import perceive_scene

# ------------------------------------------------------------------ 常量
OBS_DIM = 31
ACT_DIM = 5
INSERT_SPEED = 0.00015            # 基座轴向进给 m/步(与脚本一致)
LATERAL_GAIN_RL = 0.3             # RL 期基座横向增益(脚本 1.0, 让权给残差)
# 感知杆长(系统性标定值): 实测 |perceived tip-base| ≈ 90mm(真 80mm, C 感知含少量余量)。
# 用 tip + axis + 该长度重建 base, 保证"预插点(由 tip 算)"与"偏心 e(由 base 算)"严格共线,
# 消除感知 base/tip 噪声导致的人为偏心(实测不重建时起始 e 达 14-23mm)。
ROD_PERC_LEN = 0.090
# 残差限幅(物理量, SI): 轴向/横向1/横向2/轴倾1/轴倾2
DELTA_LIMITS = np.array([0.002, 0.003, 0.003,
                         np.radians(2.0), np.radians(2.0)])

# 奖励权重
W_PROGRESS = 40.0     # 重叠增量(mm)。v3 攻卡阻短板: 20->40, 让"持续深推"成为主导奖励信号
                     # (v1/v2 的失败模式是轴向推进不坚决 -> 卡阻; 进度权重翻倍后停滞的机会成本翻倍)
W_CENTER = 0.5        # 横向偏心(mm)
W_ALIGN = 2.0         # sin(轴角)
W_TIME = 0.01
R_SUCCESS = 100.0
R_CROOKED = -50.0
R_TIMEOUT = -20.0
CROOKED_DEG = 15.0    # 轴角超过 -> 歪插终局
CROOKED_E_MM = 10.0   # 偏心超过 -> 脱杆终局
STALL_LIMIT = 40      # 重叠停滞步数 -> 卡阻终局(与脚本一致)
ADV_MAX = 0.12        # 指令推进安全上限(m)


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rod_frame(z):
    """由杆轴构造确定性正交基 (x_r, y_r, z_r)。任务几何全部在杆体系表达, 与杆的世界姿态无关。"""
    z = _unit(z)
    a = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


class InsertPushEnv(gym.Env):
    """套筒插装推入子阶段的 Gym 环境(残差 RL)。

    observation_space: Box(-inf, inf, (31,), float32)
    action_space:      Box(-1, 1, (5,), float32)  —— 归一化残差, 乘 DELTA_LIMITS 得物理量
    """

    metadata = {"render_modes": []}

    def __init__(self, render: bool = True, max_steps: int = 400,
                 p_perturb: float = 0.01, p_bad_start: float = 0.25,
                 tilt_range=(20.0, 40.0), azim_range=(0.0, 360.0),
                 sleeve_xy_range=SLEEVE_RANGE, reset_retries: int = 8,
                 debug: bool = False):
        super().__init__()
        self.debug = bool(debug)
        self.sim = SO100SleeveEnv(width=640, height=480, camera="base_rgbd", render=render)
        self.max_steps = int(max_steps)
        self.p_perturb = float(p_perturb)
        self.p_bad_start = float(p_bad_start)
        self.tilt_range = tuple(tilt_range)
        self.azim_range = tuple(azim_range)
        self.sleeve_xy_range = sleeve_xy_range
        self.reset_retries = int(reset_retries)
        self.rng = np.random.default_rng(0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)

    # ------------------------------------------------------------- 内部工具
    def _overlap(self, center) -> float:
        """感知重叠: 套筒中心相对感知杆的轴向投影与杆/套筒长度求交(纯感知+FK)。"""
        t = float((np.asarray(center, float) - self.rod_base) @ self.z_r)
        return min(ROD_LEN, t + SLEEVE_LEN / 2) - max(0.0, t - SLEEVE_LEN / 2)

    def _ctrl(self, des, d, perturb=None):
        """基座目标 + 残差 + (可选)扰动 -> 执行一个控制周期。des 已由调用方推进。"""
        cur_c = self.sim.sleeve_pose()[0]
        rel = cur_c - self.rod_base
        e = rel - float(rel @ self.z_r) * self.z_r
        base_lat = np.clip(-LATERAL_GAIN_RL * e, -LATERAL_MAX, LATERAL_MAX)
        tgt = des + base_lat - self.z_r * d[0] + self.x_r * d[1] + self.y_r * d[2]
        axis = _unit(self.z_r + self.x_r * d[3] + self.y_r * d[4])
        if perturb is not None:
            kind, vec, ang, _ = perturb
            if kind == "lat":
                tgt = tgt + vec
            else:
                axis = _unit(self.z_r + self.x_r * ang[0] + self.y_r * ang[1])
        control_step(self.sim, tgt, axis, self.sim.jaw_grip, self.cur, self.sol_prev,
                     mode="pose", pre_offset=0.0, clamp_ref=None, slip_rot=False)

    def _maybe_start_perturb(self):
        """按概率调度一个扰动(横向推移或轴倾斜, 持续若干步), 制造需要恢复的坏状态。"""
        if self.perturb is not None or self.p_perturb <= 0.0:
            return
        if self.rng.random() >= self.p_perturb:
            return
        self.episode_perturbed = True
        if self.rng.random() < 0.6:                      # 横向推移
            th = self.rng.uniform(0.0, 2.0 * np.pi)
            mag = self.rng.uniform(0.005, 0.020)
            vec = (np.cos(th) * self.x_r + np.sin(th) * self.y_r) * mag
            ang = np.zeros(2)
        else:                                            # 轴倾斜
            vec = np.zeros(3)
            ang = self.rng.uniform(-np.radians(8.0), np.radians(8.0), size=2)
        dur = int(self.rng.uniform(3, 11))
        self.perturb = ("lat" if np.linalg.norm(vec) > 0 else "axis", vec, ang, dur)

    def _inject_bad_start(self):
        """回合开始前用基座控制执行一个偏移, 造出"未对准"的起始状态(恢复训练)。

        偏移量 3-10mm: 经基座 0.3 增益回拉后稳态偏移 ~0.77x = 2-8mm, 低于歪插终局
        阈值(10mm), 保证回合不会一开始就注定失败。
        """
        if self.rng.random() < 0.6:
            th = self.rng.uniform(0.0, 2.0 * np.pi)
            mag = self.rng.uniform(0.003, 0.010)
            vec = (np.cos(th) * self.x_r + np.sin(th) * self.y_r) * mag
            ang = np.zeros(2)
        else:
            vec = np.zeros(3)
            ang = self.rng.uniform(-np.radians(5.0), np.radians(5.0), size=2)
        dur = int(self.rng.uniform(4, 12))
        perturb = ("lat" if np.linalg.norm(vec) > 0 else "axis", vec, ang, dur)
        self._bad_start_applied = True
        des = self.sim.sleeve_pose()[0].copy()
        for _ in range(dur):
            des = des - self.z_r * INSERT_SPEED
            self._ctrl(des, np.zeros(ACT_DIM), perturb=perturb)

    # ------------------------------------------------------------- 观测
    def _obs(self) -> np.ndarray:
        env = self.sim
        q = env.arm_qpos()                               # (6,) 5 臂 + 爪
        q_cmd = np.array([env.data.ctrl[env.ctrl_idx[n]] for n in ALL_JOINTS[:5]])
        track_err = q_cmd - q[:5]                        # 指令-实际关节跟踪误差(接触的代理信号)
        tool_pos = env.grasp_point()                     # FK
        tool_axis = env.tool_axis()                      # FK
        rel = tool_pos - self.rod_base
        e = rel - float(rel @ self.z_r) * self.z_r
        tool_pos_rf = np.array([rel @ self.x_r, rel @ self.y_r, rel @ self.z_r])
        tool_axis_rf = np.array([tool_axis @ self.x_r, tool_axis @ self.y_r,
                                 tool_axis @ self.z_r])
        e_rf = np.array([e @ self.x_r, e @ self.y_r, e @ self.z_r])
        tilt = np.array([np.arcsin(np.clip(tool_axis_rf[0], -1.0, 1.0)),
                         np.arcsin(np.clip(tool_axis_rf[1], -1.0, 1.0))])
        tip_rel = self.rod_tip - tool_pos
        tip_rel_rf = np.array([tip_rel @ self.x_r, tip_rel @ self.y_r,
                               tip_rel @ self.z_r])
        return np.concatenate([
            q,                                           # 6
            track_err,                                   # 5
            tool_pos_rf,                                 # 3
            tool_axis_rf,                                # 3
            e_rf,                                        # 3
            tilt,                                        # 2
            [self._overlap(tool_pos)],                   # 1
            tip_rel_rf,                                  # 3
            self.last_action,                            # 5
        ]).astype(np.float32)

    # ------------------------------------------------------------- 杆勘测
    def _survey_rod(self, offset=(0.0, 0.0, 0.0)):
        """在指定停靠偏移处感知杆并做健全性校验。

        校验全部基于"任务设计先验 + 感知自洽", 不读 GT:
          * 倾角在任务范围内(杆立于 V 托座, 大致朝上; 实测漏检/误检常表现为 89° 的假杆);
          * |tip-base| 量级合理(感知杆长系统性 ~90mm, 真 80mm, 是标定特性);
          * base->tip 连线与 axis 自洽(<5°)。
        返回 (ok, axis, tip, base)。
        """
        drive_cart_to(self.sim, ASSEMBLY_DOCK[0] + offset[0],
                      ASSEMBLY_DOCK[1] + offset[1], offset[2])
        det = perceive_scene(self.sim, "base_rgbd")
        if "rod" not in det:
            return False, None, None, None
        rod = det["rod"]
        ax = np.asarray(rod["axis"], float)
        if np.linalg.norm(ax) < 1e-6:
            return False, None, None, None
        ax = ax / np.linalg.norm(ax)
        tip = np.asarray(rod["tip"], float)
        base = np.asarray(rod["base"], float)
        tilt = float(np.degrees(np.arccos(np.clip(ax[2], -1.0, 1.0))))
        if not (15.0 <= tilt <= 55.0):
            return False, None, None, None
        seg = float(np.linalg.norm(tip - base))
        if not (0.06 <= seg <= 0.12):
            return False, None, None, None
        seg_dir = (tip - base) / max(seg, 1e-9)
        if float(np.degrees(np.arccos(np.clip(abs(seg_dir @ ax), 0, 1)))) > 5.0:
            return False, None, None, None
        # 用 tip + axis + 感知杆长重建 base(尖和轴都准, 重建后杆完全自洽)
        base = tip - ROD_PERC_LEN * ax
        return True, ax, tip, base

    # ------------------------------------------------------------- Gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # --- 随机化 + 脚本抓取 + 脚本对准(失败重试) ---
        # 复刻批处理流水线前半段(cart_transport.run_once 的 0-3 步):
        #   勘测位测杆(无遮挡) -> 停靠视觉伺服+proven 抓取 -> 搬回装配停靠位 -> 预纠竖直 -> 对准
        # 停靠校正是必须的: 抓取的"居中校正"用固定关节构型(PROVEN_GRASP_SEED), 只有套筒位于
        # 臂的舒适相对位姿(PICKUP_REL_XY)时才对准; 不校正会空夹(实测 Δjaw=0)。
        align_deg = float("nan")
        for _ in range(self.reset_retries):
            self.sim.reset(tilt_deg=None, azim_deg=None, rng=self.rng,
                           sleeve_xy_range=self.sleeve_xy_range)
            # 0) 勘测位测杆: 抓持后臂+套筒会挡看杆视线, 故在无遮挡时先测准。
            #    双视角交叉校验: 换个停靠位再测一次, 两次必须一致(杆世界固定);
            #    防"漏检/误检"(实测随机位姿下 ~25% 勘测异常, 表现为 89° 假杆或漏检)。
            ok, ax, tip, base = self._survey_rod()
            if not ok:
                if self.debug:
                    print("[env-reset] 勘测A失败")
                continue
            ok2, ax2, tip2, _ = self._survey_rod((0.05, 0.05, 0.0))
            if not ok2:
                if self.debug:
                    print("[env-reset] 勘测B失败")
                continue
            if (float(np.degrees(np.arccos(np.clip(abs(ax @ ax2), 0, 1)))) > 4.0
                    or float(np.linalg.norm(tip - tip2)) > 0.04):
                if self.debug:
                    print("[env-reset] 双视角不一致")
                continue
            self.rod_axis, self.rod_tip, self.rod_base = ax, tip, base
            self.x_r, self.y_r, self.z_r = _rod_frame(self.rod_axis)
            # 1) 视觉伺服停靠 + proven 抓取抬起
            ok, _ginfo = _pickup(self.sim, verbose=False)
            if not ok:
                if self.debug:
                    print("[env-reset] 抓取失败")
                continue
            # 2) 夹持套筒搬回装配停靠位
            drive_cart_holding(self.sim, *ASSEMBLY_DOCK)
            # 3) 预纠竖直(proven 插入假设起始轴 = Z)
            cur = self.sim.arm_qpos().astype(float)
            sol_prev = cur[:5].copy()
            sc = self.sim.grasp_point() + self.sim.jaw_rot() @ self.sim.grasp_off_local
            sa = self.sim.jaw_rot() @ self.sim.grasp_axis_local
            ang = float(np.degrees(np.arccos(np.clip(abs(sa @ Z_AXIS), 0, 1))))
            if ang > 3.0:
                goto_p(self.sim, sc, sa, sc, Z_AXIS, 0.8, self.sim.jaw_grip,
                       cur, sol_prev, mode="pose", pre_offset=0.0)
            # 4) 对准 + 起始状态校验。标准 align(35°/0° 验证过)对随机杆位姿常落坏局部极小
            #    (轴对上但位置差几十 mm); 不合格时用 ik_restart(多起点+对齐优先+分支连续)补救一次。
            cur, sol_prev, align_deg = align_sleeve_perceived(
                self.sim, self.rod_axis, self.rod_tip, self.rod_base, verbose=False)
            ok_start = False
            for fix_try in range(2):
                pre = self.rod_tip + self.rod_axis * (SLEEVE_LEN / 2 + PRE_INSERT_GAP)
                sc_now = self.sim.grasp_point() + self.sim.jaw_rot() @ self.sim.grasp_off_local
                rel0 = sc_now - self.rod_base
                e0 = rel0 - float(rel0 @ self.rod_axis) * self.rod_axis
                ok_start = (align_deg <= 12.0
                            and float(np.linalg.norm(e0)) <= 0.008
                            and float(np.linalg.norm(sc_now - pre)) <= 0.025)
                if ok_start or fix_try == 1:
                    break
                if self.debug:
                    print(f"[env-reset] 起始不合格(align={align_deg:.1f}° "
                          f"e0={np.linalg.norm(e0)*1000:.1f}mm "
                          f"残差={np.linalg.norm(sc_now - pre)*1000:.1f}mm) -> 重启补救")
                R = self.sim.jaw_rot()
                q_try = self.sim.ik_restart(pre - R @ self.sim.grasp_off_local,
                                            self.rod_axis, q_seed=sol_prev,
                                            axis_local=self.sim.grasp_axis_local,
                                            mode="pose", n_rand=24, max_step=0.05,
                                            n_iter=300, clamp_ref=None, rng=self.rng)
                if float(np.linalg.norm(q_try - sol_prev)) > 2.5:
                    break                          # 太远分支, 不稳, 放弃补救
                goto_joint(self.sim, q_try, 2.5, self.sim.jaw_grip, cur, sol_prev)
                _, sa = self.sim.sleeve_pose()
                align_deg = float(np.degrees(np.arccos(np.clip(abs(sa @ self.rod_axis), 0, 1))))
            if not ok_start:
                if self.debug:
                    print(f"[env-reset] 补救后仍不合格: align={align_deg:.1f}°")
                continue
            break
        else:
            raise RuntimeError(f"reset 重试 {self.reset_retries} 次仍失败(抓取或对准)")

        self.cur = cur.copy()
        self.sol_prev = sol_prev.copy()
        self.sim.set_jaw_force_limit(JAW_INSERT_LIMIT)    # 柔顺夹持: 套筒可在爪内轻微自定心

        # --- (可选)坏起点扰动 ---
        self.episode_perturbed = False
        self._bad_start_applied = False
        if self.rng.random() < self.p_bad_start:
            self._inject_bad_start()

        # --- 回合状态 ---
        self.des = self.sim.sleeve_pose()[0].copy()
        self.des_start = self.des.copy()
        self.steps = 0
        self.stalled = 0
        self.best_ov = -1.0
        self.prev_ov = self._overlap(self.des)
        self.last_action = np.zeros(ACT_DIM)
        self.perturb = None
        self.episode_perturbed = bool(self.episode_perturbed
                                      or self._bad_start_applied)   # 含坏起点(也是受扰)
        info = {
            "align_deg_at_start": float(align_deg),
            "rod_tilt_deg": float(np.degrees(np.arccos(np.clip(self.rod_axis[2], -1, 1)))),
            "overlap_mm_at_start": self.prev_ov * 1000.0,
        }
        return self._obs(), info

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        d = a * DELTA_LIMITS

        # --- 基座推进 + 残差 + 扰动 ---
        self.des = self.des - self.z_r * INSERT_SPEED
        self._ctrl(self.des, d, perturb=self.perturb)
        if self.perturb is not None:
            kind, vec, ang, t_remain = self.perturb
            self.perturb = (kind, vec, ang, t_remain - 1) if t_remain - 1 > 0 else None

        # --- 度量(全部感知+FK, 无 GT/平台数据) ---
        cur_c, cur_a = self.sim.sleeve_pose()
        rel = cur_c - self.rod_base
        e = rel - float(rel @ self.z_r) * self.z_r
        e_mm = float(np.linalg.norm(e)) * 1000.0
        ov = self._overlap(cur_c)
        align_rad = float(np.arccos(np.clip(abs(cur_a @ self.z_r), 0.0, 1.0)))
        align_deg = float(np.degrees(align_rad))

        # --- 奖励 ---
        d_ov_mm = (ov - self.prev_ov) * 1000.0
        r = (W_PROGRESS * d_ov_mm - W_CENTER * e_mm
             - W_ALIGN * np.sin(align_rad) - W_TIME)

        # --- 终止判定 ---
        terminated, success, crooked, jammed = False, False, False, False
        if ov >= INSERT_SUCCESS and align_deg <= ALIGN_TOL_DEG:
            terminated, success = True, True
            r += R_SUCCESS
        elif align_deg > CROOKED_DEG or e_mm > CROOKED_E_MM:
            terminated, crooked = True, True
            r += R_CROOKED
        else:
            if ov > self.best_ov + 5e-4:
                self.best_ov = ov
                self.stalled = 0
            else:
                self.stalled += 1
                if self.stalled > STALL_LIMIT:
                    terminated, jammed = True, True
            if not terminated and float((self.des_start - self.des) @ self.z_r) > ADV_MAX:
                terminated = True                          # 过推保护

        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            r += R_TIMEOUT
        self.prev_ov = ov
        self.last_action = a
        self._maybe_start_perturb()

        info = {
            "overlap_mm": ov * 1000.0,
            "align_deg": align_deg,
            "e_mm": e_mm,
            "success": success,
            "crooked": crooked,
            "jammed": jammed,
            "steps": self.steps,
            "perturbed": bool(self.episode_perturbed),
            "rod_tilt_deg": float(np.degrees(np.arccos(np.clip(self.rod_axis[2], -1, 1)))),
        }
        return self._obs(), float(r), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        self.sim.close()


# ------------------------------------------------------------------ 冒烟测试
if __name__ == "__main__":
    """零残差(纯基座)跑若干回合, 验证环境端到端可用并给出基座基线。"""
    import time

    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    env = InsertPushEnv(render=True, p_perturb=0.0, p_bad_start=0.0,
                        debug=("--debug" in sys.argv))
    t0 = time.time()
    stats = {"success": 0, "crooked": 0, "jammed": 0, "timeout": 0}
    for ep in range(n_ep):
        obs, info = env.reset(seed=1000 + ep)
        terminated = truncated = False
        total_r = 0.0
        while not (terminated or truncated):
            obs, r, terminated, truncated, step_info = env.step(np.zeros(ACT_DIM))
            total_r += r
        tag = ("success" if step_info["success"] else
               "crooked" if step_info["crooked"] else
               "jammed" if step_info["jammed"] else "timeout")
        stats[tag] += 1
        print(f"[ep{ep}] 杆倾角={info['rod_tilt_deg']:.0f}° 起始对准={info['align_deg_at_start']:.1f}° "
              f"-> {tag}: 重叠={step_info['overlap_mm']:.1f}mm 轴={step_info['align_deg']:.1f}° "
              f"e={step_info['e_mm']:.1f}mm 步数={step_info['steps']} 总奖励={total_r:.1f}")
    dt = time.time() - t0
    print(f"\n基座(零残差)基线: {stats}  {n_ep}回合/{dt:.1f}s = {dt/max(n_ep,1):.2f}s每回合")
    env.close()
