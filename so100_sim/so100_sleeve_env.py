#!/usr/bin/env python
"""SO-100 套筒-细杆插装任务环境 (MuJoCo)。

任务
    台面上立着一根**任意倾角**的细杆(半径 5mm), 旁边立着一个**空心套筒**
    (外径 30mm / 内孔 16mm / 长 60mm)。机械臂要把套筒抓起来, 把内孔对准细杆轴线,
    再把套筒推到杆外(杆插入内孔 >= 30mm 记为成功)。

抓取方式 (为什么这样设计)
    夹爪指尖方向 = 手部 y_h 轴, 夹紧方向 = x_h 轴, 第三轴 z_h = 垂直于"指面"。
    套筒**立着**放, 夹爪**从侧面水平**伸过去夹套筒外圆柱面, 于是套筒轴线 == z_h。
    这样做的两个好处:
      1) 抓取阶段手指不用伸到台面以下, 不会撞台面;
      2) 对准阶段只需要转动腕部让 z_h 与细杆轴线平行, 手指始终与杆轴垂直,
         推入时手指在杆的两侧 15mm 外, 不会碰到杆。

接口 (与 so100_pick_env.py 保持同构)
    关节 6 维: [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw] (rad)
    观测: joint_positions (6,) float32 + image (H,W,3) uint8
    动作: (6,) 关节目标角 (rad), 直接写给 MuJoCo 位置执行器

位姿 IK
    位置 3 维 + 套筒轴线方向 2 维 = 5 个约束, 正好用 5 个臂关节, 用阻尼最小二乘解:
        J_pos  : 抓取点(pad4 中点) 的平动雅可比
        J_dir  : 轴向误差 a_target - a 的雅可比 = -skew(a) @ J_rot[:, 臂自由度]
    由于套筒是回转体, 绕自身轴线的滚转不需要约束, 5 自由度刚好够。

运行 (自测: 随机倾角跑若干回合, 录一段视频)
    MUJOCO_GL=egl python so100_sleeve_env.py --episodes 6
    MUJOCO_GL=egl python so100_sleeve_env.py --tilt 25 --azim 150 --episodes 3

已知限制 (实测, 待解决)
    1) 抓取链路是稳的: 接近误差 ~4mm、夹持偏移 <2mm、夹持力 ~17N、不会滑脱。
    2) 推入阶段目前成功率是 0: 手臂沿杆轴推套筒时, 肩/肘力矩会在 ±3.5Nm 打满,
       同时夹爪的指侧网格会压到套筒(接触力 ~80N), 形成自锁 —— 位置控制在推不动时
       会把力一路顶到执行器上限。控制器里已加"卡阻检测"(超前 2mm 持续 0.7s 就停),
       不会再硬顶, 但插装本身还没成功。
       根因: 这个 5 自由度臂在 z≈0.11 / y≈-0.19 附近已接近伸展极限, 沿 -y 方向再推
       30~40mm 需要的力矩超出执行器预算; 而且"手在套筒正上方、手指朝下"的姿态下,
       手自身的指侧几何离套筒表面只有亚毫米级余量。
       可行的后续方向: (a) 把杆往机器人方向再移 1~2cm(缩短行程); (b) 给指侧网格加
       contype/conaffinity 让指侧不参与接触(只用 pad 垫片夹持); (c) 换成"套筒先被
       托架托着、杆沿轴线插进来"的主动对准方式, 或给臂加力控/导纳控制。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from video_utils import write_video_h264

SCENE_PATH = Path(__file__).resolve().parent / "task" / "so100_sleeve_scene.xml"

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
ALL_JOINTS = ARM_JOINTS + ["Jaw"]

# ---------------------------------------------------------------- 场景常量 (与 XML 一致)
TABLE_TOP = 0.04  # 台面高度(由可达性决定, 见 XML 注释)
SLEEVE_R_OUT = 0.015  # 外半径 15mm
SLEEVE_R_IN = 0.008  # 内孔半径 8mm
SLEEVE_LEN = 0.060  # 长 60mm
# 套筒**横躺**在 V 型托架上(轴线沿 ±y), 中心高度 0.110; 夹起后抬到 0.125 = 细杆轴线高度。
# 为什么横躺: 立着的 32g/60mm 套筒中段被垫片一推就会倒(临界力 ~0.13N), 躺着则由托架约束住。
# 为什么是 0.110: "手指朝下"姿态族的可达上限约 z=0.13, 而指尖比夹持中心低 55mm。
SLEEVE_HOME = np.array([0.000, -0.186, 0.110])
SLEEVE_AXIS = np.array([0.0, -1.0, 0.0])  # 套筒横躺时的轴线方向(手部 z_h 要对上它)

# 细杆从后方立柱水平伸出指向机器人: 局部 +z = 杆轴(基准方向 +y), 杆尖在局部 z=0.08
ROD_BASE = np.array([0.000, -0.300, 0.113])
ROD_REF_AXIS = np.array([0.0, 1.0, 0.0])  # 倾角 0 时的杆轴方向
ROD_R = 0.005  # 半径 5mm
ROD_LEN = 0.080  # 露出 8cm(杆尖 y=-0.22: 让整个插装行程都落在臂展内)

# ---------------------------------------------------------------- 任务参数
INSERT_SUCCESS = 0.030  # 杆插入内孔 >= 30mm 记为成功
INSERT_TARGET = 0.042  # 脚本演示推到此深度后停止
ALIGN_TOL_DEG = 8.0  # 轴线夹角容差
LATERAL_TOL = (SLEEVE_R_IN - ROD_R) * 0.8  # 两轴线间距容差 = 2.4mm

HOVER_DZ = 0.0  # 起始悬停高度(该姿态族抬不高, 直接同高接近)
HOVER_BACK = 0.028  # 悬停在套筒**后方**(+y 方向)2.8cm, 沿套筒轴线水平接近
LIFT_DZ = 0.003  # 不需要抬起: 套筒与细杆同轴同高, 顺着轴线滑过去即可
PRE_INSERT_GAP = 0.012  # 对准时套筒端面离杆尖 12mm
CLEAR_GAP = 0.030  # 粗对准点再往杆外多退 30mm(先对轴线, 再沿轴线推进)
PRE_GRASP_OFFSET = 0.0025  # 接近套筒时手部沿夹紧方向偏置, 让固定侧垫片不蹭到套筒
SLEEVE_JITTER_M = 0.006  # 套筒摆放位置抖动

JAW_OPEN = 1.4
# 夹紧时两指间距比套筒外径小多少。过盈越大夹持力越大, 但套筒是 16 边形近似圆柱,
# 平垫片压在多边形面上会随腕部姿态出现"楔形"效应把力放大到上百牛, 所以取小过盈,
# 再靠较大的摩擦系数保证不掉。
GRIP_SQUEEZE = 0.001
GRIP_EXTRA_SQUEEZE = 0.001  # 若闭爪后没接触到套筒, 再多夹这么多
# 夹爪力矩上限(Nm)。原来的 ±3.5Nm 会让夹持力到几十上百牛, 套筒在爪子里完全动不了;
# 插装时靠导入喇叭口自定心需要套筒能在爪子里轻微滑动, 所以夹持力做成柔顺的(约 5N)。
JAW_CLOSE_LIMIT = 3.5  # 闭合阶段力矩上限(和模型默认一致): 够大才能快速闭爪
JAW_FORCE_LIMIT = 0.40  # 夹住之后的力矩上限 -> 约 24N 夹持力(拖动套筒需要, 小了会滑脱)
GRIP_FORCE_TARGET = 2.0  # 接触判据: 夹持力超过它就切到柔顺档

FPS = 30
TIMESTEP = 0.002
SUBSTEPS = int(round(1.0 / FPS / TIMESTEP))  # 17 子步 -> 34ms/步
MAX_JOINT_STEP = 0.08  # 每控制周期关节最大变化(rad), 保证平滑

Z_AXIS = np.array([0.0, 0.0, 1.0])

# 抓取姿态的 IK 种子: 手指朝下、夹爪 z_h 水平(= 横躺套筒的轴线)。
# 就是 so100_pick_env 那套"从上方抓"的姿态族, 探测确认它是本臂唯一宽可达的姿态族。
GRASP_SEED = np.array([0.25, -1.50, 1.62, 1.45, -1.83])


# ---------------------------------------------------------------------- 小工具
def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def quat_align_z(axis: np.ndarray) -> np.ndarray:
    """最小旋转四元数 (w x y z): 把局部 +z 转到给定单位轴。"""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    z = Z_AXIS
    if np.allclose(axis, z):
        return np.array([1.0, 0.0, 0.0, 0.0])
    if np.allclose(axis, -z):
        return np.array([0.0, 1.0, 0.0, 0.0])
    n = np.cross(z, axis)
    n /= np.linalg.norm(n)
    ang = float(np.arccos(np.clip(axis @ z, -1.0, 1.0)))
    return np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * n])


def axis_from_tilt(tilt_deg: float, azim_deg: float,
                   ref: np.ndarray | None = None) -> np.ndarray:
    """基准方向(默认 +y)绕任意方位倾斜 tilt 度后的单位轴向量。

    方位角在垂直于基准方向的平面里定义, 因此 "倾角 0~90° + 方位 0~360°" 可以覆盖
    任意方向的杆姿态。
    """
    ref = ROD_REF_AXIS if ref is None else np.asarray(ref, float)
    t, p = np.radians(tilt_deg), np.radians(azim_deg)
    # 与 ref 垂直的两个基向量(对 +y 就是 x 与 z)
    e1 = np.cross(ref, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(ref, np.array([1.0, 0.0, 0.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ref, e1)
    return np.cos(t) * ref + np.sin(t) * (np.cos(p) * e1 + np.sin(p) * e2)


def smoothstep(t: float) -> float:
    """三次缓动, 让插值轨迹起步/收尾平缓。"""
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------- 环境
class SO100SleeveEnv:
    """套筒-细杆插装环境。渲染可关闭(render=False)以加速。"""

    def __init__(self, width: int = 224, height: int = 224, camera: str = "closeup",
                 render: bool = True):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)  # IK scratch, 不污染物理状态

        self.width, self.height = width, height
        self.camera = camera
        self.renderer = mujoco.Renderer(self.model, height, width) if render else None

        m = self.model
        self.qadr = {
            n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ALL_JOINTS
        }
        self.dofadr = {
            n: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in ALL_JOINTS
        }
        self.arm_dofs = np.array([self.dofadr[n] for n in ARM_JOINTS])
        self.act = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ALL_JOINTS}
        self.ctrl_idx = {
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(m.nu)
        }
        self.jnt_range = {
            n: m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM_JOINTS
        }

        self.sleeve_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "sleeve")
        self.rod_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "insert_rod")
        sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "sleeve_freejoint")
        self.sleeve_qadr = m.jnt_qposadr[sj]
        self.fixed_jaw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.pad_f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_4")
        self.pad_m = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_4")
        self.sleeve_geoms = set(
            range(
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sleeve_wall_00"),
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "sleeve_wall_15") + 1,
            )
        )
        self.rod_geoms = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "rod_geom"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "rod_base_geom"),
        }
        # 内孔/外圆半径 (按 16 边形内切圆, 与 XML 一致)
        self._bore_r = SLEEVE_R_IN
        self._outer_r = SLEEVE_R_OUT

        # 被抓住的套筒在"手部坐标"里的固定偏移/轴线 (抓取后测量), 默认按理想情况
        self.grasp_off_local = np.zeros(3)
        self.grasp_axis_local = Z_AXIS.copy()
        self.grasp_active = False
        # 套筒躺姿的默认四元数(直接取模型默认值, 与 XML 保持一致)
        self.sleeve_home_quat = self.model.qpos0[
            self.sleeve_qadr + 3:self.sleeve_qadr + 7
        ].copy()

        # 夹爪关节: 摩擦损失调小(menagerie 默认 0.1Nm 会把小力矩下的夹爪锁死);
        # 力矩上限默认给"闭合档", 控制器夹住以后再切到柔顺档(见 set_jaw_force_limit)。
        self.model.dof_frictionloss[self.dofadr["Jaw"]] = 0.004
        self.set_jaw_force_limit(JAW_CLOSE_LIMIT)

        # 接触分组: 只让"垫片"和套筒接触。
        # 夹爪/腕部的**网格**碰撞体与套筒之间只有亚毫米余量(套筒外径 30mm 正好等于两指内距),
        # 腕部稍微一歪网格就会咬进套筒, 形成 80N 级的自锁把手臂顶到力矩上限。
        # 这里给套筒单独一个 contype 位, 网格碰撞体不与该位相碰, 垫片(盒体)保留接触。
        self._set_contact_groups()

        self._calibrate_jaw()

        self._calibrate_jaw()
        self._calibrate_grip_point()
        self.set_rod_pose(20.0, 30.0)
        mujoco.mj_forward(m, self.data)

    # ---------------------------------------------------------------- 接触分组
    def _set_contact_groups(self):
        """套筒单独占一个 contype 位(4); 手臂的网格碰撞体只碰 1/2 位, 垫片额外碰 4 位。"""
        m = self.model
        SLEEVE_BIT = 4
        for g in self.sleeve_geoms:
            m.geom_contype[g] = SLEEVE_BIT
            m.geom_conaffinity[g] = 3  # 与环境(1)和其它部件(2)相碰
        for g in range(m.ngeom):
            if g in self.sleeve_geoms:
                continue
            if g in (self.pad_f, self.pad_m):
                m.geom_contype[g] = 1
                m.geom_conaffinity[g] = 7  # 垫片保留与套筒(4)的接触
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_contype[g] != 0:
                m.geom_contype[g] = 8
                m.geom_conaffinity[g] = 3  # 网格碰撞体不再碰套筒

    # ---------------------------------------------------------------- 标定
    def _calibrate_jaw(self, n: int = 81):
        """实测夹爪开口(Jaw角 -> 两指内表面间距), 用于把目标开口换算成 Jaw 角。"""
        js = np.linspace(-0.174, 1.75, n)
        widths = np.empty(n)
        hold = float(self.data.qpos[self.qadr["Jaw"]])
        for i, j in enumerate(js):
            self.data.qpos[self.qadr["Jaw"]] = j
            mujoco.mj_forward(self.model, self.data)
            pf, pm = self.data.geom_xpos[self.pad_f], self.data.geom_xpos[self.pad_m]
            widths[i] = np.linalg.norm(pf - pm) - 0.002  # 减去两片 pad 的厚度
        self.data.qpos[self.qadr["Jaw"]] = hold
        mujoco.mj_forward(self.model, self.data)
        self._jaw_angles, self._jaw_widths = js, widths

    def jaw_for_width(self, width: float) -> float:
        """给定两指内表面间距(mm 级), 返回对应 Jaw 关节角。"""
        return float(np.interp(width, self._jaw_widths, self._jaw_angles))

    @property
    def jaw_grip(self) -> float:
        return self.jaw_for_width(2 * self._outer_r - GRIP_SQUEEZE)

    def _calibrate_grip_point(self):
        """把手部"夹持中心"(两指 pad4 中点在夹紧姿态下的位置) 标定成手部局部常量。

        抓取点是刚体上的固定点, 因此只需在任意一个姿态下量一次, 之后就能从
        Fixed_Jaw 的世界位姿直接算出它 —— 不用真的去闭合夹爪。
        """
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.data.qpos[:]
        d.qpos[self.qadr["Jaw"]] = self.jaw_grip
        mujoco.mj_forward(self.model, d)
        p = (d.geom_xpos[self.pad_f] + d.geom_xpos[self.pad_m]) / 2
        R = d.xmat[self.fixed_jaw_id].reshape(3, 3)
        self.grip_center_local = R.T @ (p - d.xpos[self.fixed_jaw_id])

    def grasp_point(self) -> np.ndarray:
        """当前姿态下"夹紧时会抱住的那个点"(夹持中心) 的世界坐标。"""
        return self.data.xpos[self.fixed_jaw_id] + self.jaw_rot() @ self.grip_center_local

    # ---------------------------------------------------------------- 细杆
    def set_rod_pose(self, tilt_deg: float, azim_deg: float, base: np.ndarray | None = None):
        """把手部姿态写进模型: 任意倾角, 任意方位, 底座固定在台面。"""
        if base is not None:
            self.model.body_pos[self.rod_body] = base
        self.model.body_quat[self.rod_body] = quat_align_z(axis_from_tilt(tilt_deg, azim_deg))
        self.tilt_deg, self.azim_deg = float(tilt_deg), float(azim_deg)
        mujoco.mj_forward(self.model, self.data)

    def rod_base(self) -> np.ndarray:
        return self.model.body_pos[self.rod_body].copy()

    def rod_axis(self) -> np.ndarray:
        return self.data.xmat[self.rod_body].reshape(3, 3)[:, 2].copy()

    def rod_tip(self) -> np.ndarray:
        return self.rod_base() + ROD_LEN * self.rod_axis()

    # ---------------------------------------------------------------- 套筒 / 手部
    def sleeve_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(套筒中心世界坐标, 套筒轴线单位向量)"""
        return (
            self.data.xpos[self.sleeve_body].copy(),
            self.data.xmat[self.sleeve_body].reshape(3, 3)[:, 2].copy(),
        )

    def tcp(self) -> np.ndarray:
        """抓取点 = 两指 pad4 中点。"""
        return (self.data.geom_xpos[self.pad_f] + self.data.geom_xpos[self.pad_m]) / 2

    def jaw_rot(self) -> np.ndarray:
        return self.data.xmat[self.fixed_jaw_id].reshape(3, 3).copy()

    def tool_axis(self) -> np.ndarray:
        """当前套筒轴线在世界的方向(按抓取时标定的手部局部轴推算)。"""
        return self.jaw_rot() @ self.grasp_axis_local

    def finger_dir(self) -> np.ndarray:
        return -self.jaw_rot()[:, 1]  # 指尖朝向

    def arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self.qadr[n]] for n in ALL_JOINTS], dtype=np.float32)

    def measure_grasp(self):
        """测量被夹住的套筒在 hand 局部的偏移与轴向(抓取后调用)。"""
        c, a = self.sleeve_pose()
        R = self.jaw_rot()
        self.grasp_off_local = R.T @ (c - self.tcp())
        self.grasp_axis_local = R.T @ a
        self.grasp_active = True
        return self.grasp_off_local.copy(), self.grasp_axis_local.copy()

    def grasp_slip(self, max_pos: float = 0.02, max_deg: float = 25.0):
        """实测套筒相对"按手握关系预测"的偏差: (位置偏差, 姿态修正旋转)。

        套筒在夹爪里可能轻微滑移/倾斜(只有摩擦和面接触约束), 用实测值修正控制目标,
        相当于积分反馈 —— 这样对准的是套筒本体, 而不是"假设刚性连接"的预测值。
        """
        if not self.grasp_active:
            return np.zeros(3), np.eye(3)
        c, a = self.sleeve_pose()
        R = self.jaw_rot()
        pred_c = self.grasp_point() + R @ self.grasp_off_local
        a_pred = R @ self.grasp_axis_local
        dpos = c - pred_c
        n = float(np.linalg.norm(dpos))
        if n > max_pos:
            dpos *= max_pos / n
        v = np.cross(a_pred, a)
        sn, cs = float(np.linalg.norm(v)), float(np.clip(a_pred @ a, -1.0, 1.0))
        if sn < 1e-9:
            return dpos, np.eye(3) if cs > 0 else -np.eye(3)
        ang = float(np.arctan2(sn, cs))
        ang = min(ang, np.radians(max_deg))
        k = v / sn
        K = skew(k)
        R_corr = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
        return dpos, R_corr

    # ---------------------------------------------------------------- 成功判定
    def insert_metrics(self) -> dict:
        """插装状态: 轴线夹角 / 两轴线间距 / 杆在内孔中的重叠长度。"""
        axis, base = self.rod_axis(), self.rod_base()
        c, sa = self.sleeve_pose()
        rel = c - base
        t = float(rel @ axis)
        lateral = float(np.linalg.norm(rel - t * axis))
        overlap = min(ROD_LEN, t + SLEEVE_LEN / 2) - max(0.0, t - SLEEVE_LEN / 2)
        return {
            "align_cos": float(abs(axis @ sa)),
            "lateral": lateral,
            "overlap": float(overlap),
            "sleeve_center": c,
            "rod_tip": base + ROD_LEN * axis,
        }

    def is_success(self) -> bool:
        mm = self.insert_metrics()
        return bool(
            mm["align_cos"] > np.cos(np.radians(ALIGN_TOL_DEG))
            and mm["lateral"] < LATERAL_TOL
            and mm["overlap"] > INSERT_SUCCESS
        )

    def contact_force(self) -> float:
        """作用在套筒上的最大接触力(N), 用于插装时的卡阻保护。"""
        fmax = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self.sleeve_geoms or c.geom2 in self.sleeve_geoms:
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                fmax = max(fmax, float(np.linalg.norm(f6[:3])))
        return fmax

    def contact_report(self, top: int = 6, thresh: float = 0.5) -> list[tuple[str, str, float]]:
        """列出当前受力最大的若干接触对(调试用)。"""
        out = []
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            mujoco.mj_contactForce(self.model, self.data, i, f6)
            f = float(np.linalg.norm(f6[:3]))
            if f > thresh:
                c = self.data.contact[i]
                n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
                n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
                out.append((n1, n2, f))
        out.sort(key=lambda t: -t[2])
        return out[:top]

    def rod_contact_force(self) -> float:
        """套筒与细杆(含底座)之间的接触力 —— 这才是插装真正的卡阻力。

        不能拿"套筒上的全部接触力"当卡阻判据: 夹爪夹持力本身就有几十牛。
        """
        fmax = 0.0
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if (pair & self.sleeve_geoms) and (pair & self.rod_geoms):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                fmax = max(fmax, float(np.linalg.norm(f6[:3])))
        return fmax

    def grip_force(self) -> float:
        """夹爪两指作用在套筒上的接触力(N)。"""
        fmax = 0.0
        f6 = np.zeros(6)
        pads = {self.pad_f, self.pad_m}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if (pair & pads) and (pair & self.sleeve_geoms):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                fmax = max(fmax, float(np.linalg.norm(f6[:3])))
        return fmax

    # ---------------------------------------------------------------- 观测/复位/步进
    def get_obs(self, render: bool | None = None) -> dict:
        obs = {"joint_positions": self.arm_qpos()}
        do_render = self.renderer is not None if render is None else render
        if do_render:
            obs["image"] = self.render_image()
        return obs

    def render_image(self, camera: str | None = None) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera or self.camera)
        return self.renderer.render()

    def set_arm(self, q5: np.ndarray, jaw: float):
        for n, v in zip(ARM_JOINTS, q5):
            self.data.qpos[self.qadr[n]] = v
            self.data.ctrl[self.ctrl_idx[n]] = v
        self.data.qpos[self.qadr["Jaw"]] = jaw
        self.data.ctrl[self.ctrl_idx["Jaw"]] = jaw

    def set_jaw_force_limit(self, limit: float):
        """设置夹爪力矩上限(Nm)。

        模型里 dampratio=1 推出 kv≈4.5, 力矩上限直接决定闭合速度; 所以闭合过程用大力矩,
        夹住以后换小力矩做"柔顺夹持"(约 limit/0.0167 N 的夹持力), 插装时套筒才能靠导入
        喇叭口自定心。
        """
        self.model.actuator_forcerange[self.act["Jaw"]] = [-limit, limit]

    def reset(self, tilt_deg: float | None = None, azim_deg: float | None = None,
              sleeve_xy: np.ndarray | None = None, rng: np.random.Generator | None = None,
              tilt_range: tuple[float, float] = (0.0, 30.0),
              azim_range: tuple[float, float] = (0.0, 360.0), settle_steps: int = 80):
        """复位: 采样细杆倾角/方位 + 套筒摆放位置, 机械臂悬停在套筒上方。

        倾角/方位给 None 且传了 rng 时按 range 随机; 否则用给定值(默认 20°/30°)。
        """
        mujoco.mj_resetData(self.model, self.data)

        if rng is not None:
            t = float(rng.uniform(*tilt_range)) if tilt_deg is None else float(tilt_deg)
            a = float(rng.uniform(*azim_range)) if azim_deg is None else float(azim_deg)
        else:
            t = 20.0 if tilt_deg is None else float(tilt_deg)
            a = 30.0 if azim_deg is None else float(azim_deg)
        self.set_rod_pose(t, a, base=ROD_BASE)

        pos = SLEEVE_HOME.copy()
        if sleeve_xy is not None:
            pos[:2] = np.asarray(sleeve_xy, dtype=float)
        elif rng is not None:
            pos[:2] += rng.uniform(-SLEEVE_JITTER_M, SLEEVE_JITTER_M, size=2)
        self.data.qpos[self.sleeve_qadr:self.sleeve_qadr + 3] = pos
        # 躺姿四元数取模型默认值(轴线沿 ±y)
        self.data.qpos[self.sleeve_qadr + 3:self.sleeve_qadr + 7] = self.sleeve_home_quat

        # 悬停: 夹持中心停在套筒后上方(+y 方向后 5.5cm, 高 1.5cm), 手部轴线 = 套筒轴线
        hover = pos + np.array([0.0, HOVER_BACK, HOVER_DZ])
        sol = self.pose_ik(hover, SLEEVE_AXIS, q_seed=GRASP_SEED,
                           axis_local=Z_AXIS, n_iter=400, max_step=0.05, mode="pose")
        self.set_arm(sol, JAW_OPEN)
        self.grasp_off_local = np.zeros(3)
        self.grasp_axis_local = Z_AXIS.copy()
        self.grasp_active = False

        self.data.qvel[:] = 0
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    def step(self, action: np.ndarray):
        """action: (6,) 关节目标角 (rad), 顺序 = ALL_JOINTS"""
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (6,), f"action 形状应为 (6,), 得到 {action.shape}"
        for n, v in zip(ALL_JOINTS, action):
            self.data.ctrl[self.ctrl_idx[n]] = v
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        mm = self.insert_metrics()
        success = self.is_success()
        info = {
            "is_success": success,
            "overlap": mm["overlap"],
            "align_deg": float(np.degrees(np.arccos(min(1.0, mm["align_cos"])))),
            "lateral": mm["lateral"],
            "sleeve_pos": mm["sleeve_center"],
            "rod_tip": mm["rod_tip"],
            "force": self.contact_force(),
        }
        return self.get_obs(), 1.0 if success else 0.0, success, False, info

    # ---------------------------------------------------------------- 位姿 IK
    def pose_ik(self, target_pos: np.ndarray, target_axis: np.ndarray,
                q_seed: np.ndarray | None = None, axis_local: np.ndarray | None = None,
                q_ref: np.ndarray | None = None, n_iter: int = 120, damping: float = 0.03,
                w_dir: float = 0.30, w_reg: float = 0.02, max_step: float = 0.12,
                jaw: float | None = None, mode: str = "pose",
                w_clamp: float = 1.0, w_pref: float = 0.25) -> np.ndarray:
        """阻尼最小二乘 IK: 让夹持中心到达 target_pos, 同时约束手部姿态。

        mode="pose"  : 位置(3) + 套筒轴线方向(2)  —— 夹持中/对准/插装用
        mode="grasp" : 位置(3) + 夹紧方向水平(1, 强) + 套筒轴线接近竖直(1, 弱) —— 抓取前用。
                       抓一个立着的圆柱, 物理上只要求"两指夹紧方向垂直于圆柱轴"(1 维约束),
                       手指可以在垂直于夹紧方向的平面里任意倾斜。做成软约束后 5 个关节里
                       留出 1 维零空间, 手臂才能做出自然伸展的姿态(硬约束时只剩折叠解)。

        返回 (5,) 臂关节角。求解在 scratch data 上进行, 不影响物理状态。

        注意: 夹持中心 = 两指在夹紧姿态下的中点, 它会随 Jaw 开角移动(张开时整体偏移 ~2cm),
        所以 IK 里必须把 Jaw 固定在夹紧角上算, 不然目标点是错的。jaw=None 时取夹紧角。
        """
        d = self._ik_data
        d.qpos[:] = self.data.qpos[:]
        d.qpos[self.qadr["Jaw"]] = self.jaw_grip if jaw is None else jaw
        q = np.array([d.qpos[self.qadr[n]] for n in ARM_JOINTS])
        if q_seed is not None:
            q = np.asarray(q_seed, dtype=float).copy()
        a_local = Z_AXIS if axis_local is None else np.asarray(axis_local, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        target_axis = np.asarray(target_axis, dtype=float)
        target_axis = target_axis / np.linalg.norm(target_axis)
        lo_hi = np.array([self.jnt_range[n] for n in ARM_JOINTS])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        for _ in range(n_iter):
            for i, n in enumerate(ARM_JOINTS):
                d.qpos[self.qadr[n]] = q[i]
            mujoco.mj_forward(self.model, d)
            p = (d.geom_xpos[self.pad_f] + d.geom_xpos[self.pad_m]) / 2
            R = d.xmat[self.fixed_jaw_id].reshape(3, 3)
            e_pos = target_pos - p
            if np.linalg.norm(e_pos) < 5e-5:
                break
            mujoco.mj_jac(self.model, d, jacp, jacr, p, self.fixed_jaw_id)
            Jp = jacp[:, self.arm_dofs]
            Jr = jacr[:, self.arm_dofs]
            rows, errs = [Jp], [e_pos]
            if mode == "pose":
                a = R @ a_local
                e_dir = target_axis - a
                if np.linalg.norm(e_dir) < 4e-3 and np.linalg.norm(e_pos) < 5e-5:
                    break
                rows.append(w_dir * (-skew(a) @ Jr))
                errs.append(w_dir * e_dir)
            else:  # mode == "grasp"
                x_h = R[:, 0]  # 夹紧方向: 必须水平(垂直于套筒竖直轴)
                rows.append(w_clamp * (skew(x_h) @ Z_AXIS) @ Jr)
                errs.append(w_clamp * np.array([-(x_h @ Z_AXIS)]))
                z_h = R[:, 2]  # 软偏好: 套筒轴线尽量竖直(手指尽量水平, 避免捅到台面)
                s = 1.0 if (z_h @ Z_AXIS) >= 0 else -1.0
                rows.append(w_pref * (-skew(z_h) @ Jr))
                errs.append(w_pref * (s * Z_AXIS - z_h))
            if q_ref is not None and w_reg > 0:
                rows.append(w_reg * np.eye(5))
                errs.append(w_reg * (np.asarray(q_ref, float) - q))
            A = np.vstack(rows)
            b = np.concatenate(errs)
            dq = A.T @ np.linalg.solve(A @ A.T + damping**2 * np.eye(A.shape[0]), b)
            dq = np.clip(dq, -max_step, max_step)
            q = np.clip(q + dq, lo_hi[:, 0], lo_hi[:, 1])
        return q

    def ik_error(self, target_pos: np.ndarray, target_axis: np.ndarray | None = None):
        """当前抓取点到目标的距离 / 轴线夹角(度)。"""
        e_pos = float(np.linalg.norm(np.asarray(target_pos) - self.tcp()))
        if target_axis is None:
            return e_pos
        cos = float(np.clip(abs(self.tool_axis() @ np.asarray(target_axis)), 0, 1))
        return e_pos, float(np.degrees(np.arccos(cos)))

    def pose_error(self, sol: np.ndarray, target_pos: np.ndarray,
                   target_axis: np.ndarray | None = None,
                   axis_local: np.ndarray | None = None, jaw: float | None = None,
                   mode: str = "pose") -> float:
        """评估一组关节角对位姿目标的误差(m)。

        在 scratch data 上算, 不影响物理状态。用于判断 IK 解是否真的到位。

        评分口径必须和 IK 的约束一致: mode="pose" 是"位置 + 轴线方向",
        mode="grasp" 是"位置 + 夹紧方向水平"(此时轴线方向只是软偏好, 不能计分, 否则
        会选出"轴向好但位置差"的错误分支)。
        """
        d = self._ik_data
        d.qpos[:] = self.data.qpos[:]
        d.qpos[self.qadr["Jaw"]] = self.jaw_grip if jaw is None else jaw
        for i, n in enumerate(ARM_JOINTS):
            d.qpos[self.qadr[n]] = sol[i]
        mujoco.mj_forward(self.model, d)
        p = (d.geom_xpos[self.pad_f] + d.geom_xpos[self.pad_m]) / 2
        R = d.xmat[self.fixed_jaw_id].reshape(3, 3)
        err = float(np.linalg.norm(np.asarray(target_pos, float) - p))
        if mode == "grasp":
            err += 0.05 * abs(float(R[0, 2]))  # 夹紧方向必须水平
        elif target_axis is not None:
            a_local = Z_AXIS if axis_local is None else np.asarray(axis_local, float)
            a = R @ a_local
            t = np.asarray(target_axis, float)
            t = t / np.linalg.norm(t)
            err += 0.05 * float(np.linalg.norm(t - a))
        return err

    def ik_restart(self, target_pos: np.ndarray, target_axis: np.ndarray,
                   q_seed: np.ndarray, axis_local: np.ndarray | None = None,
                   mode: str = "pose", n_rand: int = 16, rng: np.random.Generator | None = None,
                   **kw) -> np.ndarray:
        """多起点 IK: 当前的局部解若到不了目标, 用随机起点重解, 取误差最小的解。

        5 自由度机械臂在"位置3 + 方向2"约束下解通常是离散分支, 顺着笛卡尔轨迹
        用阻尼最小二乘容易停在错的分支上; 卡住时换起点能跳到正确分支。
        """
        rng = rng or np.random.default_rng()
        lo_hi = np.array([self.jnt_range[n] for n in ARM_JOINTS])
        best = (1e9, np.asarray(q_seed, float).copy())
        cands = [np.asarray(q_seed, float)]
        # 一半在种子附近扰动(小范围换分支), 一半在工作空间内全域撒点(大范围换分支)
        for i in range(n_rand):
            if i % 2 == 0:
                cands.append(np.clip(q_seed + rng.normal(0, 0.8, 5), lo_hi[:, 0], lo_hi[:, 1]))
            else:
                cands.append(rng.uniform(lo_hi[:, 0], lo_hi[:, 1]))
        for c in cands:
            sol = self.pose_ik(target_pos, target_axis, q_seed=c, axis_local=axis_local,
                               mode=mode, **kw)
            e = self.pose_error(sol, target_pos, target_axis, axis_local, mode=mode)
            if e < best[0] - 1e-5:
                best = (e, sol)
        return best[1]

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


# ---------------------------------------------------------------------- 脚本演示
def run_scripted_insert(env: SO100SleeveEnv, collect_frames: bool = False, verbose: bool = False,
                        target_overlap: float = INSERT_TARGET, force_limit: float = 18.0,
                        insert_speed: float = 0.0012, max_insert_steps: int = 240) -> dict:
    """脚本化演示: 抓套筒 -> 对准细杆 -> 推入 -> 松爪 -> 退出。

    返回 actions/states/frames/success/success_flags/overlap 等轨迹数据。
    """
    actions, states, frames, flags, overlaps = [], [], [], [], []
    cur = env.arm_qpos().astype(np.float64)
    sol_prev = cur[:5].copy()  # IK 从当前构型延续(保持分支连续)

    def record(tgt: np.ndarray):
        pre_state = env.arm_qpos().copy()
        pre_frame = env.render_image() if collect_frames else None
        _, _, succ, _, info = env.step(tgt)
        actions.append(tgt.copy())
        states.append(pre_state)
        flags.append(bool(succ))
        overlaps.append(float(info["overlap"]))
        if collect_frames:
            frames.append(pre_frame)

    def control_step(eq_center, target_axis, jaw, mode="pose", pre_offset=0.0):
        """一个控制周期: 让"等效套筒中心"趋近 eq_center, 套筒轴线趋近 target_axis。

        等效中心 = 夹持中心 + R·off, 其中 off 是"手部局部: 夹持中心 -> 套筒中心"的偏移
        (抓取前为 0)。IK 真正控制的是刚体上的夹持中心, 这里做一次坐标换算。

        pre_offset: 未夹住时的额外偏置, 让手部沿 +x_h(固定侧)平移。固定侧垫片内表面离夹持
        中心恒定 14.5mm, 套筒半径 15mm, 不偏开的话垫片面会啃进套筒 0.5mm; 手整体往固定侧挪
        pre_offset 后, 垫片面就退到套筒表面之外, 接近/闭合过程中不会推着套筒走。
        """
        nonlocal cur, sol_prev
        R = env.jaw_rot()
        # 手握滑移反馈: 让"实测套筒"而不是"假设刚性连接的预测值"去对准
        dpos, R_corr = env.grasp_slip()
        goal = np.asarray(eq_center, float) - R @ env.grasp_off_local - dpos + pre_offset * R[:, 0]
        axis = R_corr @ np.asarray(target_axis, float)
        # 用"上一周期的 IK 解"做种子保持分支连续(不是用被限幅后的指令, 避免滞后)
        sol = env.pose_ik(goal, axis, q_seed=sol_prev, axis_local=env.grasp_axis_local,
                          q_ref=None, mode=mode, n_iter=150)
        if env.pose_error(sol, goal, axis, env.grasp_axis_local, mode=mode) > 0.003:
            # 局部解到不了目标(分支不对/被限位) -> 多起点重启
            sol = env.ik_restart(goal, axis, sol_prev, axis_local=env.grasp_axis_local,
                                 mode=mode, n_rand=20, max_step=0.15, n_iter=200)
        sol_prev = np.asarray(sol, float).copy()
        tgt = cur.copy()
        tgt[:5] = np.clip(sol, cur[:5] - MAX_JOINT_STEP, cur[:5] + MAX_JOINT_STEP)
        tgt[5] = jaw
        record(tgt)
        cur[:] = tgt

    def goto(eq_center, target_axis, seconds, jaw, mode="pose", pre_offset=0.0):
        """平滑插值: 等效中心/套筒轴从当前值过渡到目标(避免目标跳变让 IK 换分支)。"""
        n = max(1, int(seconds * FPS))
        R = env.jaw_rot()
        if env.grasp_active:  # 已夹住: 从实测套筒位姿出发插值
            c0, a0 = env.sleeve_pose()
        else:
            c0 = env.grasp_point() + R @ env.grasp_off_local
            a0 = (R @ env.grasp_axis_local).copy()
        c1 = np.asarray(eq_center, float)
        a1 = np.asarray(target_axis, float)
        a1 = a1 if (a1 @ a0) >= 0 else -a1  # 套筒是回转体, 轴向正负等价
        for i in range(n):
            t = smoothstep((i + 1) / n)
            control_step(c0 + (c1 - c0) * t, a0 + (a1 - a0) * t, jaw, mode, pre_offset)

    def grip(jaw_to, seconds, offset_from=0.0, force_target: float = GRIP_FORCE_TARGET):
        """闭爪(两段式): 先用大力矩快速闭合到接触, 再切小力矩做柔顺夹持。

        offset_from>0 时, 同步把接近偏置平滑收回 0(夹持方向回到对称), 不推着套筒走。
        夹持中心始终锁在套筒中心上(套筒由 V 托架固定住, 位置已知)。
        """
        j0 = float(cur[5])
        n = max(1, int(seconds * FPS))
        sc_now = env.sleeve_pose()[0].copy()
        env.set_jaw_force_limit(JAW_CLOSE_LIMIT)
        for i in range(n):
            t = (i + 1) / n
            control_step(sc_now, SLEEVE_AXIS, j0 + (jaw_to - j0) * t, "pose",
                         pre_offset=offset_from * (1 - t))
            if env.grip_force() > force_target:  # 已经夹到 -> 换柔顺档
                break
        env.set_jaw_force_limit(JAW_FORCE_LIMIT)
        for _ in range(max(1, int(0.3 * FPS))):
            control_step(sc_now, SLEEVE_AXIS, jaw_to, "pose", pre_offset=0.0)
        cur[5] = jaw_to

    phases = {}

    # ---- 1) 沿套筒轴线水平接近夹持位(带夹紧方向偏置, 不让固定侧垫片蹭到套筒) ----
    sc, _ = env.sleeve_pose()
    goto(sc, SLEEVE_AXIS, 1.0, JAW_OPEN, mode="pose", pre_offset=PRE_GRASP_OFFSET)
    R = env.jaw_rot()
    phases["descend_gap_mm"] = float(np.linalg.norm(env.grasp_point() - sc) * 1000)
    phases["descend_finger_deg"] = float(np.degrees(np.arcsin(np.clip(env.finger_dir()[2], -1, 1))))
    phases["descend_sleeve_axis_deg"] = float(np.degrees(np.arccos(
        np.clip(abs((R @ env.grasp_axis_local) @ SLEEVE_AXIS), 0, 1))))
    if verbose:
        print(f"    接近: 夹持中心与套筒中心差={phases['descend_gap_mm']:.1f}mm "
              f"指尖俯角={phases['descend_finger_deg']:.1f}° "
              f"轴线偏差={phases['descend_sleeve_axis_deg']:.1f}°")

    # ---- 2) 闭爪(同时收回偏置), 并标定"手部 -> 套筒"的固定关系 ----
    grip(env.jaw_grip, 0.8, offset_from=PRE_GRASP_OFFSET)
    if env.grip_force() < 0.5:  # 没夹到(套筒摆放有偏差) -> 再多夹一点
        grip(env.jaw_for_width(2 * env._outer_r - GRIP_SQUEEZE - GRIP_EXTRA_SQUEEZE), 0.5)
    env.measure_grasp()
    phases["grip_force_N"] = env.grip_force()
    if verbose:
        print(f"    夹爪={env.jaw_grip:.3f}rad(开口{2 * env._outer_r * 1000 - GRIP_SQUEEZE * 1000:.1f}mm) "
              f"夹持偏移={env.grasp_off_local.round(4)} 轴向={env.grasp_axis_local.round(3)} "
              f"夹持力={phases['grip_force_N']:.2f}N")

    # ---- 3) 抬起一点点脱离 V 托架(否则套筒被托架咬住, 轴向推不动) ----
    sc, _ = env.sleeve_pose()
    goto(sc + [0, 0, LIFT_DZ], env.tool_axis(), 0.5, env.jaw_grip)
    phases["lift_grip_N"] = env.grip_force()
    phases["lift_slip_mm"] = float(np.linalg.norm(
        env.sleeve_pose()[0] - (env.grasp_point() + env.jaw_rot() @ env.grasp_off_local)) * 1000)

    # ---- 4) 对准细杆轴线: 套筒本来就躺在细杆轴线延长线上, 只需把轴线摆正, 再沿轴滑到预插装位 ----
    rod_axis, rod_tip = env.rod_axis(), env.rod_tip()
    clear = rod_tip + rod_axis * (SLEEVE_LEN / 2 + CLEAR_GAP)
    pre = rod_tip + rod_axis * (SLEEVE_LEN / 2 + PRE_INSERT_GAP)
    goto(clear, rod_axis, 1.0, env.jaw_grip)
    goto(pre, rod_axis, 0.6, env.jaw_grip)
    _, sa = env.sleeve_pose()
    phases["align_axis_deg"] = float(np.degrees(np.arccos(np.clip(abs(sa @ rod_axis), 0, 1))))
    phases["pre_insert_err_mm"] = float(
        np.linalg.norm((env.grasp_point() + env.jaw_rot() @ env.grasp_off_local) - pre) * 1000)
    if verbose:
        print(f"    对准: 套筒轴与杆轴夹角={phases['align_axis_deg']:.2f}° "
              f"预插装点残差={phases['pre_insert_err_mm']:.1f}mm")

    # ---- 5) 沿杆轴推入(限幅推进 + 卡阻保护) ----
    # 位置控制下如果推不动, 力会一路堆到执行器上限, 所以推进时监控: 目标超前实测超过 2mm
    # 且持续一段时间没有进展, 就判定卡阻并停下(退开), 不再硬顶。
    steps, jammed = 0, False
    des = env.sleeve_pose()[0].copy()
    stalled = 0
    while steps < max_insert_steps:
        mm = env.insert_metrics()
        if mm["overlap"] >= target_overlap:
            break
        if env.rod_contact_force() > force_limit:  # 只看杆-套筒, 不看夹爪夹持力
            jammed = True
            break
        cur_c, _ = env.sleeve_pose()
        lead = float((des - cur_c) @ (-rod_axis))  # 目标领先实测多少(推进方向为正)
        if lead < 0.002:
            des = des - rod_axis * insert_speed
            stalled = 0
        else:
            stalled += 1
            if stalled > 20:  # 约 0.7s 没推进 -> 卡阻
                jammed = True
                break
        control_step(des, rod_axis, env.jaw_grip)
        steps += 1
    mm = env.insert_metrics()
    phases["insert_force_N"] = env.rod_contact_force()
    phases["grip_force_end_N"] = env.grip_force()
    if verbose:
        print(f"    推入: 重叠={mm['overlap'] * 1000:.1f}mm 夹角="
              f"{np.degrees(np.arccos(min(1, mm['align_cos']))):.2f}° "
              f"偏心={mm['lateral'] * 1000:.2f}mm 杆-套筒力={phases['insert_force_N']:.2f}N "
              f"夹持力={phases['grip_force_end_N']:.2f}N {'(卡阻停止)' if jammed else ''}")

    # ---- 6) 松爪 + 退开 ----
    grip(JAW_OPEN, 0.6)
    env.grasp_active = False
    env.grasp_off_local = np.zeros(3)
    env.grasp_axis_local = Z_AXIS.copy()
    sc, _ = env.sleeve_pose()
    goto(sc + [0.02, 0.07, 0.07], rod_axis, 1.0, JAW_OPEN)

    mm = env.insert_metrics()
    return {
        "actions": np.array(actions, dtype=np.float32),
        "states": np.array(states, dtype=np.float32),
        "frames": frames if collect_frames else None,
        "success": bool(env.is_success()),
        "success_flags": np.array(flags, dtype=bool),
        "overlap": np.array(overlaps, dtype=np.float32),
        "metrics": mm,
        "phases": phases,
        "insert_steps": steps,
        "jammed": jammed,
    }


# ---------------------------------------------------------------------- 自测
def _self_test():
    import cv2

    ap = argparse.ArgumentParser(description="SO-100 套筒插装自测")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--tilt", type=float, default=None, help="固定倾角(度), 不给则随机 0~30")
    ap.add_argument("--azim", type=float, default=None, help="固定方位(度), 不给则随机 0~360")
    ap.add_argument("--sweep", action="store_true", help="对固定倾角序列做扫描")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    env = SO100SleeveEnv(width=448, height=448, camera="closeup")
    rng = np.random.default_rng(0)

    cases: list[tuple[float, float]] = []
    if args.sweep:
        for t in (0.0, 10.0, 20.0, 30.0, 40.0):
            for a in (0.0, 90.0, 180.0, 270.0):
                cases.append((t, a))
    else:
        for _ in range(args.episodes):
            cases.append((
                float(args.tilt) if args.tilt is not None else float(rng.uniform(0, 30)),
                float(args.azim) if args.azim is not None else float(rng.uniform(0, 360)),
            ))

    results, frames = [], None
    for i, (tilt, azim) in enumerate(cases):
        env.reset(tilt_deg=tilt, azim_deg=azim, rng=rng if args.tilt is None else None)
        demo = run_scripted_insert(env, collect_frames=(i == 0 and not args.no_video),
                                   verbose=(i == 0))
        results.append(demo["success"])
        if i == 0 and not args.no_video:
            frames = demo["frames"]
        mm = demo["metrics"]
        print(f"  ep{i:2d} 倾角={tilt:5.1f}° 方位={azim:5.1f}° -> "
              f"{'✅ 成功' if demo['success'] else '❌ 失败'} "
              f"重叠={mm['overlap'] * 1000:5.1f}mm "
              f"夹角={np.degrees(np.arccos(min(1, mm['align_cos']))):4.2f}° "
              f"偏心={mm['lateral'] * 1000:4.2f}mm")

    ok = int(np.sum(results))
    print(f"\n成功率: {ok}/{len(results)} = {ok / len(results) * 100:.0f}%")

    if frames:
        video = out / "sleeve_insert_test.mp4"
        write_video_h264(video, frames, fps=FPS)
        print(f"视频: {video} ({len(frames)} 帧, {len(frames) / FPS:.1f}s)")
        picks = [0, len(frames) // 4, int(len(frames) * 0.5), int(len(frames) * 0.75), len(frames) - 1]
        for idx, name in zip(picks, ["hover", "grasp", "lift", "align", "final"]):
            cv2.imwrite(str(out / f"sleeve_{name}.png"), cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
    env.close()


if __name__ == "__main__":
    _self_test()
