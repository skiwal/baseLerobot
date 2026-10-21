#!/usr/bin/env python
"""SO-100 套筒-细杆插装任务环境 (MuJoCo)。

任务
    台面上**竖直立着一个空心套筒**(正六棱柱, 对边 26mm / 内孔 16mm / 长 60mm),
    旁边斜向上固定着一个**任意倾角的细杆**(半径 5mm, 与托座焊死, 不晃动)。
    机械臂要把套筒抓起来, 把内孔对准细杆轴线, 再把套筒推到杆上
    (杆插入内孔 >= 30mm 记为成功)。套筒是自由体(可被推倒), 细杆固定。

    (插装算法本身是后续研发目标, 本文件先把场景 + 抓取链路做实; 见"已知限制"。)

抓取方式 (为什么这样设计)
    夹爪指尖方向 = 手部 y_h 轴, 夹紧方向 = x_h 轴, 第三轴 z_h = 垂直于"指面"。
    套筒立着放, 夹爪从上方伸下去夹套筒的一对平面, 于是套筒轴线 == z_h(竖直)。
    六棱柱的一对平面正对垫片, 比圆柱好夹(不会滚, 垫片面与平面全面接触)。
    IK 里因此有一条轻量偏航偏好 + 一个偏航补偿(见 reset 里的注释):
    5 自由度臂在"位置3+方向2"约束下解是离散的, 偏航由分支决定,
    所以反过来把套筒绕竖轴转到"平面正对垫片"。

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
    MUJOCO_GL=egl python so100_sleeve_env.py --tilt 35 --azim 0 --episodes 3

已固定的部署构型 (用户确认)
    倾角方向 = 世界 +x (azim=0): 杆座放在 -x 侧(机器人右手边), 杆朝 +x(左手边)倒。
    套筒竖直立在台面 (0.12,-0.235); 竖直夹抓只在此点附近稳。+y/-x 方向因 SO-100 仅 5 自由度、
    工作空间偏置(基座 Rotation 关节 ±110°、竖直抓只成立在方位≈-63°)而不可达, 故锁定 +x。
    需要临时实验其他方位仍可用 --azim 覆盖。

已知限制 (实测, 待解决)
    1) 抓取链路已经稳了: 接近误差 2.3mm、夹持偏移 <1.5mm、夹持力 ~21N(两指都压上)、
       搬运不滑脱。偏航由 reset 里的补偿保证(平面正对垫片)。
    2) 对准也已经很准: 套筒轴与杆轴夹角 0.7°, 横向偏心 1.3mm(内孔单边间隙 3mm)。
       做法是"解一次终点 IK + 关节空间插值 + 短笛卡尔修正": 那条路径上逐步 IK 每步要丢
       5~12mm(局部解不行, 重启也救不回来), 而终点本身可解。
    3) **已做径向柔顺(等效 RCC), 楔死解决, 但插装还没完全通过**:
       - 横向柔顺: 推入时把横向位置环闭合在"实测偏心"上(目标 = -偏心), 杆尖顶着
         倒角产生侧向力时, 整手带着套筒往杆轴让开, 而不是让套筒在刚性垫片间楔死。
         实测夹持力从 **455N 降到 16~32N**, 杆-套筒力从 164N 降到 **0N**(不卡了)。
       - 姿态用夹紧持握锁住(不让套筒在爪里被顶歪), 让开完全由柔顺环负责。
       - 杆尖加了半球: 平头杆的 90° 棱边会啃在套筒倒角锥面上, 稍有偏心就 25N 级
         卡死; 换成球头后杆-套筒力直接归零。
       - 卡阻判据改成看"实际重叠有没有在长"(原来用目标超前量, 臂的稳态跟踪滞后
         就有 2mm, 会把"慢慢在进"误判成卡死)。
       现状: 推入能连续进 9~23mm(20° 倾角下), 之后臂跟不上/轴开始漂, 停在中途。
       剩余问题: 推入速度受臂跟踪能力限制(实际 ~0.11mm/步, 指令 0.5mm/步),
       且深进后套筒轴线会漂(3.7°->6°)。后续: (a) 把对准残差从 3.3mm 再压小;
       (b) 推入速度自适应(按实测跟踪速率调 insert_speed); (c) 杆再短一点/倾角再小
       一点, 让整段行程落在臂最舒服的区间。
    4) 倾角大的杆(>35°)对准更差: 手腕重构型行程不够, 和上面同源。
    5) 接触分组上, 只让夹爪的 8 个垫片与套筒接触(指侧碰撞体与套筒不碰):
       六棱柱对边 26mm 与指侧几何只剩亚毫米余量, 不分组的话下降时几十牛的侧向力
       会把套筒推歪。代价是臂的指侧可以"穿过"套筒(视觉上不真实, 功能上无碍)。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from video_utils import write_video_h264

_DBG_IK = False  # 临时诊断: 打印 ik_restart 返回的轴夹角

SCENE_PATH = Path(__file__).resolve().parent / "task" / "so100_sleeve_scene.xml"

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
ALL_JOINTS = ARM_JOINTS + ["Jaw"]

# ---------------------------------------------------------------- 场景常量 (与 XML 一致)
Z_AXIS = np.array([0.0, 0.0, 1.0])
X_AXIS = np.array([1.0, 0.0, 0.0])  # 竖立套筒的夹紧方向(对准一对平面)
TABLE_TOP = 0.08  # (验证用) 台面顶面高度: 降到 0.08 看预插点能否落进臂展
SLEEVE_AF = 0.026  # 六棱柱对边距离 26mm(夹爪要夹的两个平面之间的距离)
SLEEVE_R_CIRC = SLEEVE_AF / (2 * math.cos(math.radians(30)))  # 外接圆半径 15mm
SLEEVE_R_IN = 0.008  # 内孔内切半径 8mm
SLEEVE_LEN = 0.060  # 长 60mm
# 套筒**竖直立在台面上**(台面 0.08 -> 中心 0.11)。六棱柱端面是平的所以立得稳。
SLEEVE_HOME = np.array([0.120, -0.235, 0.110])  # 六棱柱竖直立在台面上(台面 0.08 -> 中心 0.11); 抓取只在此点附近稳
SLEEVE_AXIS = Z_AXIS.copy()  # 套筒轴线 = 竖直(手部 z_h 要对上它)

# 细杆: 自由体, 斜向上松卡在 V 型托座里。ROD_HOME 是杆的**下端**(= 托座原点),
# 托座/杆的朝向由倾角(相对竖直)+方位决定。
ROD_HOME = np.array([-0.040, -0.235, 0.085])  # 杆座: 与套筒同 y, x 取负(-x=机器人右手边), 朝 +x(左手边)倒(azim=0)
ROD_REF_AXIS = Z_AXIS.copy()  # 倾角 0 = 竖直向上
ROD_R = 0.004  # 半径 4mm(原 5mm: 减径使内孔单边间隙 3mm->4mm, 容下高预插点臂下垂的 ~2.6mm 偏心)
ROD_LEN = 0.080  # 露出 8cm(杆尖 y=-0.22: 让整个插装行程都落在臂展内)

# ---------------------------------------------------------------- 任务参数
INSERT_SUCCESS = 0.030  # 杆插入内孔 >= 30mm(套筒套过半)记为成功
INSERT_TARGET = 0.034  # 脚本演示推到此深度后停止(SO-100 臂展上限下稳定可达, 高倾角约 32-35mm)
ALIGN_TOL_DEG = 8.0  # 轴线夹角容差
LATERAL_TOL = (SLEEVE_R_IN - ROD_R) * 1.0  # 两轴线间距容差 = 3mm(=内孔-杆半径间隙, 干净插入的上限)

# IK 多起点重启用的随机数发生器: 固定种子 -> 每步重启用的是同一段候选序列, 结果确定。
# 不固定的话每步 default_rng() 会偶发跳到错分支, 使推入在 7mm 处非确定性卡死。
IK_RNG = np.random.default_rng(42)

HOVER_DZ = 0.055  # 悬停: 夹持中心在套筒中心正上方 5.5cm(套筒顶 0.14, 夹持中心 0.165)
HOVER_BACK = 0.0  # 竖直下降接近, 不需要后移
LIFT_DZ = 0.050  # 抓起后竖直抬起 5cm(该姿态族竖直方向可达)
PRE_INSERT_GAP = 0.012  # 对准时套筒端面离杆尖 12mm(杆尖露在孔外, 对准阶段不接触)
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
JAW_INSERT_LIMIT = 0.09  # 插装阶段的力矩上限 -> 约 5N, 柔顺以便喇叭口自定心
LATERAL_GAIN = 1.0   # 横向柔顺增益: 目标 = -GAIN * 实测偏心(1.0 = 完全闭合到杆轴)
LATERAL_MAX = 0.006  # 横向修正上限 6mm(留一点活动量, 又不是无限浮动)
GRIP_FORCE_TARGET = 2.0  # 接触判据: 夹持力超过它就切到柔顺档

FPS = 30
TIMESTEP = 0.002  # 物理步长(与场景 XML 一致; 曾试减半配合抬 kp, 但 SO-100 模型发散)
SUBSTEPS = int(round(1.0 / FPS / TIMESTEP))  # 17 子步 -> 34ms/步
MAX_JOINT_STEP = 0.08  # 每控制周期关节最大变化(rad), 保证平滑

# 抓取姿态的 IK 种子: 手指水平、夹爪 z_h 竖直(= 竖立套筒的轴线)。
# 就是 so100_pick_env 里"从上方抓"的同族构型(肘上肩下), 实测在该高度带内最好解。
GRASP_SEED = np.array([0.28, -2.60, 2.65, -0.05, -1.57])


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


def quat_z(angle: float) -> np.ndarray:
    """绕世界 z 轴旋转 angle(rad)的四元数。"""
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


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
        # 注: 臂执行器保持 menagerie 默认(kp=50, forcerange=±3.5Nm)。曾试抬 kp / 减半 dt
        # 压高预插点(z~0.255)的关节下垂, 但 SO-100 该模型对 kp 极敏感, 高增益在抓取/接近
        # 接触中直接发散; 见对齐段注释走几何/柔顺路线。

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
        # 细杆焊死在 rod_rest 上(静态 body, 没有 freejoint): 姿态只由 model.body_pos/quat 决定
        self.rest_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rod_rest")
        self.rod_body = self.rest_body
        sj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "sleeve_freejoint")
        self.sleeve_qadr = m.jnt_qposadr[sj]
        self.fixed_jaw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.pad_f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_4")
        self.pad_m = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_4")
        self.sleeve_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.sleeve_body}
        self.rod_geoms = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "rod_geom"),
        }
        # 内孔/外圆半径 (按 16 边形内切圆, 与 XML 一致)
        self._bore_r = SLEEVE_R_IN
        self._outer_r = SLEEVE_AF / 2

        self._pad_geoms = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("fixed_jaw_pad_1", "fixed_jaw_pad_2", "fixed_jaw_pad_3", "fixed_jaw_pad_4",
                      "moving_jaw_pad_1", "moving_jaw_pad_2", "moving_jaw_pad_3", "moving_jaw_pad_4")
        ]
        # 被抓住的套筒在"手部坐标"里的固定偏移/轴线 (抓取后测量), 默认按理想情况
        self.grasp_off_local = np.zeros(3)
        self.grasp_axis_local = Z_AXIS.copy()
        self.grasp_active = False
        # 套筒躺姿的默认四元数(直接取模型默认值, 与 XML 保持一致)
        self.sleeve_home_quat = self.model.qpos0[
            self.sleeve_qadr + 3:self.sleeve_qadr + 7
        ].copy()

        # ---- 移动底盘(cart): 把"机械臂根体 + 根部 RGB-D 相机"整体当作一个可运动的小车 ----
        # 输入约束: 任务感知只用 RGB-D; 但小车"自己在哪"用自身里程计(等同轮编码器)记录。
        # 实现: 给臂根体 Base 施加 (x,y,yaw) 的平面位姿, 把 base_rgbd 相机挂到 Base 上,
        # 这样相机随底盘平移+偏航自动随动; IK 用 MuJoCo 正运动学, 臂根一动世界系目标自动适配。
        self.base_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Base")
        self.cam_rgbd_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "base_rgbd")
        self._base0_pos = self.model.body_pos[self.base_body_id].copy()
        self._base0_quat = self.model.body_quat[self.base_body_id].copy()
        # 相机挂到臂根体: cam_pos/cam_quat 当作相对 Base 的局部位姿(初始 Base=identity, 局部==世界)
        self.model.cam_bodyid[self.cam_rgbd_id] = self.base_body_id
        mujoco.mj_forward(self.model, self.data)
        self.cart_pose = np.array([0.0, 0.0, 0.0])   # 里程计: (x, y, yaw) 世界系

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
        """只让"垫片"和套筒接触。

        menagerie 的夹爪里, 垫片是 8 个 mesh geom(fixed/moving_jaw_pad_1..4), 指侧还有
        别的碰撞体(盒/圆柱)。六棱柱套筒对边 26mm, 指侧碰撞体离垫片面只有亚毫米余量,
        下降时一旦蹭到就是几十牛的侧向力, 直接把套筒推歪/把臂顶开。所以把臂上除垫片外的
        所有 geom 换成单独的 contype 位(8), 让它们不与套筒(contype=4)相碰;
        垫片保留与套筒的接触(夹持靠它), 也与台面/细杆保持相碰。
        """
        m = self.model
        SLEEVE_BIT = 4
        arm_bodies = {m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                      for n in ALL_JOINTS}
        for g in self.sleeve_geoms:
            m.geom_contype[g] = SLEEVE_BIT
            m.geom_conaffinity[g] = 3  # 与环境(1)和其它部件(2)相碰
        for g in range(m.ngeom):
            if g in self.sleeve_geoms:
                continue
            if g in self._pad_geoms:
                m.geom_contype[g] = 1
                m.geom_conaffinity[g] = 7  # 垫片: 保留与套筒(4)的接触
            elif m.geom_bodyid[g] in arm_bodies:
                m.geom_contype[g] = 8
                m.geom_conaffinity[g] = 3  # 臂上其它 geom 不碰套筒

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
        return self.jaw_for_width(2 * self._outer_r - GRIP_SQUEEZE)  # 2*outer = 对边距离

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
    def set_rod_pose(self, tilt_deg: float, azim_deg: float,
                     base: np.ndarray | None = None, slide: float = 0.0):
        """设置细杆(及其 V 型托座)的姿态: 任意倾角(相对竖直)/方位。

        细杆是**自由体**, 所以写的是它的 qpos; 托座是静态 body, 写 model.body_pos/quat。
        托座原点 = 细杆下端, 局部 +z = 杆轴, 局部 +y = "上"(世界 +z 投到垂直于杆轴的平面),
        这样 V 口永远朝上, 杆靠摩擦留在里面(可以被夹爪拔出来)。
        """
        base = ROD_HOME if base is None else np.asarray(base, float)
        d, up, x, q = self._tilt_frame(tilt_deg, azim_deg)
        # 细杆与座焊死在同一静态 body 上, 只需写 body_pos/quat(杆固定不晃动)
        self.model.body_pos[self.rest_body] = base + slide * d
        self.model.body_quat[self.rest_body] = q
        self.tilt_deg, self.azim_deg = float(tilt_deg), float(azim_deg)
        mujoco.mj_forward(self.model, self.data)

    def set_cart_pose(self, x: float, y: float, yaw: float):
        """设置小车(臂根体+根部相机)的平面位姿 (x, y, yaw); yaw 绕世界 z。

        Base 体随之平移+偏航, 相机因挂在其上而自动随动; IK 用正运动学自动适配世界系目标。
        cart_pose 即里程计(仿真里由指令直接记录, 等同轮编码器)。
        """
        self.cart_pose[:] = [float(x), float(y), float(yaw)]
        # Base 初始为 identity, 直接以绕 z 的偏航四元数(模块 quat_z 返回 (w,x,y,z), 与 MuJoCo 一致)
        self.model.body_quat[self.base_body_id] = quat_z(float(yaw))
        self.model.body_pos[self.base_body_id] = [float(x), float(y), self._base0_pos[2]]
        mujoco.mj_forward(self.model, self.data)

    def camera_extrinsics(self, cam: str = "base_rgbd"):
        """返回相机世界系外参 (pos, right, up, back), 供 perceive.c 动态反投影用。

        back 指向观察者(相机看向 -back); 三者为单位正交基, 列即相机到世界的旋转。
        """
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        pos = self.data.cam_xpos[cid].copy()
        mat = self.data.cam_xmat[cid].reshape(3, 3).copy()   # 列: right, up, back(世界系)
        return pos, mat[:, 0], mat[:, 1], mat[:, 2]

    @staticmethod
    def _tilt_frame(tilt_deg: float, azim_deg: float):
        """倾角(相对竖直)+方位 -> (杆轴, 上方向, 横向, 四元数)。"""
        th, ph = math.radians(tilt_deg), math.radians(azim_deg)
        d = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        z = Z_AXIS
        up = z - (z @ d) * d
        n = np.linalg.norm(up)
        if n < 1e-6:  # 杆竖直时退化: V 口朝哪边都行, 取 +x
            up = X_AXIS.copy()
        else:
            up = up / n
        x = np.cross(up, d)
        M = np.eye(3)
        M[:, 0], M[:, 1], M[:, 2] = x, up, d
        tr = np.trace(M)
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            q = np.array([0.25 * s, (M[2, 1] - M[1, 2]) / s,
                          (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s])
        else:
            i = int(np.argmax([M[0, 0], M[1, 1], M[2, 2]]))
            if i == 0:
                s = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
                q = np.array([(M[2, 1] - M[1, 2]) / s, 0.25 * s,
                              (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s])
            elif i == 1:
                s = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
                q = np.array([(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s,
                              0.25 * s, (M[1, 2] + M[2, 1]) / s])
            else:
                s = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
                q = np.array([(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s,
                              (M[1, 2] + M[2, 1]) / s, 0.25 * s])
        return d, up, x, q

    def rod_base(self) -> np.ndarray:
        return self.data.xpos[self.rod_body].copy()

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

    def rod_contact_force_vec(self) -> np.ndarray:
        """套筒与细杆接触力的合力向量(N, 世界系) —— 用于力控柔顺插入。

        对全部"套筒-细杆"接触取 3D 力并求和, 即可得到把套筒顶离杆轴的合力方向;
        沿杆轴分解出的横向分量, 反向施加即把套筒推回杆心(力控 RCC), 从而把斜压在
        杆侧的径向接触力卸掉、大幅降低接触力。
        """
        f = np.zeros(3)
        f6 = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = {c.geom1, c.geom2}
            if (pair & self.sleeve_geoms) and (pair & self.rod_geoms):
                mujoco.mj_contactForce(self.model, self.data, i, f6)
                f += f6[:3]
        return f

    def grip_force(self) -> float:
        """夹爪两指作用在套筒上的最大接触力(N)。

        统计全部 8 个垫片: 套筒轴线只要偏 1~2 度, 就只有其中两三个垫片真的压到,
        只看第 4 号垫片会把"已经夹住"误报成没夹住。
        """
        fmax = 0.0
        f6 = np.zeros(6)
        pads = set(self._pad_geoms)
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

    def render_rgbd(self, camera: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """渲染指定相机的 RGB 与深度(RGB-D)。

        返回 (rgb, depth):
          rgb   : (H, W, 3) uint8
          depth : (H, W) float32, 单位米(无几何处为远平面大值)
        深度通过 Renderer 的 enable/disable_depth_rendering 切换, 与 RGB 各渲一次。
        """
        cam = camera or self.camera
        r = self.renderer
        r.disable_depth_rendering()
        r.update_scene(self.data, cam)
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(self.data, cam)
        depth = r.render()
        r.disable_depth_rendering()
        return rgb, depth

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
              tilt_range: tuple[float, float] = (20.0, 50.0),
              azim_range: tuple[float, float] = (0.0, 360.0), settle_steps: int = 80,
              sleeve_xy_range: tuple[tuple[float, float], tuple[float, float]] | None = None):
        """复位: 采样细杆倾角/方位(相对竖直, 斜向上) + 套筒摆放位置, 臂悬停在套筒侧上方。

        倾角/方位给 None 且传了 rng 时按 range 随机; 否则用给定值(默认 35°/0°)。
        方位 0 = 杆朝 +x(工作区中心/机器人左侧)内倾, 即原 270°(朝 -y 外倾)左转 90°。
        细杆是自由体, 只写初始位姿, 之后靠 V 托座的摩擦留在原地。

        套筒摆放: sleeve_xy 指定/优先; 否则传了 sleeve_xy_range 时在其范围内随机(任务随机化,
        套筒竖姿不变、位置变); 再否则传 rng 时加 SLEEVE_JITTER_M 抖动; 都不给则固定在 SLEEVE_HOME。
        """
        mujoco.mj_resetData(self.model, self.data)

        if rng is not None:
            t = float(rng.uniform(*tilt_range)) if tilt_deg is None else float(tilt_deg)
            a = float(rng.uniform(*azim_range)) if azim_deg is None else float(azim_deg)
        else:
            t = 35.0 if tilt_deg is None else float(tilt_deg)
            a = 0.0 if azim_deg is None else float(azim_deg)
        slide = 0.0 if rng is None else float(rng.uniform(-0.015, 0.015))
        self.set_rod_pose(t, a, slide=slide)

        pos = SLEEVE_HOME.copy()
        if sleeve_xy is not None:
            pos[:2] = np.asarray(sleeve_xy, dtype=float)
        elif sleeve_xy_range is not None and rng is not None:
            pos[0] = float(rng.uniform(*sleeve_xy_range[0]))
            pos[1] = float(rng.uniform(*sleeve_xy_range[1]))
        elif rng is not None:
            pos[:2] += rng.uniform(-SLEEVE_JITTER_M, SLEEVE_JITTER_M, size=2)
        self.data.qpos[self.sleeve_qadr:self.sleeve_qadr + 3] = pos
        # 竖立姿态: 四元数取单位四元数(套筒局部 +z = 世界 +z)
        self.data.qpos[self.sleeve_qadr + 3:self.sleeve_qadr + 7] = [1.0, 0.0, 0.0, 0.0]

        # 悬停: 夹持中心停在套筒正上方, 手部轴线 = 套筒轴线(竖直)
        hover = pos + np.array([0.0, HOVER_BACK, HOVER_DZ])
        sol = self.ik_restart(hover, SLEEVE_AXIS, GRASP_SEED, axis_local=Z_AXIS,
                              mode="pose", n_rand=24, max_step=0.05, n_iter=300,
                              clamp_ref=X_AXIS, rng=IK_RNG)
        # 偏航补偿: 5 自由度臂在"位置3+方向2"约束下解是离散的, 夹紧方向 x_h 的偏航由
        # 分支决定(实测约 16°, 没法用软偏好细调)。六棱柱的平面必须正对垫片, 所以反过来
        # 把套筒绕竖轴转到"平面正对夹紧方向"—— 套筒是自由体, 绕自身轴怎么转都站得稳。
        yaw = self._clamp_yaw(sol)
        if yaw is not None:
            self.data.qpos[self.sleeve_qadr + 3:self.sleeve_qadr + 7] = quat_z(yaw)
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
                w_clamp: float = 1.0, w_pref: float = 0.25,
                clamp_ref: np.ndarray | None = None, w_yaw: float = 0.005) -> np.ndarray:
        """阻尼最小二乘 IK: 让夹持中心到达 target_pos, 同时约束手部姿态。

        约束 = 位置(3) + 套筒轴线方向(2) + 可选的偏航偏好(1, 权重 w_yaw)。
        clamp_ref 给的是"夹紧方向 x_h 应对准的世界方向": 六棱柱不像圆柱那样对绕自身轴
        的偏转无所谓, 垫片必须压在一对平面上, 所以抓竖立的套筒时传 X_AXIS(套筒默认
        朝向正好有一对平面朝 ±x)。w_clamp/w_pref 只是给旧调用留着, 两种 mode 完全一样。

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
            # 位置 + 套筒轴线 + 轻量的"手指朝下"偏好。
            # 最后一项对**棱柱**很关键: 圆柱对"绕自身轴偏转"无所谓, 六棱柱的夹紧方向
            # 必须对准一对平面(x_h 落在世界 ±x), 否则垫片会压在棱角上夹不稳。
            a = R @ a_local
            e_dir = target_axis - a
            if np.linalg.norm(e_dir) < 4e-3 and np.linalg.norm(e_pos) < 5e-5:
                break
            rows.append(w_dir * (-skew(a) @ Jr))
            errs.append(w_dir * e_dir)
            if clamp_ref is not None:  # 偏航偏好: 夹紧方向对准一对平面
                x_h = R[:, 0]
                cr = np.asarray(clamp_ref, dtype=float)
                cr = cr / np.linalg.norm(cr)
                rows.append(w_yaw * (-skew(x_h) @ Jr))
                errs.append(w_yaw * (cr - x_h))
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

    def _clamp_yaw(self, sol: np.ndarray, jaw: float | None = None) -> float | None:
        """给定关节解, 返回夹紧方向(x_h)在水平面上的偏航角(rad); 失败返回 None。

        在 scratch data 上算, 不影响物理状态。用于"把套筒绕竖轴转到平面正对垫片"。
        """
        d = self._ik_data
        d.qpos[:] = self.data.qpos[:]
        d.qpos[self.qadr["Jaw"]] = self.jaw_grip if jaw is None else jaw
        for i, n in enumerate(ARM_JOINTS):
            d.qpos[self.qadr[n]] = sol[i]
        mujoco.mj_forward(self.model, d)
        x_h = d.xmat[self.fixed_jaw_id].reshape(3, 3)[:, 0]
        return float(math.atan2(x_h[1], x_h[0]))

    def pose_error(self, sol: np.ndarray, target_pos: np.ndarray,
                   target_axis: np.ndarray | None = None,
                   axis_local: np.ndarray | None = None, jaw: float | None = None,
                   mode: str = "pose", clamp_ref: np.ndarray | None = None) -> float:
        """评估一组关节角对位姿目标的误差(m)。

        在 scratch data 上算, 不影响物理状态。用于判断 IK 解是否真的到位。

        评分口径和 IK 一致: 位置 + 0.05×轴线误差 + 0.02×偏航误差(小权重, 只用来在
        同分时偏好"夹紧方向对准棱柱平面"的解)。
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
        if target_axis is not None:
            a_local = Z_AXIS if axis_local is None else np.asarray(axis_local, float)
            a = R @ a_local
            t = np.asarray(target_axis, float)
            t = t / np.linalg.norm(t)
            err += 0.05 * float(np.linalg.norm(t - a))
        if clamp_ref is not None:
            cr = np.asarray(clamp_ref, dtype=float)
            cr = cr / np.linalg.norm(cr)
            err += 0.02 * float(np.linalg.norm(cr - R[:, 0]))
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
        best_align = None  # 优先: 轴对齐且"与种子分支连续"(腕翻转最小)的对齐解
        align_cos = math.cos(0.15)  # ~8.6° 阈值: 套筒轴对齐要好于此角才被优先选
        axis_t = None if target_axis is None else np.asarray(target_axis, float)
        axis_l = Z_AXIS if axis_local is None else np.asarray(axis_local, float)
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
            e = self.pose_error(sol, target_pos, target_axis, axis_local, mode=mode,
                                clamp_ref=kw.get("clamp_ref"))
            if e < best[0] - 1e-5:
                best = (e, sol)
            # 方向对齐优先: pose_error 的方向权重只有 0.05, 不足以压制"套筒竖直"这个
            # 强吸引盆; 这里单独维护一个"轴夹角<阈值 且 误差最小"的解, 优先返回它。
            # 若真不可达(没有任何对齐解), 退回位置最优的 best。
            if target_axis is not None:
                _R = self._ik_data.xmat[self.fixed_jaw_id].reshape(3, 3)
                _a = _R @ axis_l
                if abs(float(np.clip(_a @ axis_t, -1, 1))) > align_cos:
                    # 选"分支连续"的对齐解: 与种子关节差最小, 避免腕大幅翻转
                    # (误差最小的对齐解常是翻转分支, 夹爪翻面把套筒甩出爪)
                    dq = float(np.linalg.norm(sol - np.asarray(q_seed, float)))
                    if best_align is None or dq < best_align[2]:
                        best_align = (e, sol, dq)
        chosen = best_align[:2] if best_align is not None else best
        if _DBG_IK:
            d = self._ik_data
            d.qpos[:] = self.data.qpos[:]
            d.qpos[self.qadr["Jaw"]] = self.jaw_grip
            for i, n in enumerate(ARM_JOINTS):
                d.qpos[self.qadr[n]] = chosen[1][i]
            mujoco.mj_forward(self.model, d)
            _R = d.xmat[self.fixed_jaw_id].reshape(3, 3)
            _a = _R @ axis_l
            _ang = float(np.degrees(np.arccos(np.clip(abs(_a @ axis_t), 0, 1))))
            sys.stderr.write(f"[ik_restart] err={chosen[0]:.4f} axis_deg={_ang:.1f} "
                             f"n_rand={n_rand} aligned={best_align is not None}\n")
        return chosen[1]

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


# ---------------------------------------------------------------------- 脚本演示
def run_scripted_insert(env: SO100SleeveEnv, collect_frames: bool = False, verbose: bool = False,
                        target_overlap: float = INSERT_TARGET, force_limit: float = 40.0,
                        insert_speed: float = 0.00015, max_insert_steps: int = 600) -> dict:
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

    def control_step(eq_center, target_axis, jaw, mode="pose", pre_offset=0.0,
                     clamp_ref=X_AXIS, slip_rot=True):
        """一个控制周期: 让"等效套筒中心"趋近 eq_center, 套筒轴线趋近 target_axis。

        等效中心 = 夹持中心 + R·off, 其中 off 是"手部局部: 夹持中心 -> 套筒中心"的偏移
        (抓取前为 0)。IK 真正控制的是刚体上的夹持中心, 这里做一次坐标换算。

        pre_offset: 未夹住时的额外偏置, 让手部沿 +x_h(固定侧)平移。固定侧垫片内表面离夹持
        中心恒定 14.5mm, 套筒对边半距 13mm, 不偏开的话垫片面会啃进套筒 0.5mm; 手整体往固定侧挪
        pre_offset 后, 垫片面就退到套筒表面之外, 接近/闭合过程中不会推着套筒走。
        clamp_ref: 夹紧方向应对准的世界方向(六棱柱要压在一对平面上); 传 None 则不约束偏航。
        """
        nonlocal cur, sol_prev
        R = env.jaw_rot()
        # 手握滑移反馈: 让"实测套筒"而不是"假设刚性连接的预测值"去对准。
        # slip_rot=False 时只修位置不修姿态: 插装阶段套筒被杆顶歪时, 若让臂的轴线跟着
        # 套筒歪, 就变成正反馈(越顶越歪); 关掉它, 臂的轴线钉在杆轴上, 套筒在松持的
        # 爪里自己找正 —— 这就是 RCC 想要的行为。
        dpos, R_corr = env.grasp_slip()
        if not slip_rot:
            R_corr = np.eye(3)
        goal = np.asarray(eq_center, float) - R @ env.grasp_off_local - dpos + pre_offset * R[:, 0]
        axis = R_corr @ np.asarray(target_axis, float)
        # 用"上一周期的 IK 解"做种子保持分支连续(不是用被限幅后的指令, 避免滞后)
        sol = env.pose_ik(goal, axis, q_seed=sol_prev, axis_local=env.grasp_axis_local,
                          q_ref=None, mode=mode, n_iter=150, clamp_ref=clamp_ref)
        if env.pose_error(sol, goal, axis, env.grasp_axis_local, mode=mode,
                          clamp_ref=clamp_ref) > 0.003:
            # 局部解到不了目标(分支不对/被限位) -> 多起点重启
            sol = env.ik_restart(goal, axis, sol_prev, axis_local=env.grasp_axis_local,
                                 mode=mode, n_rand=60, max_step=0.15, n_iter=200,
                                 clamp_ref=clamp_ref, rng=IK_RNG)
        sol_prev = np.asarray(sol, float).copy()
        tgt = cur.copy()
        tgt[:5] = np.clip(sol, cur[:5] - MAX_JOINT_STEP, cur[:5] + MAX_JOINT_STEP)
        tgt[5] = jaw
        record(tgt)
        cur[:] = tgt

    def goto(eq_center, target_axis, seconds, jaw, mode="pose", pre_offset=0.0,
             clamp_ref=X_AXIS):
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
            control_step(c0 + (c1 - c0) * t, a0 + (a1 - a0) * t, jaw, mode, pre_offset,
                         clamp_ref)

    def goto_joint(q_target, seconds, jaw):
        """关节空间平滑插值到给定构型(不做逐步 IK)。

        对准阶段用: 那条路径上逐步 IK 每步丢 5mm+, 而终点可以精确解出, 所以直接插关节。
        """
        nonlocal cur, sol_prev
        # 从真实关节角出发, 不用 cur(指令目标): 物理滞后使 cur 超前实际构型,
        # 直接拿它插值会让臂"瞬移"回指令位姿, 轨迹突变把套筒甩出爪。
        q0 = env.arm_qpos()[:5].copy()
        q1 = np.asarray(q_target, dtype=float).copy()
        n = max(1, int(seconds * FPS))
        for i in range(n):
            t = smoothstep((i + 1) / n)
            tgt = cur.copy()
            tgt[:5] = q0 + (q1 - q0) * t
            tgt[5] = jaw
            record(tgt)
            env.step(tgt)
            cur[:] = tgt
        sol_prev = q1.copy()

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
            control_step(sc_now, SLEEVE_AXIS, j0 + (jaw_to - j0) * t, "grasp",
                         pre_offset=offset_from * (1 - t))
            if env.grip_force() > force_target:  # 已经夹到 -> 换柔顺档
                break
        env.set_jaw_force_limit(JAW_FORCE_LIMIT)
        for _ in range(max(1, int(0.3 * FPS))):
            control_step(sc_now, SLEEVE_AXIS, jaw_to, "grasp", pre_offset=0.0)
        cur[5] = jaw_to

    phases = {}

    # ---- 1) 沿套筒轴线水平接近夹持位(带夹紧方向偏置, 不让固定侧垫片蹭到套筒) ----
    sc, _ = env.sleeve_pose()
    goto(sc, SLEEVE_AXIS, 1.0, JAW_OPEN, mode="grasp", pre_offset=PRE_GRASP_OFFSET)
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
        con = [(a, b, round(f, 1)) for a, b, f in env.contact_report(3, 0.5)]
        if con:
            print(f"          接触: {con}")

    # ---- 3) 抬起一点点脱离 V 托架(否则套筒被托架咬住, 轴向推不动) ----
    sc, _ = env.sleeve_pose()
    goto(sc + [0, 0, LIFT_DZ], env.tool_axis(), 0.5, env.jaw_grip)
    phases["lift_grip_N"] = env.grip_force()
    phases["lift_slip_mm"] = float(np.linalg.norm(
        env.sleeve_pose()[0] - (env.grasp_point() + env.jaw_rot() @ env.grasp_off_local)) * 1000)
    if verbose:
        print(f"    抬起: 夹持力={phases['lift_grip_N']:.2f}N 握滑={phases['lift_slip_mm']:.1f}mm")

    # ---- 4) 对准细杆轴线: 先竖直平移到 pre 位置(套筒轴保持竖直, 纯平移不换分支),
    # 再在 pre 处用关节空间把轴从竖直转到 rod_axis("先移后转")。
    # 直接笛卡尔 goto(clear/rod_axis) 会在逐帧 IK 里卡在"套筒竖直"分支(位置对、轴向错),
    # 因为 5 自由度在该位姿下"竖直"与"倾斜"是两个离散分支; 关节空间插值在已经解出的
    # 倾斜构型上做连续运动, 不会重新解 IK 换分支, 套筒轴就能真正转到 rod_axis。
    # goto / ik_restart 默认 clamp_ref=X_AXIS -> 六棱柱平面始终正对垫片。对齐段加紧固夹持防滑。
    rod_axis, rod_tip = env.rod_axis(), env.rod_tip()
    pre = rod_tip + rod_axis * (SLEEVE_LEN / 2 + PRE_INSERT_GAP)
    env.set_jaw_force_limit(0.8)  # 对齐段更紧的夹持, 防套筒轴向滑脱
    # 段1: 竖直平移到 pre 位置(轴仍竖直)
    goto(pre, SLEEVE_AXIS, 1.3, env.jaw_grip)
    # 段2: 解出 pre 处的倾斜构型, 关节空间原地绕腕转(不重新解 IK, 不换分支)
    q_tilt = env.ik_restart(pre, rod_axis, cur[:5], axis_local=env.grasp_axis_local,
                            mode="pose", n_rand=800, max_step=0.06, n_iter=300,
                            clamp_ref=X_AXIS, rng=IK_RNG)
    goto_joint(q_tilt, 1.2, env.jaw_grip)
    sol_prev = cur[:5].copy()
    _, sa = env.sleeve_pose()
    phases["align_axis_deg"] = float(np.degrees(np.arccos(np.clip(abs(sa @ rod_axis), 0, 1))))
    phases["pre_insert_err_mm"] = float(
        np.linalg.norm((env.grasp_point() + env.jaw_rot() @ env.grasp_off_local) - pre) * 1000)
    phases["align_slip_mm"] = float(np.linalg.norm(
        env.sleeve_pose()[0] - (env.grasp_point() + env.jaw_rot() @ env.grasp_off_local)) * 1000)
    mm0 = env.insert_metrics()
    phases["align_lateral_mm"] = float(mm0["lateral"] * 1000)
    if verbose:
        print(f"    对准: 轴夹角={phases['align_axis_deg']:.2f}° 预插装残差={phases['pre_insert_err_mm']:.1f}mm "
              f"握滑={phases['align_slip_mm']:.1f}mm 横向偏心={phases['align_lateral_mm']:.1f}mm")

    # ---- 5) 沿杆轴推入(限幅推进 + 横向柔顺 + 卡阻保护) ----
    # 横向柔顺 = 等效 RCC(远端柔顺中心): 横向位置环不钉死目标, 而是闭合在"实测偏心"上 ——
    # 每步量出套筒中心相对杆轴的偏心 e, 目标就往 -e 方向修。杆尖顶着喇叭口产生侧向力时,
    # 手会带着套筒往杆轴让开(而不是把套筒楔死在刚性垫片上), 喇叭口就能自定心。
    # 真实机器人上这一环由 RCC 被动机构或腕力传感器完成, 这里用仿真里可测的偏心等效实现。
    # 夹紧持握(锁住套筒姿态, 不让它在爪里被顶歪), 横向让开完全由柔顺环负责 ——
    # 整手带着套筒往杆轴偏, 而不是让套筒在爪里滑动, 这才是 RCC 的行为。
    env.set_jaw_force_limit(0.7)  # 推入段稍紧一点: 高倾角时套筒受重力矩易在爪里转, 夹紧防漂
    steps, jammed = 0, False
    des = env.sleeve_pose()[0].copy()
    stalled = 0
    best_overlap = -1.0
    rod_base = env.rod_base()
    while steps < max_insert_steps:
        mm = env.insert_metrics()
        if mm["overlap"] >= target_overlap:
            break
        if env.rod_contact_force() > force_limit:  # 安全阀: 超过就停, 不再硬顶
            jammed = True
            break
        cur_c, _ = env.sleeve_pose()
        # 实测偏心(套筒中心到杆轴的垂距向量) -> 横向修正量
        rel = cur_c - rod_base
        e = rel - float(rel @ rod_axis) * rod_axis
        lateral = np.clip(-LATERAL_GAIN * e, -LATERAL_MAX, LATERAL_MAX)
        lead = float((des - cur_c) @ (-rod_axis))  # 目标领先实测多少(推进方向为正)
        # 原逻辑: lead<2mm 才前进 des。但高偏心下(套筒轴偏出杆轴 2.6mm) lead 被垂直分量
        # 污染, 误判成"滞后 2mm" -> des 冻结 -> 套筒不跟 -> 5mm 死锁。改为 des 单调前进
        # (每步沿 -rod_axis 推 insert_speed), 由 stalled 重叠停滞判据兜底卡阻。
        des = des - rod_axis * insert_speed
        # 卡阻判定看"实际重叠有没有在长", 不看目标超前量: 臂的稳态跟踪滞后就有 ~2mm,
        # 用 lead 判会把"慢慢在进"误判成卡死。
        if mm["overlap"] > best_overlap + 5e-4:
            best_overlap = mm["overlap"]
            stalled = 0
        else:
            stalled += 1
            if stalled > 40:  # 约 1.3s 重叠完全没长 -> 判定卡阻
                jammed = True
                break
        control_step(des + lateral, rod_axis, env.jaw_grip, slip_rot=False, clamp_ref=None)
        steps += 1
    env.set_jaw_force_limit(JAW_FORCE_LIMIT)
    mm = env.insert_metrics()
    phases["insert_overlap_mm"] = float(mm["overlap"] * 1000)
    phases["insert_axis_deg"] = float(np.degrees(np.arccos(min(1.0, mm["align_cos"]))))
    phases["insert_lateral_mm"] = float(mm["lateral"] * 1000)
    phases["insert_force_N"] = env.rod_contact_force()
    phases["grip_force_end_N"] = env.grip_force()
    if verbose:
        print(f"    推入: 重叠={mm['overlap'] * 1000:.1f}mm 夹角="
              f"{np.degrees(np.arccos(min(1, mm['align_cos']))):.2f}° "
              f"偏心={mm['lateral'] * 1000:.2f}mm 杆-套筒力={phases['insert_force_N']:.2f}N "
              f"夹持力={phases['grip_force_end_N']:.2f}N {'(卡阻停止)' if jammed else ''}")

    # ---- 6) 收尾: 成功则保持(套筒在杆上), 失败则抬起来松爪放回台面, 然后退开 ----
    success = env.is_success()
    if not success:
        sc, _ = env.sleeve_pose()
        goto(sc + [0, 0, 0.045], env.tool_axis(), 0.6, env.jaw_grip)  # 先抬起来再松爪, 免得刮到杆
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
        # 用**释放前**(套筒还夹在杆上时)的判定; 不能用释放后的 is_success()——
        # 那时套筒已脱手、从斜杆上滑脱, 度量乱掉, 永远 False。
        "success": success,
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
    ap.add_argument("--azim", type=float, default=None, help="固定方位(度); 部署锁定 +x(azim=0), 不给则 azim=0")
    ap.add_argument("--camera", type=str, default=None,
                   help="录制相机: closeup(默认, 特写) 或 base_rgbd(机械臂根部视角, 看完整运动)")
    ap.add_argument("--sweep", action="store_true", help="对固定倾角序列做扫描")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    cam = args.camera or "closeup"
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    env = SO100SleeveEnv(width=448, height=448, camera=cam)
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
                float(args.azim) if args.azim is not None else 0.0,
            ))

    results, frames = [], None
    for i, (tilt, azim) in enumerate(cases):
        env.reset(tilt_deg=tilt, azim_deg=azim, rng=rng if args.tilt is None else None)
        demo = run_scripted_insert(env, collect_frames=(i == 0 and not args.no_video),
                                   verbose=(i == 0))
        results.append(demo["success"])
        if i == 0 and not args.no_video:
            frames = demo["frames"]
        ph = demo["phases"]
        mm = demo["metrics"]
        # 注意: demo['metrics'] 是**释放后**度量的(套筒已脱手, 数值会乱滚), 不能用来判定成败;
        # 成败看 demo['success'](释放前在杆上还夹着时算的), 重叠/夹角/偏心也用释放前存的 phases。
        print(f"  ep{i:2d} 倾角={tilt:5.1f}° 方位={azim:5.1f}° -> "
              f"{'✅ 成功' if demo['success'] else '❌ 失败'} "
              f"重叠={ph.get('insert_overlap_mm', float('nan')):5.1f}mm "
              f"夹角={ph.get('insert_axis_deg', float('nan')):4.2f}° "
              f"偏心={ph.get('insert_lateral_mm', float('nan')):4.2f}mm")

    ok = int(np.sum(results))
    print(f"\n成功率: {ok}/{len(results)} = {ok / len(results) * 100:.0f}%")

    if frames:
        video = out / ("sleeve_insert_test.mp4" if cam == "closeup" else f"sleeve_insert_test_{cam}.mp4")
        write_video_h264(video, frames, fps=FPS)
        print(f"视频: {video} ({len(frames)} 帧, {len(frames) / FPS:.1f}s)")
        picks = [0, len(frames) // 4, int(len(frames) * 0.5), int(len(frames) * 0.75), len(frames) - 1]
        for idx, name in zip(picks, ["hover", "grasp", "lift", "align", "final"]):
            cv2.imwrite(str(out / f"sleeve_{name}.png"), cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
    env.close()


if __name__ == "__main__":
    _self_test()
