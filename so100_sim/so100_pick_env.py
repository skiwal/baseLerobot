#!/usr/bin/env python
"""SO-100 抓取任务环境 (MuJoCo)。

任务: 把矮台上的 3cm 方块抓起并抬升 6cm。
接口: get_obs() / step(action) —— 与 LeRobot Robot 约定同构, 也与 gym 习惯接近。

关节约定 (6 维, 弧度):
    [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]  (等价于真机 6 个电机)

观测:
    joint_positions: (6,) float32  关节角度 (rad)
    image:           (H, W, 3) uint8 相机图像

动作:
    (6,) float32  6 个关节的目标角度 (rad), 直接写给 MuJoCo 位置执行器
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from video_utils import write_video_h264

SCENE_PATH = Path(__file__).resolve().parent / "task" / "so100_pick_scene.xml"

ARM_JOINTS = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll"]
ALL_JOINTS = ARM_JOINTS + ["Jaw"]

# 场景常量 (与 XML 保持一致)
PEDESTAL_TOP = 0.085
CUBE_HALF = 0.015
CUBE_HOME = np.array([0.0, -0.24, PEDESTAL_TOP + CUBE_HALF])  # (0, -0.24, 0.100)
SUCCESS_LIFT = 0.06  # 抬起 6cm 算成功
SUCCESS_Z = CUBE_HOME[2] + SUCCESS_LIFT  # 0.160

# 起始姿态的 IK 种子 (home 关键帧), 实际起始位由 IK 悬停位决定
START_SEED_QPOS = np.array([0.0, -1.57, 1.57, 1.57, -1.57])
HOVER_DZ = 0.10  # 起始时抓取点悬停在方块上方 10cm
CUBE_RANDOMIZE_M = 0.012  # 方块随机化范围 ±1.2cm
JAW_OPEN = 1.4
JAW_CLOSED = -0.174

FPS = 30
TIMESTEP = 0.002
SUBSTEPS = int(round(1.0 / FPS / TIMESTEP))  # 17 个子步 = 34ms


class SO100PickEnv:
    """抓取任务环境。渲染可关闭(render=False)以加速数据采集。"""

    def __init__(self, width: int = 224, height: int = 224, camera: str = "side", render: bool = True):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)  # IK 专用 scratch, 避免污染物理状态

        self.width, self.height = width, height
        self.camera = camera
        self.renderer = mujoco.Renderer(self.model, height, width) if render else None

        m = self.model
        # 关节 qpos 地址
        self.qadr = {
            name: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ALL_JOINTS
        }
        # 关节 dof 地址
        self.dofadr = {
            name: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ALL_JOINTS
        }
        self.arm_dofs = np.array([self.dofadr[name] for name in ARM_JOINTS])

        # 执行器索引 (actuator 名与 joint 名一致)
        self.act = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ALL_JOINTS}
        self.act_order = [self.act[name] for name in ALL_JOINTS]  # ctrl 向量的物理顺序

        # ctrl 是按 actuator 定义顺序排列的, 这里记录 "关节名 -> ctrl 下标"
        self.ctrl_idx = {}
        for i in range(m.nu):
            aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.ctrl_idx[aname] = i

        # 方块 freejoint qpos 地址 (7 维: pos + quat)
        cube_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
        self.cube_qadr = m.jnt_qposadr[cube_jid]

        # Fixed_Jaw body 与抓取点局部偏移
        self.fixed_jaw_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Fixed_Jaw")
        self.grasp_geoms = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_pad_4"),
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_pad_4"),
        ]
        self.cube_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pick_cube")

        # 计算抓取点在 Fixed_Jaw 局部坐标系里的偏移 (Jaw=0 张开状态)
        self._tcp_offset_local = self._compute_tcp_offset()

        self.reset()

    # ------------------------------------------------------------------ 工具
    def _compute_tcp_offset(self) -> np.ndarray:
        """抓取点(两指 pad4 中点) 在 Fixed_Jaw 局部系的偏移。"""
        # 在一个中性姿态下计算
        self.data.qpos[self.qadr["Jaw"]] = JAW_OPEN
        mujoco.mj_forward(self.model, self.data)
        tcp_w = (self.data.geom_xpos[self.grasp_geoms[0]] + self.data.geom_xpos[self.grasp_geoms[1]]) / 2
        xpos = self.data.xpos[self.fixed_jaw_id]
        xmat = self.data.xmat[self.fixed_jaw_id].reshape(3, 3)
        return xmat.T @ (tcp_w - xpos)

    def tcp(self) -> np.ndarray:
        """当前抓取点世界坐标 (Jaw 状态随实际关节角变化)。"""
        return (self.data.geom_xpos[self.grasp_geoms[0]] + self.data.geom_xpos[self.grasp_geoms[1]]) / 2

    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body_id].copy()

    def arm_qpos(self) -> np.ndarray:
        return np.array([self.data.qpos[self.qadr[n]] for n in ALL_JOINTS], dtype=np.float32)

    def is_success(self) -> bool:
        return bool(self.cube_pos()[2] > SUCCESS_Z)

    def cube_on_table(self) -> bool:
        """方块是否还在台面附近(没被撞飞)。"""
        return bool(self.cube_pos()[2] > PEDESTAL_TOP - 0.01)

    # ------------------------------------------------------------- 接口: 观测
    def get_obs(self, render: bool | None = None) -> dict:
        obs = {"joint_positions": self.arm_qpos()}
        do_render = self.renderer is not None if render is None else render
        if do_render:
            obs["image"] = self.render_image()
        return obs

    def render_image(self, camera: str | None = None) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera or self.camera)
        return self.renderer.render()

    # ------------------------------------------------------------- 接口: 复位
    def reset(self, cube_xy: np.ndarray | None = None, rng: np.random.Generator | None = None,
              settle_steps: int = 60):
        """复位: 方块归位 -> 臂通过 IK 悬停在方块上方 -> 静置。

        cube_xy: 方块的水平位置 (None 用默认; 给 rng 则随机)
        """
        mujoco.mj_resetData(self.model, self.data)

        # 方块位置
        pos = CUBE_HOME.copy()
        if cube_xy is not None:
            pos[:2] = cube_xy
        elif rng is not None:
            pos[:2] += rng.uniform(-CUBE_RANDOMIZE_M, CUBE_RANDOMIZE_M, size=2)
        self.data.qpos[self.cube_qadr : self.cube_qadr + 3] = pos
        self.data.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = [1, 0, 0, 0]

        # 悬停姿态: 抓取点悬在方块上方 HOVER_DZ
        hover = self.ik(pos + [0, 0, HOVER_DZ], q_seed=START_SEED_QPOS, n_iter=300, max_step=0.05)
        for name, val in zip(ARM_JOINTS, hover):
            self.data.qpos[self.qadr[name]] = val
            self.data.ctrl[self.ctrl_idx[name]] = val
        self.data.qpos[self.qadr["Jaw"]] = JAW_OPEN
        self.data.ctrl[self.ctrl_idx["Jaw"]] = JAW_OPEN

        self.data.qvel[:] = 0
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    # ------------------------------------------------------------- 接口: 步进
    def step(self, action: np.ndarray):
        """action: (6,) 六个关节的目标角度 (rad), 顺序 = ALL_JOINTS"""
        action = np.asarray(action, dtype=np.float64)
        assert action.shape == (6,), f"action 形状应为 (6,), 得到 {action.shape}"
        for name, val in zip(ALL_JOINTS, action):
            self.data.ctrl[self.ctrl_idx[name]] = val
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        obs = self.get_obs()
        success = self.is_success()
        info = {
            "is_success": success,
            "cube_pos": self.cube_pos(),
            "tcp": self.tcp(),
        }
        reward = 1.0 if success else 0.0
        return obs, reward, success, False, info

    # ------------------------------------------------------------------- IK
    def ik(self, target: np.ndarray, q_seed: np.ndarray | None = None,
           n_iter: int = 80, damping: float = 0.02, max_step: float = 0.15) -> np.ndarray:
        """阻尼最小二乘 IK: 求 5 个臂关节角, 使抓取点到达 target (世界坐标)。

        返回 (5,) 臂关节角。不改变物理状态 (在 scratch data 上计算)。
        抓取点固定在 Fixed_Jaw 上, 与 Jaw 关节无关, 因此雅可比精确。
        """
        d = self._ik_data
        d.qpos[:] = self.data.qpos[:]
        if q_seed is not None:
            for name, val in zip(ARM_JOINTS, q_seed):
                d.qpos[self.qadr[name]] = val

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        eye = np.eye(3)

        for _ in range(n_iter):
            mujoco.mj_forward(self.model, d)
            xpos = d.xpos[self.fixed_jaw_id]
            xmat = d.xmat[self.fixed_jaw_id].reshape(3, 3)
            tcp_w = xpos + xmat @ self._tcp_offset_local
            err = target - tcp_w
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jac(self.model, d, jacp, jacr, tcp_w, self.fixed_jaw_id)
            J = jacp[:, self.arm_dofs]  # (3, 5)
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * eye, err)
            dq = np.clip(dq, -max_step, max_step)
            for i, name in enumerate(ARM_JOINTS):
                adr = self.qadr[name]
                lo, hi = self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                d.qpos[adr] = np.clip(d.qpos[adr] + dq[i], lo, hi)

        mujoco.mj_forward(self.model, d)
        return np.array([d.qpos[self.qadr[n]] for n in ARM_JOINTS])

    def ik_error(self, target: np.ndarray) -> float:
        """当前抓取点到 target 的距离。"""
        return float(np.linalg.norm(target - self.tcp()))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


# --------------------------------------------------------------- 脚本演示策略
# 每个控制周期臂关节的最大变化量 (rad), 保证运动平滑
MAX_JOINT_STEP = 0.08

# 演示各阶段: (名称, TCP 相对方块的偏移, 时长s, 夹爪目标)
DEMO_PHASES = [
    ("hover",  [0, 0, HOVER_DZ], 0.3, JAW_OPEN),
    ("descend", [0, 0, 0.050], 0.7, JAW_OPEN),
    ("close",  None, 0.5, JAW_CLOSED),      # None = 臂保持不动, 只关夹爪
    ("lift",   [0, 0, 0.160], 1.0, JAW_CLOSED),
    ("hold",   [0, 0, 0.160], 0.4, JAW_CLOSED),
]


def run_scripted_demo(env: SO100PickEnv, collect_frames: bool = False, verbose: bool = False) -> dict:
    """执行一次脚本化抓取, 返回轨迹数据。

    返回:
        actions:  (T, 6) float32  发给执行器的关节目标
        states:   (T, 6) float32  观测到的关节角
        frames:   list[np.ndarray] | None  每步的 RGB 图 (collect_frames=True 时)
        success:  bool            结束时是否成功抬起
        success_flags: (T,) bool  每步的成功标志
        cube_z:   (T,) float32    方块高度轨迹
    """
    cube = env.cube_pos().copy()
    actions, states, frames, flags, cube_zs = [], [], [], [], []
    cur = env.arm_qpos().astype(np.float64)

    def record(tgt):
        # 模仿学习约定: 记录"施加动作之前的观测" o_t 与动作 a_t
        pre_state = env.arm_qpos().copy()
        if collect_frames:
            pre_frame = env.render_image()
        obs, r, succ, _, info = env.step(tgt)
        actions.append(tgt.copy())
        states.append(pre_state)
        flags.append(bool(succ))
        cube_zs.append(float(info["cube_pos"][2]))
        if collect_frames:
            frames.append(pre_frame)

    for name, offset, secs, jaw in DEMO_PHASES:
        n = max(1, int(secs * FPS))
        if offset is None:
            # 臂保持, 只移动夹爪
            jaw0 = cur[5]
            for i in range(n):
                tgt = cur.copy()
                tgt[5] = jaw0 + (jaw - jaw0) * (i + 1) / n
                record(tgt)
                cur = tgt
        else:
            target = cube + np.asarray(offset, dtype=float)
            for _ in range(n):
                sol = env.ik(target, q_seed=cur[:5])
                tgt = cur.copy()
                tgt[:5] = np.clip(sol, cur[:5] - MAX_JOINT_STEP, cur[:5] + MAX_JOINT_STEP)
                tgt[5] = jaw
                record(tgt)
                cur = tgt
        if verbose:
            print(f"  [{name}] TCP={env.tcp().round(4)} 方块z={env.cube_pos()[2]:.4f} "
                  f"夹爪={env.arm_qpos()[5]:.3f}")

    return {
        "actions": np.array(actions, dtype=np.float32),
        "states": np.array(states, dtype=np.float32),
        "frames": frames if collect_frames else None,
        "success": bool(env.is_success()),
        "success_flags": np.array(flags, dtype=bool),
        "cube_z": np.array(cube_zs, dtype=np.float32),
    }


# --------------------------------------------------------------------- 自测
def _self_test():
    """随机复位 5 次并执行脚本抓取, 报告成功率 + 录一次视频。"""
    import cv2

    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)

    env = SO100PickEnv(width=448, height=448)
    rng = np.random.default_rng(0)

    results = []
    frames = None
    for ep in range(5):
        obs = env.reset(rng=rng)
        demo = run_scripted_demo(env, collect_frames=(ep == 0), verbose=(ep == 0))
        results.append(demo["success"])
        if ep == 0:
            frames = demo["frames"]  # 用第 0 集录的画面做视频
        print(f"episode {ep}: 末z={demo['cube_z'][-1]:.4f} {'✅' if demo['success'] else '❌'}")

    print(f"\n脚本演示成功率: {np.mean(results)*100:.0f}% ({sum(results)}/{len(results)})")

    video = out / "task_grasp_test.mp4"
    write_video_h264(video, frames, fps=FPS)
    print(f"视频: {video} ({len(frames)} 帧)")

    picks = [0, len(frames)//4, int(len(frames)*0.55), int(len(frames)*0.8), len(frames)-1]
    for idx, name in zip(picks, ["reset", "approach", "close", "lift", "final"]):
        cv2.imwrite(str(out / f"task_{name}.png"), cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR))
    env.close()


if __name__ == "__main__":
    _self_test()
