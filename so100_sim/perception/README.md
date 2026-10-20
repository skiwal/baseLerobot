# 第一步：图像感知（套筒 / 杆检测）

纯 **C** 实现的感知算法（CMake 管理），仅用 Python 做 MuJoCo 仿真桥接与图像 I/O。
算法以**机械臂根部**为基准（相机外参固定、与 `so100_sleeve_scene.xml` 的 `base_rgbd` 一致），
从 RGB-D 中检测出套筒与杆，并输出**高亮效果图**。

## 算法流程（perceive.c）
1. **背景差分**：`|depth - bg_depth| > 阈值` 得到动态点（套筒 + 杆）。
   - 背景深度由仿真渲染“无目标”场景得到；机械臂在两次渲染中不动，自动抵消。
2. **连通域分割**（4 邻域 BFS）。
3. **PCA 主轴**：对每个连通域的世界点云计算主轴 → 即物体轴线。
4. **分类**：轴线最竖直 → 套筒；最倾斜 → 杆。
5. **几何拟合 + 标注**：输出 6D 位姿（位置/轴线/杆尖），在图上高亮
   （套筒绿、杆红、画轴线/边界框/杆尖圆点）。

> 这是传统几何/滤波算法，**无训练、无神经网络、CPU 即可实时**（验证阶段在仿真内，后续接真实相机同布局即可迁移）。

## 构建
```bash
cd so100_sim/perception
mkdir -p build && cd build
cmake .. && cmake --build .      # 生成 build/perceive
```

## 运行（需 lerobot 环境的 mujoco / numpy / Pillow）
```bash
cd so100_sim
MUJOCO_GL=egl python perception/python/sim_bridge.py
```
- 自动渲染根部相机 rgb/depth/bg → 写 raw → 调 `build/perceive` → 输出
  `so100_sim/out/perception_result.png`（高亮效果图）与 `perception_raw.png`（原图对比）。
- 终端打印检测到的位姿，例如：
  ```
  SLEEVE pos=(0.120,-0.235,0.110) axis=(0.01,0.02,1.00)
  ROD    base=(...) tip=(...) axis=(...)
  ```

## 输入/输出文件格式
- `rgb.bin`   : `W*H*3` uint8（行主序，RGB）
- `depth.bin` / `bg.bin` : `W*H` uint16（毫米，0 = 无效）
- `meta.txt`  : 一行 `W H`
- `result.bin` : `W*H*3` uint8（高亮后 RGB）

## 后续（ROS2 集成，环境已就绪）
`perceive.c` 的算法是 ROS 无关的。可把同样的 `project()/unproject()/pca_axis()/分类` 封装成
ROS2 C++ 节点：`sensor_msgs/Image`(+`depth`) → 感知 → 发布检测结果（标注图 / 位姿 `PoseStamped`）。
CMakeLists 里已留 `perceive_node` 的注释位，接 rclcpp 即可。
