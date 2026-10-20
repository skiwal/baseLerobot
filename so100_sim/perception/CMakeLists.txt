cmake_minimum_required(VERSION 3.10)
project(so100_perception C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# 纯 C 感知核心, 无第三方依赖
add_executable(perceive src/perceive.c)
target_link_libraries(perceive m)   # libm (sqrt 等)

# 可选: 后续包成 ROS2 节点时取消注释(需 ROS2 环境)
# find_package(rclcpp REQUIRED)
# add_executable(perceive_node src/perceive_node.cpp)
# target_link_libraries(perceive_node rclcpp)
