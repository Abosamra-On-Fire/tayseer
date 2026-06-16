cmake_minimum_required(VERSION 3.1)
project(graph_slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ── Find dependencies ─────────────────────────────────────────────────────────

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  message_generation
  pcl_ros
  pcl_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL   REQUIRED COMPONENTS common io filters registration)

# ── g2o ───────────────────────────────────────────────────────────────────────
find_package(g2o REQUIRED)
if(NOT g2o_FOUND)
  message(WARNING "g2o cmake config not found, trying manual paths")
  find_library(G2O_CORE_LIB         g2o_core)
  find_library(G2O_SOLVER_EIGEN_LIB g2o_solver_eigen)
  find_library(G2O_TYPES_SLAM2D_LIB g2o_types_slam2d)
  set(G2O_LIBRARIES
      ${G2O_CORE_LIB}
      ${G2O_SOLVER_EIGEN_LIB}
      ${G2O_TYPES_SLAM2D_LIB})
else()
  set(G2O_LIBRARIES
      g2o::core
      g2o::solver_eigen
      g2o::types_slam2d)
endif()

# ── Messages / services ───────────────────────────────────────────────────────

add_service_files(
  FILES
    AddNode.srv
    AddEdge.srv
    GetGraph.srv
    UpdatePoses.srv
    ScanMatch.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
    sensor_msgs
    geometry_msgs
)

# ── catkin package ────────────────────────────────────────────────────────────

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    roscpp
    std_msgs
    nav_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    message_runtime
  DEPENDS
    EIGEN3
    PCL
)

# ── Include paths ─────────────────────────────────────────────────────────────

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}/unsupported   # Eigen FFT for loop_closure
  ${PCL_INCLUDE_DIRS}
)

find_path(G2O_INCLUDE_DIR g2o/core/sparse_optimizer.h
  PATHS /usr/include /usr/local/include)
if(G2O_INCLUDE_DIR)
  include_directories(${G2O_INCLUDE_DIR})
endif()

# ── Helper macro ──────────────────────────────────────────────────────────────
# RobustKernel symbols live inside g2o_core on Noetic — no separate lib needed.

macro(slam_node name src)
  add_executable(${name} ${src})
  add_dependencies(${name}
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
    ${catkin_EXPORTED_TARGETS})
  target_link_libraries(${name}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    ${ARGN})
endmacro()

# ── Executables ───────────────────────────────────────────────────────────────

slam_node(graph_manager    src/graph_manager.cpp)
slam_node(pipeline         src/pipeline.cpp)
slam_node(scan_matcher     src/scan_matcher.cpp)
slam_node(loop_closure     src/loop_closure.cpp)
slam_node(occupancy_mapper src/occupancy_mapper.cpp)
slam_node(graph_optimizer  src/graph_optimizer.cpp
          ${G2O_LIBRARIES})

# ── Install ───────────────────────────────────────────────────────────────────

install(TARGETS
    graph_manager pipeline scan_matcher
    loop_closure graph_optimizer occupancy_mapper
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
install(DIRECTORY config/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)
install(DIRECTORY rviz/   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz)
install(DIRECTORY worlds/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds)
install(DIRECTORY urdf/   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf)
install(DIRECTORY include/ DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/include)
install(TARGETS
    graph_manager pipeline scan_matcher
    loop_closure graph_optimizer occupancy_mapper
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})