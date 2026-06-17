cmake_minimum_required(VERSION 3.8)
project(graph_slam)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters registration)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  /usr/local/include
)

link_directories(
  /usr/local/lib
)

set(AMENT_DEPS
  rclcpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
)

set(G2O_LIBS
  g2o_core
  g2o_stuff
  g2o_solver_eigen
  g2o_types_slam2d
  g2o_types_slam3d
  g2o_csparse_extension
)

# Generate headers from your .srv files
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/AddEdge.srv"
  "srv/AddNode.srv"
  "srv/GetGraph.srv"
  "srv/ScanMatch.srv"
  "srv/UpdatePoses.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

macro(slam_node name src)
  add_executable(${name} ${src})
  ament_target_dependencies(${name} ${AMENT_DEPS})
  target_link_libraries(${name}
    ${PCL_LIBRARIES}
    fmt::fmt
  )
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(${name} "${cpp_typesupport_target}")
endmacro()

slam_node(graph_manager    src/graph_manager.cpp)
slam_node(pipeline         src/pipeline.cpp)
slam_node(scan_matcher     src/scan_matcher.cpp)
slam_node(loop_closure     src/loop_closure.cpp)
slam_node(occupancy_mapper src/occupancy_mapper.cpp)
slam_node(graph_optimizer  src/graph_optimizer.cpp)


target_link_libraries(graph_optimizer
  ${G2O_LIBS}
  pthread
)


target_link_libraries(loop_closure
  ${G2O_LIBS}
  pthread
)

install(TARGETS
  graph_manager
  pipeline
  scan_matcher
  loop_closure
  graph_optimizer
  occupancy_mapper
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/  DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/  DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY rviz/    DESTINATION share/${PROJECT_NAME}/rviz)
install(DIRECTORY worlds/  DESTINATION share/${PROJECT_NAME}/worlds)
install(DIRECTORY urdf/    DESTINATION share/${PROJECT_NAME}/urdf)
install(DIRECTORY include/ DESTINATION include)

ament_package()