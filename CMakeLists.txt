cmake_minimum_required(VERSION 3.8)
project(prob_lab)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(kf_node src/kf_node.cpp)
ament_target_dependencies(kf_node rclcpp geometry_msgs nav_msgs std_msgs tf2 tf2_geometry_msgs)
target_link_libraries(kf_node Eigen3::Eigen)

add_executable(ekf_node src/ekf_node.cpp)
ament_target_dependencies(ekf_node rclcpp geometry_msgs nav_msgs std_msgs tf2 tf2_geometry_msgs)
target_link_libraries(ekf_node Eigen3::Eigen)

add_executable(pf_node src/pf_node.cpp)
ament_target_dependencies(pf_node rclcpp geometry_msgs nav_msgs sensor_msgs std_msgs tf2 tf2_ros tf2_geometry_msgs)
target_link_libraries(pf_node Eigen3::Eigen)

add_executable(trajectory_node src/trajectory_node.cpp)
ament_target_dependencies(trajectory_node rclcpp geometry_msgs nav_msgs std_msgs tf2 tf2_geometry_msgs)

add_executable(path_publisher_node src/path_publisher_node.cpp)
ament_target_dependencies(path_publisher_node rclcpp geometry_msgs nav_msgs std_msgs)

add_executable(evaluation_node src/evaluation_node.cpp)
ament_target_dependencies(evaluation_node rclcpp geometry_msgs nav_msgs std_msgs)

install(TARGETS
  kf_node ekf_node pf_node
  trajectory_node path_publisher_node evaluation_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config rviz maps
  DESTINATION share/${PROJECT_NAME})

ament_package()
