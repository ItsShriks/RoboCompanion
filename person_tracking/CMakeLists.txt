cmake_minimum_required(VERSION 3.0.2)
project(person_tracking)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs std_msgs cv_bridge
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/person_tracker_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)