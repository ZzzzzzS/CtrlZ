option(BUILD_ROS2_EXTENSION "Build ROS2 extension" OFF)

if(BUILD_ROS2_EXTENSION)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    message(STATUS "Building ROS2 extension for CtrlZ")



    add_library(ext_ROS2 STATIC
        ./RobotStatusPubWorker.hpp
        ./Ros2Launcher.cpp
        ./Ros2Launcher.h
        ./RobotStatusPubNode.hpp
        ./Ros2CmdWorker.hpp
    )

    target_compile_definitions(ext_ROS2 PUBLIC
        BUILD_ROS2_EXT
    )

    target_include_directories(ext_ROS2 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR})

    ament_target_dependencies(ext_ROS2
        rclcpp
        std_msgs
        sensor_msgs
        geometry_msgs
    )
else()
    message(STATUS "ROS2 extension is disabled. Skipping ROS2 related configurations.")
endif()
