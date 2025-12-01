cmake_minimum_required(VERSION 3.14)

project(BitbotUnitreeDancing)

set(BITBOT_DEPENDENCY_USE_PROXY OFF)
set(BITBOT_DEPENDENCY_USE_LOCAL_FILE TRUE)
set(BITBOT_DEPENDENCY_LOCAL_FILE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/dependencies)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(bitbot.cmake)


BitbotAddDependency(
  NAME bitbot_mujoco
  FILE_NAME "bitbot_mujoco-main.zip"
  FILE_PATH ${BITBOT_DEPENDENCY_LOCAL_FILE_PATH}
  USE_LOCAL_FILE TRUE
)

BitbotAddDependency(
  NAME bitbot_kernel
  FILE_NAME "bitbot_kernel-main.zip"
  FILE_PATH ${BITBOT_DEPENDENCY_LOCAL_FILE_PATH}
  USE_LOCAL_FILE TRUE
)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")
message("PROJECT_ROOT_DIR: ${PROJECT_ROOT_DIR}")

add_executable(dance_app
  main.cpp
  user_func.h
  user_func.cpp
  types.hpp
)


add_definitions(-DMUJOCO_SIM)
target_link_libraries(dance_app BitbotMujoco)

target_link_libraries(dance_app
  CtrlZ
)

target_compile_options(dance_app PUBLIC)
target_compile_features(dance_app PUBLIC cxx_std_20)
