cmake_minimum_required(VERSION 3.14)

project(CtrlZ)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${PROJECT_SOURCE_DIR}/CtrlZ)

option(BUILD_DOC "Build documentation" OFF)

if(WIN32)
  set(ONNXRUNTIME_ROOT "C:/ProgramFiles/lib/onnxruntime")
elseif(UNIX)
    set(ONNXRUNTIME_ROOT "/usr/local/")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

add_subdirectory(CtrlZ)
add_subdirectory(doc)
add_subdirectory(example)