cmake_minimum_required(VERSION 3.14)

project(CtrlZ)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(${PROJECT_SOURCE_DIR}/CtrlZ)

option(BUILD_DOC "Build documentation" OFF)

if(WIN32)
  set(ONNXRUNTIME_ROOT "C:/ProgramFiles/lib/onnxruntime" CACHE PATH "Default onnxruntime path")
elseif(UNIX)
    set(ONNXRUNTIME_ROOT "/usr/local/" CACHE PATH "Default onnxruntime path")
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

if(DEFINED VCPKG_INSTALLED_DIR)
    message("vcpkgDIR=${VCPKG_INSTALLED_DIR}")
    message("using vcpkg onnxruntime")
    set(ONNXRUNTIME_ROOT "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
endif()

add_subdirectory(CtrlZ)
add_subdirectory(doc)
add_subdirectory(example)