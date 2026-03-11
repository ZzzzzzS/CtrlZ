cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(CtrlZ)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(UNIX)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty)

find_package(nlohmann_json REQUIRED)

# OpenVINO 选项
option(USE_OPENVINO "Use OpenVINO instead of ONNXRuntime for inference" OFF)
set(OPENVINO_DEVICE "CPU" CACHE STRING "OpenVINO inference device (CPU, GPU, NPU, AUTO, etc.)")

if(USE_OPENVINO)
    message(STATUS "Using OpenVINO for neural network inference")
    message(STATUS "OpenVINO default device: ${OPENVINO_DEVICE}")
    find_package(OpenVINO REQUIRED)
else()
    if(UNIX)
        find_package(onnxruntime REQUIRED)
    endif()
    message(STATUS "Using ONNXRuntime for neural network inference")
endif()

#aux_source_directory(Schedulers SRC_SCHEDULERS)
#aux_source_directory(Workers SRC_WORKERS)
#aux_source_directory(Utils SRC_UTILS)

file(GLOB_RECURSE SRC_SCHEDULERS Schedulers/*.cpp Schedulers/*.h)
file(GLOB_RECURSE SRC_WORKERS Workers/*.cpp Workers/*.h)
file(GLOB_RECURSE SRC_UTILS Utils/*.cpp Utils/*.h)

# tinyxml2 源文件
set(TINYXML2_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/include/tinyxml2/tinyxml2.cpp)


message("
Ctrl-Z is a muti-thread RL deployment framework for Robot locomotion.")
message("Visit http://opensource.zzshub.cn/CtrlZ for detailed documentation.")

message("+----------------------------------------+
|  CCCC  TTTTT  RRRRR  R           ZZZZZ |
| C        T    R   R  L              Z  |
| C        T    RRRRR  L    =====    Z   |
| C        T    R R    L            Z    |
|  CCCC    T    R  RR  LLLLL       ZZZZZ |
+----------------------------------------+
")

# include directory for all target
add_library(CtrlZ STATIC
    ${SRC_SCHEDULERS}
    ${SRC_WORKERS}
    ${SRC_UTILS}
    ${TINYXML2_SRC}
)

add_subdirectory(Extensions)

target_include_directories(CtrlZ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/include ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty/include/tinyxml2)

# 通过目标属性暴露宏定义给依赖项
if(USE_OPENVINO)
    target_compile_definitions(CtrlZ PUBLIC -DUSE_OPENVINO)
    # 可选：启用 GPU 插件特定功能
    option(OPENVINO_ENABLE_GPU "Enable OpenVINO GPU plugin specific features" OFF)
    if(OPENVINO_ENABLE_GPU)
        target_compile_definitions(CtrlZ PUBLIC -DOPENVINO_ENABLE_GPU)
    endif()
endif()

if(USE_OPENVINO)
    # OpenVINO 配置
    target_link_libraries(CtrlZ PUBLIC
        openvino::runtime
    )
else()
    # ONNXRuntime 配置
    if(WIN32)
        message("Build on windows")
        target_include_directories(CtrlZ PUBLIC ${ONNXRUNTIME_ROOT}/include)
        message("${ONNXRUNTIME_ROOT}/include")
        target_link_libraries(CtrlZ PUBLIC
            ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib
            ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
    elseif(UNIX)
        message("Build on unix like system")
        target_link_libraries(CtrlZ PUBLIC onnxruntime::onnxruntime)
    else()
        message(FATAL_ERROR "Unsupported platform")
    endif()
endif()

#set(READERWRITERQUEUE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty)
#target_include_directories(CtrlZ PUBLIC ${READERWRITERQUEUE_INCLUDE_DIRS})
target_link_libraries(CtrlZ PUBLIC
    nlohmann_json::nlohmann_json
)
