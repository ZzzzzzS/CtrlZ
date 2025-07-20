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
#aux_source_directory(Schedulers SRC_SCHEDULERS)
#aux_source_directory(Workers SRC_WORKERS)
#aux_source_directory(Utils SRC_UTILS)

file(GLOB_RECURSE SRC_SCHEDULERS Schedulers/*.cpp Schedulers/*.h)
file(GLOB_RECURSE SRC_WORKERS Workers/*.cpp Workers/*.h)
file(GLOB_RECURSE SRC_UTILS Utils/*.cpp Utils/*.h)


message("
Ctrl-Z is a muti-thread RL deployment framework for Robot locomotion.")
message("Visit http://opensource.zzshub.cn/CtrlZ for detailed documentation.")

message("+----------------------------------------+
|  CCCC  TTTTT  RRRRR  L           ZZZZZ |
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
)

add_subdirectory(Extensions)

target_include_directories(CtrlZ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
    message("Build on windows")
    target_include_directories(CtrlZ PUBLIC ${ONNXRUNTIME_ROOT}/include)
    message("${ONNXRUNTIME_ROOT}/include")
    target_link_libraries(CtrlZ PUBLIC
        ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib
        ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
elseif(UNIX)
    message("Build on unix like system")
    target_include_directories(CtrlZ PUBLIC ${ONNXRUNTIME_ROOT}/include)
    message("${ONNXRUNTIME_ROOT}/include")
    target_link_libraries(CtrlZ PUBLIC
        ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_shared.so
        ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
else()
    message(FATAL_ERROR "Unsupported platform")
endif()

#set(READERWRITERQUEUE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/ThirdParty)
#target_include_directories(CtrlZ PUBLIC ${READERWRITERQUEUE_INCLUDE_DIRS})
target_link_libraries(CtrlZ PUBLIC
    nlohmann_json::nlohmann_json
)
