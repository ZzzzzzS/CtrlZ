cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(CtrlZ)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)

#aux_source_directory(Schedulers SRC_SCHEDULERS)
#aux_source_directory(Workers SRC_WORKERS)
#aux_source_directory(Utils SRC_UTILS)

file(GLOB_RECURSE SRC_SCHEDULERS Schedulers/*.cpp Schedulers/*.h)
file(GLOB_RECURSE SRC_WORKERS Workers/*.cpp Workers/*.h)
file(GLOB_RECURSE SRC_UTILS Utils/*.cpp Utils/*.h)

# include directory for all target
add_library(CtrlZ STATIC
    ./types.hpp
    ${SRC_SCHEDULERS}
    ${SRC_WORKERS}
    ${SRC_UTILS}
)
target_include_directories(CtrlZ PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(WIN32)
message("Build on windows")
 target_include_directories(CtrlZ PUBLIC ${ONNXRUNTIME_ROOT}/include)
 message("${ONNXRUNTIME_ROOT}/include")
 if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Build in debug mode")
  target_link_libraries(CtrlZ PUBLIC 
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_tensorrt.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
 else()
  message("Build in release mode")
  target_link_libraries(CtrlZ PUBLIC 
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_tensorrt.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib)
 endif()

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


target_link_libraries(CtrlZ PUBLIC
nlohmann_json::nlohmann_json
readerwriterqueue
Eigen3::Eigen
)