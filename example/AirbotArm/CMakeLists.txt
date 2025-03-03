project(AirbotArmControl VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
elseif(WIN32)
message(FATAL_ERROR "Unsupported platform")
endif()


find_package(airbot_play REQUIRED)

set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-DPROJECT_ROOT_DIR="${PROJECT_ROOT_DIR}")
message("PROJECT_ROOT_DIR: ${PROJECT_ROOT_DIR}")

set(SOURCES
    main.cpp
    ArmTrackingInferenceWorker.hpp
    types.hpp
    )

add_executable(AirbotArmControl ${SOURCES})
target_link_libraries(AirbotArmControl 
CtrlZ
airbot_play
)



