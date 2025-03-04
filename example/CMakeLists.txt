project(${CMAKE_PROJECT_NAME}_example)

option(BUILD_EXAMPLE_BITBOT_SIM "Build bitbot simulation example" OFF)
option(BUILD_EXAMPLE_AIRBOT_ARM "Build airbot arm control example" OFF)

if(BUILD_EXAMPLE_BITBOT_SIM)
    add_subdirectory(BitbotSimulation)
endif()

if(BUILD_EXAMPLE_AIRBOT_ARM)
    add_subdirectory(AirbotArm)
endif()