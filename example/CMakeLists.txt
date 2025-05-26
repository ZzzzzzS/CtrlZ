project(${CMAKE_PROJECT_NAME}_example)

option(BUILD_EXAMPLE_BITBOT_SIM "Build bitbot simulation example" OFF)

if(BUILD_EXAMPLE_BITBOT_SIM)
    add_subdirectory(BitbotSimulation)
endif()
