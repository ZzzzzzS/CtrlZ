project(${CMAKE_PROJECT_NAME}_example)

option(BUILD_EXAMPLE_BITBOT_SIM "Build bitbot simulation example" OFF)
option(BUILD_EXAMPLE_UNITREE_DANCE "Build bitbot unitree dance example" OFF)

if(BUILD_EXAMPLE_BITBOT_SIM)
    add_subdirectory(BitbotSimulation)
endif()

if(BUILD_EXAMPLE_UNITREE_DANCE)
    add_subdirectory(BitbotUnitreeDancing)
endif()

