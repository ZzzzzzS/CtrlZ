file(GLOB _subdirs LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(dir ${_subdirs})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir} AND
        EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/CMakeLists.txt)
        message(STATUS "Adding Extension Package: ${dir}")
        add_subdirectory(${dir})
        if(BUILD_${dir}_EXTENSION)
            set(lib_name ext_${dir})
            target_link_libraries(CtrlZ INTERFACE ${lib_name})
        endif()
    endif()
endforeach()
