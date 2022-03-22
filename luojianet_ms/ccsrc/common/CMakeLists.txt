if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(GLOB_RECURSE _COMMON_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "trans.cc"
        "utils.cc"
        "duplex_pipe_win.cc"
        "thread_pool.cc"
        )
else()
    file(GLOB_RECURSE _COMMON_ALL_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "trans.cc"
        "utils.cc"
        "duplex_pipe.cc"
        "thread_pool.cc"
        )
endif()

set_property(SOURCE ${_COMMON_ALL_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=luojianet_ms::SubModuleId::SM_COMMON)
add_library(_luojianet_common_obj OBJECT ${_COMMON_ALL_SRC_FILES})
