set(SRC_LIST
    "dnnengines.cc"
    "engine_manage.cc"
)

############ libengine.so ############
add_library(engine SHARED ${SRC_LIST})

add_dependencies(engine
    graphengine_protos
)

target_compile_options(engine PRIVATE
    -Werror
    -fno-common
    -fvisibility=hidden
)

target_compile_definitions(engine PRIVATE
    REUSE_MEMORY=1
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    FUNC_VISIBILITY
)

target_include_directories(engine PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/framework
    ${GE_CODE_DIR}/inc/external
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
)

target_link_options(engine PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(engine PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--no-as-needed
    c_sec
    slog
    -Wl,--as-needed
    -lrt
    -ldl
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS engine OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
