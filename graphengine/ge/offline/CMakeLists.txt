set(SRC_LIST
    "main.cc"
    "single_op_parser.cc"
    "../session/omg.cc"
    "../ir_build/option_utils.cc"
)

############ atc_atc.bin ############
add_executable(atc_atc.bin ${SRC_LIST})

add_dependencies(atc_atc.bin
    graphengine_protos
)

target_compile_options(atc_atc.bin PRIVATE
    -Werror
    -O2
    -Wno-deprecated-declarations
    -fno-common
    -fvisibility=hidden
)

target_compile_definitions(atc_atc.bin PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
)

target_include_directories(atc_atc.bin PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${PARSER_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
    ${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain
)

target_link_options(atc_atc.bin PRIVATE
    -Wl,-Bsymbolic
    -Wl,-rpath-link,${ASCEND_ATC_DIR}/stub
)

target_link_libraries(atc_atc.bin PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
    ge_common
    register
    c_sec
    graph
    error_manager
    ge_compiler
    parser_common
    gflags
    json
    runtime
    slog
    static_mmpa
    -lrt
    -ldl
)

set_target_properties(atc_atc.bin PROPERTIES
    OUTPUT_NAME atc.bin
    RUNTIME_OUTPUT_DIRECTORY atclib
)

############ fwk_atc.bin ############
add_executable(fwk_atc.bin ${SRC_LIST})

add_dependencies(fwk_atc.bin
    graphengine_protos
)

target_compile_options(fwk_atc.bin PRIVATE
    -Werror
    -O2
    -Wno-deprecated-declarations
    -fno-common
    -fvisibility=hidden
)

target_compile_definitions(fwk_atc.bin PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
)

target_include_directories(fwk_atc.bin PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${PARSER_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    #### blue zone ####
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
    ${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain
)

target_link_options(fwk_atc.bin PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(fwk_atc.bin PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
    ge_common
    register
    c_sec
    graph
    error_manager
    ge_runner
    parser_common
    gflags
    json
    runtime
    slog
    static_mmpa
    -lrt
    -ldl
)

set_target_properties(fwk_atc.bin PROPERTIES
    OUTPUT_NAME atc.bin
    RUNTIME_OUTPUT_DIRECTORY fwkacl
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS atc_atc.bin OPTIONAL
    RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}/atclib
)

install(TARGETS fwk_atc.bin OPTIONAL
    RUNTIME DESTINATION ${INSTALL_LIBRARY_DIR}/fwkacl
)
