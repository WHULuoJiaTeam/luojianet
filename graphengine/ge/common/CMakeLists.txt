set(SRC_LIST
    "${GE_CODE_DIR}/ge/common/auth/file_saver.cc"
    "${GE_CODE_DIR}/ge/common/bcast.cc"
    "${GE_CODE_DIR}/ge/common/context/ctx.cc"
    "${GE_CODE_DIR}/ge/common/cust_aicpu_kernel_store.cc"
    "${GE_CODE_DIR}/ge/common/debug/memory_dumper.cc"
    "${GE_CODE_DIR}/ge/common/dump/dump_manager.cc"
    "${GE_CODE_DIR}/ge/common/dump/dump_properties.cc"
    "${GE_CODE_DIR}/ge/common/fmk_error_codes.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/datatype_transfer.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_c1hwncoc0_hwcn.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_dhwcn_fracz3D.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_dhwnc_fracz3D_transpose.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fractal_nz.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fractal_z.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fractal_zz.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fracz_hwcn.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fracz_nchw.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_fracz_nhwc.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_hwcn_c1hwncoc0.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nc1hwc0_nchw.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nc1hwc0_nhwc.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nchw_fz_c04.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nchw_nc1hwc0.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_nhwc_nc1hwc0.cc"
    "${GE_CODE_DIR}/ge/common/formats/format_transfers/format_transfer_transpose.cc"
    "${GE_CODE_DIR}/ge/common/formats/formats.cc"
    "${GE_CODE_DIR}/ge/common/formats/utils/formats_trans_utils.cc"
    "${GE_CODE_DIR}/ge/common/fp16_t.cc"
    "${GE_CODE_DIR}/ge/common/ge/datatype_util.cc"
    "${GE_CODE_DIR}/ge/common/ge/op_tiling_manager.cc"
    "${GE_CODE_DIR}/ge/common/ge/plugin_manager.cc"
    "${GE_CODE_DIR}/ge/common/ge/tbe_plugin_manager.cc"
    "${GE_CODE_DIR}/ge/common/ge_format_util.cc"
    "${GE_CODE_DIR}/ge/common/helper/model_helper.cc"
    "${GE_CODE_DIR}/ge/common/helper/om_file_helper.cc"
    "${GE_CODE_DIR}/ge/common/kernel_store.cc"
    "${GE_CODE_DIR}/ge/common/local_context.cc"
    "${GE_CODE_DIR}/ge/common/math/fp16_math.cc"
    "${GE_CODE_DIR}/ge/common/model/ge_model.cc"
    "${GE_CODE_DIR}/ge/common/model/ge_root_model.cc"
    "${GE_CODE_DIR}/ge/common/model_parser/model_parser.cc"
    "${GE_CODE_DIR}/ge/common/model_saver.cc"
    "${GE_CODE_DIR}/ge/common/omg_util.cc"
    "${GE_CODE_DIR}/ge/common/op/attr_value_util.cc"
    "${GE_CODE_DIR}/ge/common/op/ge_op_utils.cc"
    "${GE_CODE_DIR}/ge/common/properties_manager.cc"
    "${GE_CODE_DIR}/ge/common/tbe_kernel_store.cc"
    "${GE_CODE_DIR}/ge/common/thread_pool.cc"
    "${GE_CODE_DIR}/ge/common/transop_util.cc"
    "${GE_CODE_DIR}/ge/common/types.cc"
    "${GE_CODE_DIR}/ge/common/util.cc"
)

if (NOT ENABLE_D AND NOT ENABLE_ACL)
############ libge_common.so ############
add_library(ge_common SHARED ${SRC_LIST})

add_dependencies(ge_common
    graphengine_protos
)

target_compile_definitions(ge_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    FUNC_VISIBILITY
)

target_compile_options(ge_common PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wno-deprecated-declarations
    -fno-common
)

target_include_directories(ge_common PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${GE_DEPEND_DIR}/inc>
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain>
)

target_link_options(ge_common PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    static_mmpa
    -Wl,--no-as-needed
    graph
    ascend_protobuf
    register
    c_sec
    error_manager
    slog
    -Wl,--as-needed
    json
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

############ libge_common.a ############
add_library(ge_common_static STATIC ${SRC_LIST})

add_dependencies(ge_common_static
    graphengine_protos
)

target_compile_definitions(ge_common_static PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    LOG_CPP
    FUNC_VISIBILITY
)

target_compile_options(ge_common_static PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-fvisibility=hidden -O2 -Werror -Wno-deprecated-declarations -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_include_directories(ge_common_static PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${GE_DEPEND_DIR}/inc>
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain>
)

target_link_libraries(ge_common_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    ascend_protobuf_static
    json
    c_sec
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

else ()
############ libge_common.so w/static protobuf ############
add_library(ge_common SHARED ${SRC_LIST})

add_dependencies(ge_common
    graphengine_protos
)

target_compile_definitions(ge_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    HOST_VISIBILITY
    FMK_SUPPORT_DUMP
    OS_CENTOS
    google=ascend_private
    LOG_CPP
    FUNC_VISIBILITY
)

target_compile_options(ge_common PRIVATE
    -fvisibility=default
    -O2
    -Werror
    -Wno-deprecated-declarations
    -fno-common
)

target_include_directories(ge_common PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${GE_CODE_DIR}/third_party/fwkacllib/inc
    ${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain
)

target_link_options(ge_common PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf_static
    -Wl,--no-as-needed
    graph
    register
    c_sec
    error_manager
    slog
    static_mmpa
    -Wl,--as-needed
    json
    -lrt
    -ldl
)
endif ()

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS ge_common OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
