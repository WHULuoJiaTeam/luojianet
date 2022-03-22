set(SRC_LIST
    "register.cpp"
    "register_pass.cpp"
    "prototype_pass_registry.cc"
    "ops_kernel_builder_registry.cc"
    "graph_optimizer/graph_fusion/graph_fusion_pass_base.cc"
    "graph_optimizer/graph_fusion/large_bm.cc"
    "graph_optimizer/graph_fusion/connection_matrix.cc"
    "graph_optimizer/graph_fusion/fusion_pass_registry.cc"
    "graph_optimizer/graph_fusion/fusion_pattern.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.cc"
    "graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
    "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.cc"
    "graph_optimizer/buffer_fusion/buffer_fusion_pass_base.cc"
    "graph_optimizer/buffer_fusion/buffer_fusion_pattern.cc"
    "graph_optimizer/fusion_statistic/fusion_statistic_recorder.cc"
    "register_format_transfer.cc"
    "op_kernel_registry.cpp"
    "auto_mapping_util.cpp"
    "ffts_plus_update_manager.cc"
    "host_cpu_context.cc"
    "tensor_assign.cpp"
    "infer_data_slice_registry.cc"
    "scope/scope_graph.cc"
    "scope/scope_pass.cc"
    "scope/scope_pattern.cc"
    "scope/scope_util.cc"
    "scope/scope_pass_registry.cc"
)

############ libregister.so ############
add_library(register SHARED
    ${SRC_LIST}
    $<TARGET_OBJECTS:metadef_tensorflow_protos_obj>
)

target_compile_options(register PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
)

target_compile_definitions(register PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(register PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/..
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../inc>
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    #### temp in parser ####
    ${METADEF_DIR}/../../graphengine/inc
    ${METADEF_DIR}/../../graphengine/inc/framework
    ${METADEF_DIR}/../../graphengine/inc/external
    ${METADEF_DIR}/../../inc
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/driver/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/fwkacllib/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/../third_party/fwkacllib/inc>
    #### blue independent compile #####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/ge/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc/framework>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc/external>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc>
)

target_link_options(register PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(register PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:cce_headers>>
    -Wl,--whole-archive
    op_tiling_o2
    -Wl,--no-whole-archive
    -Wl,--no-as-needed
    ascend_protobuf
    c_sec
    slog
    graph
    -Wl,--as-needed
    json
)

############ libregister.a ############
add_library(register_static STATIC
    ${SRC_LIST}
    $<TARGET_OBJECTS:metadef_tensorflow_protos_obj>
    "op_tiling/op_tiling.cc"
    "op_tiling/op_tiling_info.cc"
    "op_tiling/op_tiling_utils.cc"
    "op_tiling/op_tiling_registry.cc"
    "op_tiling/op_tiling_py.cc"
)

target_compile_options(register_static PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:/utf-8>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(register_static PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    LOG_CPP
)

target_include_directories(register_static PUBLIC
    ${METADEF_DIR}/graph
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/register
)

target_include_directories(register_static PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/..
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../inc>
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    #### temp in parser ####
    ${METADEF_DIR}/../../graphengine/inc
    ${METADEF_DIR}/../../graphengine/inc/framework
    ${METADEF_DIR}/../../graphengine/inc/external
    ${METADEF_DIR}/../../inc
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/driver/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/fwkacllib/include>
    #### blue independent compile #####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/../third_party/fwkacllib/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc/framework>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc/external>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc>
)

target_link_libraries(register_static PRIVATE
    c_sec
    json
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:cce_headers>>
)

set_target_properties(register_static PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,libregister,register>
)

############ libop_tiling_o2.a ############
add_library(op_tiling_o2 STATIC
    "op_tiling/op_tiling.cc"
    "op_tiling/op_tiling_info.cc"
    "op_tiling/op_tiling_utils.cc"
    "op_tiling/op_tiling_registry.cc"
    "op_tiling/op_tiling_py.cc"
)

add_dependencies(op_tiling_o2
    metadef_protos
)

target_include_directories(op_tiling_o2 PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/..
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../inc>
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    #### temp in parser ####
    ${METADEF_DIR}/../../graphengine/inc
    ${METADEF_DIR}/../../graphengine/inc/framework
    ${METADEF_DIR}/../../graphengine/inc/external
    ${METADEF_DIR}/../../inc
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/driver/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/fwkacllib/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/../third_party/fwkacllib/inc>
    #### blue independent compile ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc/external>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc>
)

target_compile_options(op_tiling_o2 PRIVATE
    -O2
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
)

target_compile_definitions(op_tiling_o2 PRIVATE
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    LOG_CPP
)

target_link_libraries(op_tiling_o2 PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    ascend_protobuf
    json
    c_sec
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS register register_static OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
    ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR}
)
