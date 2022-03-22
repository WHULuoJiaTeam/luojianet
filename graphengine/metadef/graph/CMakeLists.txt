set(GRAPH_SOURCE_LIST
    "types.cc"
    "anchor.cc"
    "ge_attr_value.cc"
    "any_value.cc"
    "attr_store.cc"
    "attr_value.cc"
    "buffer.cc"
    "aligned_ptr.cc"
    "compute_graph.cc"
    "ascend_string.cc"
    "gnode.cc"
    "graph.cc"
    "inference_context.cc"
    "shape_refiner.cc"
    "format_refiner.cc"
    "ref_relation.cc"
    "model.cc"
    "model_serialize.cc"
    "node.cc"
    "op_desc.cc"
    "operator.cc"
    "operator_factory.cc"
    "operator_factory_impl.cc"
    "ge_attr_define.cc"
    "ge_tensor.cc"
    "detail/attributes_holder.cc"
    "utils/anchor_utils.cc"
    "utils/tuning_utils.cc"
    "utils/graph_utils.cc"
    "utils/ffts_graph_utils.cc"
    "utils/dumper/ge_graph_dumper.cc"
    "utils/ge_ir_utils.cc"
    "utils/node_utils.cc"
    "utils/op_desc_utils.cc"
    "utils/type_utils.cc"
    "utils/tensor_utils.cc"
    "utils/constant_utils.cc"
    "tensor.cc"
    "debug/graph_debug.cc"
    "opsproto/opsproto_manager.cc"
    "${METADEF_DIR}/ops/op_imp.cpp"
    "option/ge_context.cc"
    "option/ge_local_context.cc"
    "runtime_inference_context.cc"
    "${METADEF_DIR}/third_party/transformer/src/axis_util.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc"
    "${METADEF_DIR}/third_party/transformer/src/expand_dimension.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_range_according_to_format.cc"
    "utils/transformer_utils.cc"
    "utils/file_utils.cc"
    "resource_context_mgr.cc"
    "serialization/attr_serializer.cc"
    "serialization/string_serializer.cc"
    "serialization/data_type_serializer.cc"
    "serialization/named_attrs_serializer.cc"
    "serialization/bool_serializer.cc"
    "serialization/buffer_serializer.cc"
    "serialization/float_serializer.cc"
    "serialization/int_serializer.cc"
    "serialization/tensor_serializer.cc"
    "serialization/tensor_desc_serializer.cc"
    "serialization/graph_serializer.cc"
    "serialization/list_value_serializer.cc"
    "serialization/list_list_int_serializer.cc"
    "serialization/list_list_float_serializer.cc"
    "serialization/attr_serializer_registry.cc"
    "small_vector.cc"
    "operator_impl.cc"
    "compile_cache_policy/compile_cache_policy.cc"
    "compile_cache_policy/compile_cache_state.cc"
    "compile_cache_policy/compile_cache_hasher.cc"
    "compile_cache_policy/policy_management/match_policy/match_policy_exact_only.cc"
    "compile_cache_policy/policy_management/aging_policy/aging_policy_lru.cc"
    "profiler.cc"
)

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
######### libgraph.so #############
add_library(graph SHARED
    ${GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_graph_protos_obj>
)

target_compile_options(graph PRIVATE
    -O2
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
)

target_compile_definitions(graph PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../inc>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../ops/built-in/op_proto/inc>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../cann/ops/built-in/op_proto/inc>
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    ${METADEF_DIR}/../../ops/built-in/op_proto/inc
    ${METADEF_DIR}/../../cann/ops/built-in/op_proto/inc
    #### temp in ge ####
    ${METADEF_DIR}/../../graphengine/inc
    ${METADEF_DIR}/../../graphengine/inc/framework
    ${METADEF_DIR}/../../graphengine/inc/external
    ${METADEF_DIR}/../../inc
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/driver/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/fwkacllib/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/../third_party/fwkacllib/inc/ops>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/../third_party/fwkacllib/inc>
    #### blue independent compile #####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/ge/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc/external>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc/ops>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party>
    ${METADEF_DIR}/third_party/transformer/inc
)


target_link_options(graph PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(graph PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    static_mmpa
    -Wl,--no-as-needed
    ascend_protobuf
    c_sec
    slog
    error_manager
    -Wl,--as-needed
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

######### libgraph.a #############
add_library(graph_static STATIC
    ${GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_graph_protos_obj>
)

target_compile_options(graph_static PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

target_compile_definitions(graph_static PRIVATE
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:FMK_SUPPORT_DUMP>
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    LOG_CPP
)

target_include_directories(graph_static PUBLIC
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/graph
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc/external/graph
)

target_include_directories(graph_static PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/metadef_protos
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    #### yellow zone ####
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../inc>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../ops/built-in/op_proto/inc>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${METADEF_DIR}/../cann/ops/built-in/op_proto/inc>
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    ${METADEF_DIR}/../../ops/built-in/op_proto/inc
    ${METADEF_DIR}/../../cann/ops/built-in/op_proto/inc
    #### temp in ge ####
    ${METADEF_DIR}/../../graphengine/inc
    ${METADEF_DIR}/../../graphengine/inc/framework
    ${METADEF_DIR}/../../graphengine/inc/external
    ${METADEF_DIR}/../../inc
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/driver/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${ASCEND_DIR}/fwkacllib/include>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/../third_party/fwkacllib/inc/ops>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/../third_party/fwkacllib/inc>
    #### blue independent compile #####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/ge/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/graphengine/inc/external>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party/fwkacllib/inc/ops>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${METADEF_DIR}/third_party>
    ${METADEF_DIR}/third_party/transformer/inc
)

target_link_libraries(graph_static PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    c_sec
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

set_target_properties(graph_static PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS TRUE
    OUTPUT_NAME $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,libgraph,graph>
)


##############################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub_attr_value.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_graph.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_operator.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_operator_factory.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_tensor.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_inference_context.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_ascend_string.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_gnode.cc
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/stub/gen_stubapi.py ${METADEF_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
            && mv attr_value.cc stub_attr_value.cc
            && mv graph.cc stub_graph.cc
            && mv operator.cc stub_operator.cc
            && mv operator_factory.cc stub_operator_factory.cc
            && mv tensor.cc stub_tensor.cc
            && mv inference_context.cc stub_inference_context.cc
            && mv ascend_string.cc stub_ascend_string.cc
            && mv gnode.cc stub_gnode.cc
            &&  echo "Generating stub files end."
)

add_custom_target(graph_stub
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stub_attr_value.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_graph.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_operator.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_operator_factory.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_tensor.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_inference_context.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_ascend_string.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_gnode.cc
)

#############################################################


############ stub/libgraph.so ############
add_library(atc_stub_graph SHARED
    stub_graph.cc
    stub_operator.cc
    stub_operator_factory.cc
    stub_tensor.cc
    stub_attr_value.cc
    stub_ascend_string.cc
    stub_gnode.cc
)
add_dependencies(atc_stub_graph graph_stub)

target_include_directories(atc_stub_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    #### yellow zone ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../ops/built-in/op_proto/inc
    ${METADEF_DIR}/../cann/ops/built-in/op_proto/inc
    ${METADEF_DIR}/../libc_sec/include
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    ${METADEF_DIR}/../../ops/built-in/op_proto/inc
    ${METADEF_DIR}/../../cann/ops/built-in/op_proto/inc
    ${METADEF_DIR}/third_party
    ${METADEF_DIR}/third_party/transformer/inc
)

target_compile_options(atc_stub_graph PRIVATE
    -fno-common
)

target_link_libraries(atc_stub_graph PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(atc_stub_graph PROPERTIES
    OUTPUT_NAME graph
    LIBRARY_OUTPUT_DIRECTORY atc_stub
)

############ fwk_stub/libgraph.so ############
add_library(fwk_stub_graph SHARED
    stub_graph.cc
    stub_operator.cc
    stub_operator_factory.cc
    stub_tensor.cc
    stub_attr_value.cc
    stub_inference_context.cc
    stub_ascend_string.cc
    stub_gnode.cc
)

add_dependencies(fwk_stub_graph graph_stub)

target_compile_options(fwk_stub_graph PRIVATE
    -fno-common
)

target_include_directories(fwk_stub_graph PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_BINARY_DIR}
    ${METADEF_DIR}
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    #### yellow zone ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../ops/built-in/op_proto/inc
    ${METADEF_DIR}/../cann/ops/built-in/op_proto/inc
    ${METADEF_DIR}/../libc_sec/include
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    #### temp independent ####
    ${METADEF_DIR}/../graphengine/inc
    ${METADEF_DIR}/../graphengine/inc/framework
    ${METADEF_DIR}/../graphengine/inc/external
    #### temp in ge ####
    ${METADEF_DIR}/../inc
    ${METADEF_DIR}/../inc/framework
    ${METADEF_DIR}/../inc/external
    ${METADEF_DIR}/../../ops/built-in/op_proto/inc
    ${METADEF_DIR}/../../cann/ops/built-in/op_proto/inc
    ${METADEF_DIR}/third_party
    ${METADEF_DIR}/third_party/transformer/inc
)

target_link_libraries(fwk_stub_graph PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(fwk_stub_graph PROPERTIES
    OUTPUT_NAME graph
    LIBRARY_OUTPUT_DIRECTORY fwk_stub
)

else ()
######### libgraph.so w/static protobuf #############
add_library(graph SHARED
    ${GRAPH_SOURCE_LIST}
    $<TARGET_OBJECTS:metadef_graph_protos_obj>
)

target_compile_options(graph PRIVATE
        -O2
        $<$<STREQUAL:${PRODUCT_SIDE},host>:-fexceptions>
        $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
        )

target_compile_definitions(graph PRIVATE
        $<$<STREQUAL:${PRODUCT_SIDE},host>:FMK_SUPPORT_DUMP>
        $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
        google=ascend_private
        LOG_CPP
        )

target_include_directories(graph PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}
        ${CMAKE_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/proto/metadef_protos
        ${METADEF_DIR}
        ${METADEF_DIR}/inc
        ${METADEF_DIR}/inc/external
        ${METADEF_DIR}/../third_party/fwkacllib/inc/ops
        ${METADEF_DIR}/../third_party/fwkacllib/inc
        ${METADEF_DIR}/../inc
        ${METADEF_DIR}/../inc/framework
        ${METADEF_DIR}/../inc/external
        ${METADEF_DIR}/../../ops/built-in/op_proto/inc
        ${METADEF_DIR}/../../cann/ops/built-in/op_proto/inc
        ${METADEF_DIR}/third_party
        ${METADEF_DIR}/third_party/transformer/inc
        )

target_link_options(graph PRIVATE
        -Wl,-Bsymbolic
)

target_link_libraries(graph PRIVATE
        $<BUILD_INTERFACE:intf_pub>
        ascend_protobuf_static
        static_mmpa
        -Wl,--no-as-needed
        c_sec
        slog
        error_manager
        -Wl,--as-needed
        -lrt
        -ldl
        )
endif ()

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
install(TARGETS atc_stub_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub
)

install(TARGETS fwk_stub_graph OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/fwk_stub
)
endif ()
