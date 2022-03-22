set(SRC_LIST
    "ge_executor.cc"
    "../common/profiling/profiling_manager.cc"
    "../common/dump/dump_op.cc"
    "../common/dump/opdebug_register.cc"
    "../common/dump/exception_dumper.cc"
    "../common/profiling/ge_profiling.cc"
    "../graph/load/graph_loader.cc"
    "../graph/execute/graph_execute.cc"
    "../graph/manager/graph_manager_utils.cc"
    "../graph/manager/graph_var_manager.cc"
    "../graph/manager/graph_mem_allocator.cc"
    "../graph/manager/graph_caching_allocator.cc"
    "../graph/manager/session_scope_mem_allocator.cc"
    "../graph/manager/graph_mem_manager.cc"
    "../graph/manager/trans_var_data_utils.cc"
    "../graph/manager/util/debug.cc"
    "../graph/manager/rdma_pool_allocator.cc"
    "../graph/manager/host_mem_allocator.cc"
    "../hybrid/node_executor/aicpu/aicpu_ext_info.cc"
    "../graph/load/model_manager/davinci_model.cc"
    "../graph/load/model_manager/model_manager.cc"
    "../graph/load/model_manager/tbe_handle_store.cc"
    "../graph/load/model_manager/cpu_queue_schedule.cc"
    "../graph/load/model_manager/model_utils.cc"
    "../graph/load/model_manager/aipp_utils.cc"
    "../graph/load/model_manager/data_inputer.cc"
    "../graph/load/model_manager/data_dumper.cc"
    "../graph/load/model_manager/zero_copy_task.cc"
    "../graph/load/model_manager/zero_copy_offset.cc"
    "../graph/load/model_manager/task_info/task_info.cc"
    "../graph/load/model_manager/task_info/event_record_task_info.cc"
    "../graph/load/model_manager/task_info/event_wait_task_info.cc"
    "../graph/load/model_manager/task_info/ffts_task_info.cc"
    "../graph/load/model_manager/task_info/fusion_start_task_info.cc"
    "../graph/load/model_manager/task_info/fusion_stop_task_info.cc"
    "../graph/load/model_manager/task_info/kernel_ex_task_info.cc"
    "../graph/load/model_manager/task_info/kernel_task_info.cc"
    "../graph/load/model_manager/task_info/label_set_task_info.cc"
    "../graph/load/model_manager/task_info/label_switch_by_index_task_info.cc"
    "../graph/load/model_manager/task_info/label_goto_ex_task_info.cc"
    "../graph/load/model_manager/task_info/memcpy_async_task_info.cc"
    "../graph/load/model_manager/task_info/memcpy_addr_async_task_info.cc"
    "../graph/load/model_manager/task_info/profiler_trace_task_info.cc"
    "../graph/load/model_manager/task_info/stream_active_task_info.cc"
    "../graph/load/model_manager/task_info/stream_switch_task_info.cc"
    "../graph/load/model_manager/task_info/stream_switchn_task_info.cc"
    "../graph/load/model_manager/task_info/end_graph_task_info.cc"
    "../graph/load/model_manager/task_info/model_exit_task_info.cc"
    "../graph/load/model_manager/task_info/super_kernel/super_kernel_factory.cc"
    "../graph/load/model_manager/task_info/super_kernel/super_kernel.cc"
    "../opskernel_manager/ops_kernel_builder_manager.cc"
    "../single_op/single_op_manager.cc"
    "../single_op/single_op_model.cc"
    "../single_op/single_op.cc"
    "../single_op/stream_resource.cc"
    "../single_op/task/op_task.cc"
    "../single_op/task/build_task_utils.cc"
    "../single_op/task/tbe_task_builder.cc"
    "../single_op/task/aicpu_task_builder.cc"
    "../single_op/task/aicpu_kernel_task_builder.cc"
    "../single_op/task/rts_kernel_task_builder.cc"
    "../hybrid/common/tensor_value.cc"
    "../hybrid/common/npu_memory_allocator.cc"
    "../hybrid/executor/rt_callback_manager.cc"
    "../hybrid/executor/node_state.cc"
    "../hybrid/executor/node_done_manager.cc"
    "../hybrid/executor/hybrid_profiler.cc"
    "../hybrid/executor/hybrid_model_executor.cc"
    "../hybrid/executor/hybrid_model_pipeline_executor.cc"
    "../hybrid/executor/hybrid_model_async_executor.cc"
    "../hybrid/executor/hybrid_execution_context.cc"
    "../hybrid/executor/subgraph_context.cc"
    "../hybrid/executor/subgraph_executor.cc"
    "../hybrid/executor/worker/task_compile_engine.cc"
    "../hybrid/executor/worker/shape_inference_engine.cc"
    "../hybrid/executor/worker/execution_engine.cc"
    "../hybrid/model/hybrid_model.cc"
    "../hybrid/model/hybrid_model_builder.cc"
    "../hybrid/model/node_item.cc"
    "../hybrid/model/graph_item.cc"
    "../hybrid/node_executor/aicore/aicore_node_executor.cc"
    "../hybrid/node_executor/aicore/aicore_op_task.cc"
    "../hybrid/node_executor/aicore/aicore_task_builder.cc"
    "../hybrid/node_executor/aicpu/aicpu_node_executor.cc"
    "../hybrid/node_executor/compiledsubgraph/known_node_executor.cc"
    "../hybrid/node_executor/ge_local/ge_local_node_executor.cc"
    "../hybrid/node_executor/host_cpu/host_cpu_node_executor.cc"
    "../hybrid/node_executor/controlop/control_op_executor.cc"
    "../hybrid/node_executor/partitioned_call/partitioned_call_node_executor.cc"
    "../hybrid/node_executor/rts/rts_node_executor.cc"
    "../hybrid/node_executor/rts/rts_node_task.cc"
    "../hybrid/node_executor/rts/rts_task_factory.cc"
    "../hybrid/node_executor/node_executor.cc"
    "../hybrid/node_executor/task_context.cc"
    "../hybrid/hybrid_davinci_model.cc"
    "../ge_local_engine/engine/host_cpu_engine.cc"
    "../graph/manager/host_mem_manager.cc"
    "../graph/build/memory/var_mem_assign_util.cc"
    "../host_kernels/transpose_kernel.cc"
    "../host_kernels/add_kernel.cc"
    "../host_kernels/broadcast_args_kernel.cc"
    "../host_kernels/broadcast_gradient_args_kernel.cc"
    "../host_kernels/cast_kernel.cc"
    "../host_kernels/concat_offset_kernel.cc"
    "../host_kernels/concat_v2_kernel.cc"
    "../host_kernels/dynamic_stitch_kernel.cc"
    "../host_kernels/identity_kernel.cc"
    "../host_kernels/empty_kernel.cc"
    "../host_kernels/expanddims_kernel.cc"
    "../host_kernels/fill_kernel.cc"
    "../host_kernels/floordiv_kernel.cc"
    "../host_kernels/floormod_kernel.cc"
    "../host_kernels/gather_v2_kernel.cc"
    "../host_kernels/greater_kernel.cc"
    "../host_kernels/kernel_utils.cc"
    "../host_kernels/maximum_kernel.cc"
    "../host_kernels/mul_kernel.cc"
    "../host_kernels/pack_kernel.cc"
    "../host_kernels/permute_kernel.cc"
    "../host_kernels/range_kernel.cc"
    "../host_kernels/rank_kernel.cc"
    "../host_kernels/reduce_prod_kernel.cc"
    "../host_kernels/reshape_kernel.cc"
    "../host_kernels/rsqrt_kernel.cc"
    "../host_kernels/shape_kernel.cc"
    "../host_kernels/shape_n_kernel.cc"
    "../host_kernels/size_kernel.cc"
    "../host_kernels/slice_d_kernel.cc"
    "../host_kernels/slice_kernel.cc"
    "../host_kernels/squeeze_kernel.cc"
    "../host_kernels/unsqueeze_kernel.cc"
    "../host_kernels/ssd_prior_box_kernel.cc"
    "../host_kernels/strided_slice_kernel.cc"
    "../host_kernels/sub_kernel.cc"
    "../host_kernels/transdata_kernel.cc"
    "../host_kernels/unpack_kernel.cc"
    "../graph/passes/pass_utils.cc"
)

######## libge_executor.a ########
add_library(ge_executor STATIC ${SRC_LIST})

add_dependencies(ge_executor
    graphengine_protos
)

target_compile_options(ge_executor PRIVATE
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-fvisibility=hidden -O2 -Werror -Wno-deprecated-declarations -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-variable>
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-const-variable -Werror=format>
)

target_compile_definitions(ge_executor PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    DAVINCI_SUPPORT_PROFILING
    google=ascend_private
    $<IF:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,OS_TYPE=WIN,OS_TYPE=0>
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>:SECUREC_USING_STD_SECURE_LIB=0 NOMINMAX>
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
    LOG_CPP
)

target_include_directories(ge_executor SYSTEM PRIVATE
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
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<TARGET_PROPERTY:runtime_headers,INTERFACE_INCLUDE_DIRECTORIES>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<TARGET_PROPERTY:cce_headers,INTERFACE_INCLUDE_DIRECTORIES>>
    #### blue zone ####
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain>
)

target_link_libraries(ge_executor PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    json
    ascend_protobuf_static
    c_sec
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

######## libge_executor.so ########
add_library(ge_executor_shared SHARED ${SRC_LIST})

add_dependencies(ge_executor_shared
    graphengine_protos
)

target_compile_options(ge_executor_shared PRIVATE
    -fno-common
    -Werror
    -O2
    -Wno-deprecated-declarations
    -fvisibility=hidden
)

target_compile_definitions(ge_executor_shared PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    DAVINCI_SUPPORT_PROFILING
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_include_directories(ge_executor_shared PRIVATE
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
)

target_link_options(ge_executor_shared PRIVATE
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_libraries(ge_executor_shared PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:cce_headers>>
    -Wl,--no-as-needed
    ge_common
    runtime
    slog
    graph
    register
    error_manager
    ascend_protobuf
    c_sec
    -Wl,--as-needed
    json
    $<$<NOT:$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>:-lrt>
    -ldl
)

set_target_properties(ge_executor_shared PROPERTIES
    OUTPUT_NAME ge_executor
)

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS ge_executor_shared OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)
