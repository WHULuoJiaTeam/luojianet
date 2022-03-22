if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
    add_subdirectory(common)
    add_subdirectory(plugin/engine)
    add_subdirectory(ge_local_engine)
    add_subdirectory(offline)
elseif (ENABLE_D)
    add_subdirectory(common)
    add_subdirectory(ge_runtime)
endif ()

set(GRAPHENGINE_PROTO_LIST
    "${METADEF_DIR}/proto/om.proto"
    "${METADEF_DIR}/proto/task.proto"
    "${METADEF_DIR}/proto/insert_op.proto"
    "${METADEF_DIR}/proto/ge_ir.proto"
    "${METADEF_DIR}/proto/ge_api.proto"
    "${METADEF_DIR}/proto/fusion_model.proto"
    "${METADEF_DIR}/proto/optimizer_priority.proto"
    "${METADEF_DIR}/proto/fwk_adapter.proto"
    "${METADEF_DIR}/proto/op_mapping.proto"
    "${METADEF_DIR}/proto/dump_task.proto"
    "${METADEF_DIR}/proto/tensorflow/graph_library.proto"
    "${METADEF_DIR}/proto/tensorflow/graph.proto"
    "${METADEF_DIR}/proto/tensorflow/node_def.proto"
    "${METADEF_DIR}/proto/tensorflow/function.proto"
    "${METADEF_DIR}/proto/tensorflow/versions.proto"
    "${METADEF_DIR}/proto/tensorflow/attr_value.proto"
    "${METADEF_DIR}/proto/tensorflow/op_def.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor_shape.proto"
    "${METADEF_DIR}/proto/tensorflow/types.proto"
    "${METADEF_DIR}/proto/tensorflow/resource_handle.proto"
)

protobuf_generate(graphengine_protos GRAPHENGINE_PROTO_SRCS GRAPHENGINE_PROTO_HDRS ${GRAPHENGINE_PROTO_LIST} TARGET)

set(GE_FUSION_MODEL_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto/fusion_model.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto/optimizer_priority.pb.cc"
)


add_library(ge_fusion_model_protos_obj OBJECT ${GE_FUSION_MODEL_PROTO_SRCS})
add_dependencies(ge_fusion_model_protos_obj graphengine_protos)
target_include_directories(ge_fusion_model_protos_obj PRIVATE
    #### blue zone ####
    ${PROTOBUF_SHARED_PKG_DIR}/include
    #### yellow zone ####
    ${ASCEND_PROTOBUF_SHARED_PKG_DIR}/include
)
target_compile_definitions(ge_fusion_model_protos_obj PRIVATE
    google=ascend_private
)
target_link_libraries(ge_fusion_model_protos_obj PRIVATE ascend_protobuf $<BUILD_INTERFACE:intf_pub>)
target_compile_options(ge_fusion_model_protos_obj PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
############ libge_proto_common.a ############
add_library(ge_proto_common STATIC
    $<TARGET_OBJECTS:ge_fusion_model_protos_obj>
)

target_compile_definitions(ge_proto_common PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(ge_proto_common PRIVATE
    -O2
    -fno-common
)

target_link_libraries(ge_proto_common PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
)

############ libge_proto_client.a ############
add_library(ge_proto_client STATIC
    $<TARGET_OBJECTS:ge_fusion_model_protos_obj>
)

target_compile_definitions(ge_proto_client PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_include_directories(ge_proto_client PRIVATE
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos/proto
)

target_compile_options(ge_proto_client PRIVATE
    -O2
    -fno-common
)

target_link_libraries(ge_proto_client PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
)
endif ()

##################################################################
set(EXECUTOR_SRC_LIST
    "common/dump/dump_op.cc"
    "common/dump/exception_dumper.cc"
    "common/dump/opdebug_register.cc"
    "common/ge/op_tiling_manager.cc"
    "common/ge/plugin_manager.cc"
    "common/profiling/ge_profiling.cc"
    "common/profiling/profiling_manager.cc"
    "executor/ge_executor.cc"
    "ge_local_engine/engine/host_cpu_engine.cc"
    "graph/build/memory/var_mem_assign_util.cc"
    "graph/execute/graph_execute.cc"
    "graph/execute/model_executor.cc"
    "graph/load/graph_loader.cc"
    "graph/load/model_manager/aipp_utils.cc"
    "graph/load/model_manager/cpu_queue_schedule.cc"
    "graph/load/model_manager/data_dumper.cc"
    "graph/load/model_manager/data_inputer.cc"
    "graph/load/model_manager/davinci_model.cc"
    "graph/load/model_manager/model_manager.cc"
    "graph/load/model_manager/model_utils.cc"
    "graph/load/model_manager/task_info/end_graph_task_info.cc"
    "graph/load/model_manager/task_info/event_record_task_info.cc"
    "graph/load/model_manager/task_info/event_wait_task_info.cc"
    "graph/load/model_manager/task_info/ffts_task_info.cc"
    "graph/load/model_manager/task_info/fusion_start_task_info.cc"
    "graph/load/model_manager/task_info/fusion_stop_task_info.cc"
    #"graph/load/model_manager/task_info/hccl_task_info.cc"         # Just for runner.
    "graph/load/model_manager/task_info/kernel_ex_task_info.cc"
    "graph/load/model_manager/task_info/kernel_task_info.cc"
    "graph/load/model_manager/task_info/label_goto_ex_task_info.cc"
    "graph/load/model_manager/task_info/label_set_task_info.cc"
    "graph/load/model_manager/task_info/label_switch_by_index_task_info.cc"
    "graph/load/model_manager/task_info/memcpy_addr_async_task_info.cc"
    "graph/load/model_manager/task_info/memcpy_async_task_info.cc"
    "graph/load/model_manager/task_info/model_exit_task_info.cc"
    "graph/load/model_manager/task_info/profiler_trace_task_info.cc"
    "graph/load/model_manager/task_info/stream_active_task_info.cc"
    "graph/load/model_manager/task_info/stream_switch_task_info.cc"
    "graph/load/model_manager/task_info/stream_switchn_task_info.cc"
    "graph/load/model_manager/task_info/super_kernel/super_kernel.cc"
    "graph/load/model_manager/task_info/super_kernel/super_kernel_factory.cc"
    "graph/load/model_manager/task_info/task_info.cc"
    "graph/load/model_manager/tbe_handle_store.cc"
    "graph/load/model_manager/zero_copy_offset.cc"
    "graph/load/model_manager/zero_copy_task.cc"
    "graph/manager/graph_caching_allocator.cc"
    "graph/manager/graph_manager_utils.cc"
    "graph/manager/graph_mem_allocator.cc"
    "graph/manager/graph_mem_manager.cc"
    "graph/manager/graph_var_manager.cc"
    "graph/manager/host_mem_allocator.cc"
    "graph/manager/host_mem_manager.cc"
    #"graph/manager/memory_api.cc"                   # Just for runner.
    "graph/manager/rdma_pool_allocator.cc"
    "graph/manager/session_scope_mem_allocator.cc"
    "graph/manager/trans_var_data_utils.cc"
    "graph/manager/util/debug.cc"
    #"graph/manager/util/hcom_util.cc"              # Just for runner.
    "graph/passes/pass_utils.cc"
    "host_kernels/add_kernel.cc"
    "host_kernels/broadcast_args_kernel.cc"
    "host_kernels/broadcast_gradient_args_kernel.cc"
    "host_kernels/cast_kernel.cc"
    "host_kernels/concat_offset_kernel.cc"
    "host_kernels/concat_v2_kernel.cc"
    "host_kernels/dynamic_stitch_kernel.cc"
    "host_kernels/empty_kernel.cc"
    "host_kernels/expanddims_kernel.cc"
    "host_kernels/fill_kernel.cc"
    "host_kernels/floordiv_kernel.cc"
    "host_kernels/floormod_kernel.cc"
    "host_kernels/gather_v2_kernel.cc"
    "host_kernels/greater_kernel.cc"
    "host_kernels/identity_kernel.cc"
    "host_kernels/kernel_utils.cc"
    "host_kernels/maximum_kernel.cc"
    "host_kernels/mul_kernel.cc"
    "host_kernels/pack_kernel.cc"
    "host_kernels/permute_kernel.cc"
    "host_kernels/range_kernel.cc"
    "host_kernels/rank_kernel.cc"
    "host_kernels/reduce_prod_kernel.cc"
    "host_kernels/reformat_kernel.cc"
    "host_kernels/reshape_kernel.cc"
    "host_kernels/rsqrt_kernel.cc"
    "host_kernels/shape_kernel.cc"
    "host_kernels/shape_n_kernel.cc"
    "host_kernels/size_kernel.cc"
    "host_kernels/slice_d_kernel.cc"
    "host_kernels/slice_kernel.cc"
    "host_kernels/squeeze_kernel.cc"
    "host_kernels/ssd_prior_box_kernel.cc"
    "host_kernels/strided_slice_kernel.cc"
    "host_kernels/sub_kernel.cc"
    "host_kernels/transdata_kernel.cc"
    "host_kernels/transpose_kernel.cc"
    "host_kernels/unpack_kernel.cc"
    "host_kernels/unsqueeze_kernel.cc"
    "hybrid/common/npu_memory_allocator.cc"
    "hybrid/common/tensor_value.cc"
    "hybrid/executor/hybrid_execution_context.cc"
    "hybrid/executor/hybrid_model_async_executor.cc"
    "hybrid/executor/hybrid_model_executor.cc"
    "hybrid/executor/hybrid_model_pipeline_executor.cc"
    "hybrid/executor/hybrid_profiler.cc"
    "hybrid/executor/node_done_manager.cc"
    "hybrid/executor/node_state.cc"
    "hybrid/executor/rt_callback_manager.cc"
    "hybrid/executor/subgraph_context.cc"
    "hybrid/executor/subgraph_executor.cc"
    "hybrid/executor/worker/execution_engine.cc"
    "hybrid/executor/worker/shape_inference_engine.cc"
    "hybrid/executor/worker/task_compile_engine.cc"
    "hybrid/hybrid_davinci_model.cc"
    "hybrid/model/graph_item.cc"
    "hybrid/model/hybrid_model.cc"
    "hybrid/model/hybrid_model_builder.cc"
    "hybrid/model/node_item.cc"
    "hybrid/node_executor/aicore/aicore_node_executor.cc"
    "hybrid/node_executor/aicore/aicore_op_task.cc"
    "hybrid/node_executor/aicore/aicore_task_builder.cc"
    "hybrid/node_executor/aicpu/aicpu_ext_info.cc"
    "hybrid/node_executor/aicpu/aicpu_node_executor.cc"
    "hybrid/node_executor/compiledsubgraph/known_node_executor.cc"
    "hybrid/node_executor/controlop/control_op_executor.cc"
    "hybrid/node_executor/ge_local/ge_local_node_executor.cc"
    #"hybrid/node_executor/hccl/hccl_node_executor.cc"           # Just for runner.
    "hybrid/node_executor/host_cpu/host_cpu_node_executor.cc"
    "hybrid/node_executor/node_executor.cc"
    "hybrid/node_executor/partitioned_call/partitioned_call_node_executor.cc"
    "hybrid/node_executor/rts/rts_node_executor.cc"
    "hybrid/node_executor/rts/rts_node_task.cc"
    "hybrid/node_executor/rts/rts_task_factory.cc"
    "hybrid/node_executor/task_context.cc"
    "opskernel_manager/ops_kernel_builder_manager.cc"
    "single_op/single_op.cc"
    "single_op/single_op_manager.cc"
    "single_op/single_op_model.cc"
    "single_op/stream_resource.cc"
    "single_op/task/aicpu_kernel_task_builder.cc"
    "single_op/task/aicpu_task_builder.cc"
    "single_op/task/build_task_utils.cc"
    "single_op/task/op_task.cc"
    "single_op/task/rts_kernel_task_builder.cc"
    "single_op/task/tbe_task_builder.cc"
)

##################################################################
set(COMPILER_SRC_LIST
    "analyzer/analyzer.cc"
    "common/dump/dump_op.cc"
    "common/ge/op_tiling_manager.cc"
    "common/ge/plugin_manager.cc"
    "common/profiling/profiling_manager.cc"
    "engine_manager/dnnengine_manager.cc"
    "ge_local_engine/engine/host_cpu_engine.cc"
    "ge_opt_info/ge_opt_info.cc"
    "generator/ge_generator.cc"
    "generator/generator_api.cc"
    "graph/build/graph_builder.cc"
    "graph/build/label_allocator.cc"
    "graph/build/logical_stream_allocator.cc"
    "graph/build/memory/binary_block_mem_assigner.cc"
    "graph/build/memory/block_mem_assigner.cc"
    "graph/build/memory/buffer_pool_mem_assigner.cc"
    "graph/build/memory/graph_mem_assigner.cc"
    "graph/build/memory/hybrid_mem_assigner.cc"
    "graph/build/memory/max_block_mem_assigner.cc"
    "graph/build/memory/memory_assigner.cc"
    "graph/build/memory/var_mem_assign_util.cc"
    "graph/build/model_builder.cc"
    "graph/build/run_context.cc"
    "graph/build/stream_allocator.cc"
    "graph/build/stream_graph_optimizer.cc"
    "graph/build/task_generator.cc"
    "graph/label/case_label_maker.cc"
    "graph/label/if_label_maker.cc"
    "graph/label/label_maker.cc"
    "graph/label/partitioned_call_label_maker.cc"
    "graph/label/while_label_maker.cc"
    "graph/load/model_manager/model_utils.cc"
    "graph/manager/graph_caching_allocator.cc"
    "graph/manager/graph_context.cc"
    "graph/manager/graph_manager.cc"
    "graph/manager/graph_manager_utils.cc"
    "graph/manager/graph_mem_allocator.cc"
    "graph/manager/graph_mem_manager.cc"
    "graph/manager/graph_var_manager.cc"
    "graph/manager/host_mem_allocator.cc"
    "graph/manager/host_mem_manager.cc"
    "graph/manager/rdma_pool_allocator.cc"
    "graph/manager/session_scope_mem_allocator.cc"
    "graph/manager/trans_var_data_utils.cc"
    "graph/manager/util/debug.cc"
    "graph/manager/util/rt_context_util.cc"
    "graph/manager/util/variable_accelerate_ctrl.cc"
    "graph/optimize/graph_optimize.cc"
    "graph/optimize/mem_rw_conflict_optimize.cc"
    "graph/optimize/summary_optimize.cc"
    "graph/partition/dynamic_shape_partition.cc"
    "graph/partition/engine_place.cc"
    "graph/partition/graph_partition.cc"
    "graph/partition/stage_partition.cc"
    "graph/passes/addn_pass.cc"
    "graph/passes/aicpu_constant_folding_pass.cc"
    "graph/passes/assert_pass.cc"
    "graph/passes/assign_remove_pass.cc"
    "graph/passes/atomic_addr_clean_pass.cc"
    "graph/passes/attach_stream_label_pass.cc"
    "graph/passes/base_pass.cc"
    "graph/passes/bitcast_pass.cc"
    "graph/passes/buffer_pool_memory_pass.cc"
    "graph/passes/cast_remove_pass.cc"
    "graph/passes/cast_translate_pass.cc"
    "graph/passes/common_subexpression_elimination_pass.cc"
    "graph/passes/compile_nodes_pass.cc"
    "graph/passes/cond_pass.cc"
    "graph/passes/cond_remove_pass.cc"
    "graph/passes/constant_folding_pass.cc"
    "graph/passes/constant_fuse_same_pass.cc"
    "graph/passes/control_trigger_pass.cc"
    "graph/passes/ctrl_edge_transfer_pass.cc"
    "graph/passes/data_pass.cc"
    "graph/passes/dimension_adjust_pass.cc"
    "graph/passes/dimension_compute_pass.cc"
    "graph/passes/dropout_pass.cc"
    "graph/passes/end_of_sequence_add_control_pass.cc"
    "graph/passes/enter_pass.cc"
    "graph/passes/flow_ctrl_pass.cc"
    "graph/passes/folding_pass.cc"
    "graph/passes/for_pass.cc"
    "graph/passes/fuse_data_nodes_with_common_input_pass.cc"
    "graph/passes/get_original_format_pass.cc"
    "graph/passes/global_step_insert_pass.cc"
    "graph/passes/guarantee_const_pass.cc"
    "graph/passes/hccl_continuous_memcpy_pass.cc"
    "graph/passes/hccl_group_pass.cc"
    "graph/passes/hccl_memcpy_pass.cc"
    "graph/passes/hccl_tailing_optimization_pass.cc"
    "graph/passes/identity_pass.cc"
    "graph/passes/infer_base_pass.cc"
    "graph/passes/infer_value_range_pass.cc"
    "graph/passes/infershape_pass.cc"
    "graph/passes/inplace_support_check_pass.cc"
    "graph/passes/input_output_connection_identify_pass.cc"
    "graph/passes/iterator_op_pass.cc"
    "graph/passes/link_gen_mask_nodes_pass.cc"
    "graph/passes/mark_agnostic_pass.cc"
    "graph/passes/mark_force_unknown_for_cond_pass.cc"
    "graph/passes/mark_graph_unknown_status_pass.cc"
    "graph/passes/mark_node_unknown_shape_pass.cc"
    "graph/passes/mark_same_addr_pass.cc"
    "graph/passes/memcpy_addr_async_pass.cc"
    "graph/passes/merge_input_memcpy_pass.cc"
    "graph/passes/merge_pass.cc"
    "graph/passes/merge_to_stream_merge_pass.cc"
    "graph/passes/multi_batch_clone_pass.cc"
    "graph/passes/multi_batch_pass.cc"
    "graph/passes/net_output_pass.cc"
    "graph/passes/next_iteration_pass.cc"
    "graph/passes/no_use_reshape_remove_pass.cc"
    "graph/passes/parallel_concat_start_op_pass.cc"
    "graph/passes/parallel_group_pass.cc"
    "graph/passes/pass_manager.cc"
    "graph/passes/pass_utils.cc"
    "graph/passes/permute_pass.cc"
    "graph/passes/placeholder_with_default_pass.cc"
    "graph/passes/prevent_gradient_pass.cc"
    "graph/passes/print_op_pass.cc"
    "graph/passes/prune_pass.cc"
    "graph/passes/ref_identity_delete_op_pass.cc"
    "graph/passes/remove_same_const_pass.cc"
    "graph/passes/replace_transshape_pass.cc"
    "graph/passes/replace_with_empty_const_pass.cc"
    "graph/passes/reshape_recovery_pass.cc"
    "graph/passes/reshape_remove_pass.cc"
    "graph/passes/resource_pair_add_control_pass.cc"
    "graph/passes/resource_pair_remove_control_pass.cc"
    "graph/passes/same_transdata_breadth_fusion_pass.cc"
    "graph/passes/save_pass.cc"
    "graph/passes/set_input_output_offset_pass.cc"
    "graph/passes/shape_operate_op_remove_pass.cc"
    "graph/passes/snapshot_pass.cc"
    "graph/passes/stop_gradient_pass.cc"
    "graph/passes/subexpression_migration_pass.cc"
    "graph/passes/subgraph_const_migration_pass.cc"
    "graph/passes/subgraph_pass.cc"
    "graph/passes/switch_data_edges_bypass.cc"
    "graph/passes/switch_dead_branch_elimination.cc"
    "graph/passes/switch_logic_remove_pass.cc"
    "graph/passes/switch_to_stream_switch_pass.cc"
    "graph/passes/transop_breadth_fusion_pass.cc"
    "graph/passes/transop_depth_fusion_pass.cc"
    "graph/passes/transop_nearby_allreduce_fusion_pass.cc"
    "graph/passes/transop_symmetry_elimination_pass.cc"
    "graph/passes/transop_without_reshape_fusion_pass.cc"
    "graph/passes/transpose_transdata_pass.cc"
    "graph/passes/unused_args_clean_pass.cc"
    "graph/passes/unused_const_pass.cc"
    "graph/passes/useless_control_out_remove_pass.cc"
    "graph/passes/var_is_initialized_op_pass.cc"
    "graph/passes/variable_op_pass.cc"
    "graph/passes/variable_prepare_op_pass.cc"
    "graph/passes/variable_ref_delete_op_pass.cc"
    "graph/passes/variable_ref_useless_control_out_delete_pass.cc"
    "graph/preprocess/graph_preprocess.cc"
    "graph/preprocess/insert_op/ge_aipp_op.cc"
    "graph/preprocess/insert_op/util_insert_aipp_op.cc"
    "graph/preprocess/multi_batch_copy_graph.cc"
    "graph/preprocess/multi_batch_options.cc"
    "host_kernels/add_kernel.cc"
    "host_kernels/broadcast_args_kernel.cc"
    "host_kernels/broadcast_gradient_args_kernel.cc"
    "host_kernels/cast_kernel.cc"
    "host_kernels/concat_offset_kernel.cc"
    "host_kernels/concat_v2_kernel.cc"
    "host_kernels/dynamic_stitch_kernel.cc"
    "host_kernels/empty_kernel.cc"
    "host_kernels/expanddims_kernel.cc"
    "host_kernels/fill_kernel.cc"
    "host_kernels/floordiv_kernel.cc"
    "host_kernels/floormod_kernel.cc"
    "host_kernels/gather_v2_kernel.cc"
    "host_kernels/greater_kernel.cc"
    "host_kernels/identity_kernel.cc"
    "host_kernels/kernel_utils.cc"
    "host_kernels/maximum_kernel.cc"
    "host_kernels/mul_kernel.cc"
    "host_kernels/pack_kernel.cc"
    "host_kernels/permute_kernel.cc"
    "host_kernels/range_kernel.cc"
    "host_kernels/rank_kernel.cc"
    "host_kernels/reduce_prod_kernel.cc"
    "host_kernels/reformat_kernel.cc"
    "host_kernels/reshape_kernel.cc"
    "host_kernels/rsqrt_kernel.cc"
    "host_kernels/shape_kernel.cc"
    "host_kernels/shape_n_kernel.cc"
    "host_kernels/size_kernel.cc"
    "host_kernels/slice_d_kernel.cc"
    "host_kernels/slice_kernel.cc"
    "host_kernels/squeeze_kernel.cc"
    "host_kernels/ssd_prior_box_kernel.cc"
    "host_kernels/strided_slice_kernel.cc"
    "host_kernels/sub_kernel.cc"
    "host_kernels/transdata_kernel.cc"
    "host_kernels/transpose_kernel.cc"
    "host_kernels/unpack_kernel.cc"
    "host_kernels/unsqueeze_kernel.cc"
    "hybrid/node_executor/aicpu/aicpu_ext_info.cc"
    "init/gelib.cc"
    "ir_build/attr_options/keep_dtype_option.cc"
    "ir_build/attr_options/utils.cc"
    "ir_build/attr_options/weight_compress_option.cc"
    "ir_build/ge_ir_build.cc"
    "ir_build/option_utils.cc"
    "opskernel_manager/ops_kernel_builder_manager.cc"
    "opskernel_manager/ops_kernel_manager.cc"
)

set(RUNNER_SRC_LIST
    "client/ge_api.cc"
    "session/inner_session.cc"
    "session/session_manager.cc"
    "common/profiling/ge_runner_profiling.cc"
    "graph/manager/memory_api.cc"
    "graph/manager/util/hcom_util.cc"
    "graph/load/model_manager/task_info/hccl_task_info.cc"
    "hybrid/node_executor/hccl/hccl_node_executor.cc"
    "hybrid/node_executor/aicore/aicore_task_compiler.cc"
)

if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
message("CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")
############ libge_runner.so ############
add_library(ge_runner SHARED
    ${EXECUTOR_SRC_LIST}
    ${COMPILER_SRC_LIST}
    ${RUNNER_SRC_LIST}
    $<TARGET_OBJECTS:$<IF:$<TARGET_EXISTS:msprofiler_fwk>,msprofiler_fwk,msprofiler_fwk_object>>
)

add_library(msprofiler_fwk_object OBJECT IMPORTED GLOBAL)

if (msprofiler_fwk_ext_LIBRARY_DIR)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_fwk_object)
    execute_process(
        COMMAND ar x ${msprofiler_fwk_ext_LIBRARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_fwk_object
    )
    file(GLOB MSPROFILER_FWK_OBJECT_LIST ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_fwk_object/*.o)
    set_property(TARGET msprofiler_fwk_object PROPERTY IMPORTED_OBJECTS ${MSPROFILER_FWK_OBJECT_LIST})
endif()

target_compile_definitions(ge_runner PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    DAVINCI_SUPPORT_PROFILING
    REUSE_MEMORY=1
    FMK_SUPPORT_DUMP
    DAVINCI_CLOUD
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_runner PRIVATE
    -O2
    -fno-common
    -fvisibility=hidden
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-variable>
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-const-variable -Werror=format>
)

target_include_directories(ge_runner SYSTEM PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    ${GE_CODE_DIR}/../toolchain/ide/ide-daemon/external
    ${GE_CODE_DIR}/../abl/adump/external
    ${GE_CODE_DIR}/../abl/licctrl
    ${GE_CODE_DIR}/../ace/comop/inc
    ${GE_CODE_DIR}/../ace/comop/inc/external
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${GE_DEPEND_DIR}/inc>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<TARGET_PROPERTY:runtime_headers,INTERFACE_INCLUDE_DIRECTORIES>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<TARGET_PROPERTY:cce_headers,INTERFACE_INCLUDE_DIRECTORIES>>
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc/opt_info>
)

target_link_options(ge_runner PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_runner PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:cce_headers>>
    adump_server
    static_mmpa
    ge_proto_common
    ge_proto_client
    -Wl,--no-as-needed
    graph
    ge_common
    ascend_protobuf
    register
    c_sec
    slog
    runtime
    error_manager
    ascend_hal_stub
    opt_feature
    -Wl,--as-needed
    json
    -lrt
    -ldl
)

############ libge_compiler.so ############
add_library(ge_compiler SHARED
    ${COMPILER_SRC_LIST}
)

add_dependencies(ge_compiler
    graphengine_protos
)

target_compile_definitions(ge_compiler PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    REUSE_MEMORY=1
    FMK_SUPPORT_DUMP
    FMK_HOST_INFER
    google=ascend_private
    FUNC_VISIBILITY
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_compile_options(ge_compiler PRIVATE
    -O2
    -fno-common
    -fvisibility=hidden
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-variable>
    $<$<STREQUAL:${CMAKE_CXX_COMPILER_VERSION},7.3.0>:-Werror=unused-const-variable -Werror=format>
)

target_include_directories(ge_compiler SYSTEM PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc
    ${METADEF_DIR}/inc/external
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/proto/graphengine_protos
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    ${GE_CODE_DIR}/../toolchain/ide/ide-daemon/external
    ${GE_CODE_DIR}/../abl/adump/external
    ${GE_CODE_DIR}/../abl/licctrl
    ${GE_CODE_DIR}/../ace/comop/inc
    ${GE_CODE_DIR}/../ace/comop/inc/external
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:${GE_DEPEND_DIR}/inc>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<TARGET_PROPERTY:runtime_headers,INTERFACE_INCLUDE_DIRECTORIES>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<TARGET_PROPERTY:cce_headers,INTERFACE_INCLUDE_DIRECTORIES>>
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain>
    $<$<BOOL:${ENABLE_OPEN_SRC}>:${GE_CODE_DIR}/third_party/fwkacllib/inc/opt_info>
)

target_link_options(ge_compiler PRIVATE
    -Wl,-Bsymbolic
)

target_link_libraries(ge_compiler PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:slog_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:msprof_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:mmpa_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:runtime_headers>>
    $<$<NOT:$<BOOL:${ENABLE_OPEN_SRC}>>:$<BUILD_INTERFACE:cce_headers>>
    static_mmpa
    ge_proto_common
    -Wl,--no-as-needed
    graph
    ge_common
    ascend_protobuf
    register
    c_sec
    error_manager
    slog
    runtime
    opt_feature
    -Wl,--as-needed
    json
    -lrt
    -ldl
)

######## libge_executor.a ########
add_library(ge_executor STATIC
    ${EXECUTOR_SRC_LIST}
)

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
    ${GE_CODE_DIR}/../ace/comop/inc
    ${GE_CODE_DIR}/../ace/comop/inc/external
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
add_library(ge_executor_shared SHARED
    ${EXECUTOR_SRC_LIST}
)

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
    ${GE_CODE_DIR}/../ace/comop/inc
    ${GE_CODE_DIR}/../ace/comop/inc/external
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

############ libascendcl.so ############
file(GENERATE OUTPUT ${CMAKE_BINARY_DIR}/dummy.c CONTENT "")
#add_library(dummy_obj OBJECT ${CMAKE_BINARY_DIR}/dummy.c)
#set(DUMMY_OBJ $<TARGET_OBJECTS:dummy_obj>)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ascendcl_object)

if(EXISTS ${STATIC_ACL_LIB}/libascendcl.a)
    execute_process(
        COMMAND ar x ${STATIC_ACL_LIB}/libascendcl.a
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ascendcl_object
    )
    file(GLOB OBJECT_LIST ${CMAKE_CURRENT_BINARY_DIR}/ascendcl_object/*.o)
else()
    set(OBJECT_LIST ${CMAKE_BINARY_DIR}/dummy.c)
endif()

add_library(opensrc_ascendcl SHARED
    ${OBJECT_LIST}
    $<TARGET_OBJECTS:$<IF:$<TARGET_EXISTS:msprofiler>,msprofiler,msprofiler_object>>
)

add_library(msprofiler_object OBJECT IMPORTED GLOBAL)

if (msprofiler_ext_LIBRARY_DIR)
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_object)
    execute_process(
        COMMAND ar x ${msprofiler_ext_LIBRARY_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_object
    )
    file(GLOB MSPROFILER_OBJECT_LIST ${CMAKE_CURRENT_BINARY_DIR}/msprofiler_object/*.o)
    set_property(TARGET msprofiler_object PROPERTY IMPORTED_OBJECTS ${MSPROFILER_OBJECT_LIST})
endif()

target_compile_definitions(opensrc_ascendcl PRIVATE
    google=ascend_private
)

target_compile_options(opensrc_ascendcl PRIVATE
    -O2
    -fvisibility=hidden
)

target_link_options(opensrc_ascendcl PRIVATE
    -rdynamic
    -Wl,--allow-multiple-definition
    -Wl,-z,muldefs
    -Wl,-Bsymbolic
    -Wl,--exclude-libs,ALL
)

target_link_libraries(opensrc_ascendcl PRIVATE
                     -Wl,--whole-archive
                     ge_executor
                     ge_common_static
                     graph_static
                     static_mmpa
                     ascend_protobuf_static
                     register_static
                     error_manager_static
                     adump_server
                     -Wl,--no-whole-archive
                     -Wl,--no-as-needed
                     c_sec
                     runtime
                     slog
                     ascend_hal_stub
                     -Wl,--as-needed
                     -lrt
                     -ldl
                     json
)

set_target_properties(opensrc_ascendcl PROPERTIES
    OUTPUT_NAME ascendcl
)

##################################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_ir_build.cc
           ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_api.cc
    COMMAND echo "Generating stub files."
            && ${HI_PYTHON} ${CMAKE_CURRENT_LIST_DIR}/stub/gen_stubapi.py ${GE_CODE_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
            && mv ge_ir_build.cc stub_ge_ir_build.cc
            && mv ge_api.cc stub_ge_api.cc
            &&  echo "Generating stub files end."
    #WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    #DEPENDS stub/gen_stubapi.py ${TOP_DIR}/inc/external ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_target(ge_stub
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_ir_build.cc
            ${CMAKE_CURRENT_BINARY_DIR}/stub_ge_api.cc
)

##################################################################
############ stub/libge_compiler.so ############
add_library(atc_stub_ge_compiler SHARED
    stub_ge_ir_build.cc
)

add_dependencies(atc_stub_ge_compiler ge_stub)

target_compile_options(atc_stub_ge_compiler PRIVATE
    -fno-common
)

target_link_libraries(atc_stub_ge_compiler PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(atc_stub_ge_compiler PROPERTIES
    OUTPUT_NAME ge_compiler
    LIBRARY_OUTPUT_DIRECTORY atc_stub
)

target_include_directories(atc_stub_ge_compiler PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/framework
    ${GE_CODE_DIR}/inc/external
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    ${GE_CODE_DIR}/../toolchain/ide/ide-daemon/external
    ${GE_CODE_DIR}/../abl/adump/external
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
)

############ stub/libge_runner.so ############
add_library(fwk_stub_ge_runner SHARED
    stub_ge_api.cc
    stub_ge_ir_build.cc
)

add_dependencies(fwk_stub_ge_runner ge_stub)

target_compile_options(fwk_stub_ge_runner PRIVATE
    -fno-common
)

target_link_libraries(fwk_stub_ge_runner PRIVATE
    $<BUILD_INTERFACE:intf_pub>
)

set_target_properties(fwk_stub_ge_runner PROPERTIES
    OUTPUT_NAME ge_runner
    LIBRARY_OUTPUT_DIRECTORY fwk_stub
)

target_include_directories(fwk_stub_ge_runner PRIVATE
    ${GE_CODE_DIR}/ge
    ${GE_CODE_DIR}/inc
    ${GE_CODE_DIR}/inc/external
    ${GE_CODE_DIR}/inc/framework
    ${METADEF_DIR}/inc/external
    ${METADEF_DIR}/inc
    #### yellow zone ####
    ${GE_CODE_DIR}/../inc
    ${GE_CODE_DIR}/../toolchain/ide/ide-daemon/external
    ${GE_CODE_DIR}/../abl/adump/external
    #### blue zone ####
    ${ASCEND_DIR}/driver/include
    ${ASCEND_DIR}/fwkacllib/include
)

###############################################################
add_custom_target(
    engine_conf.json ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
    COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/engine_manager/engine_conf.json ${CMAKE_CURRENT_BINARY_DIR}/
)

###############################################################
add_custom_target(
    optimizer_priority.pbtxt ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt
)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt
    COMMAND cp ${CMAKE_CURRENT_LIST_DIR}/opskernel_manager/optimizer_priority.pbtxt ${CMAKE_CURRENT_BINARY_DIR}/
)

###############################################################

############ install ############
set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)

install(TARGETS ge_runner ge_compiler ge_executor_shared opensrc_ascendcl OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
)

install(TARGETS atc_stub_ge_compiler fwk_stub_ge_runner OPTIONAL
    LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}/stub
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/engine_conf.json
    ${CMAKE_CURRENT_BINARY_DIR}/optimizer_priority.pbtxt OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)
endif()
