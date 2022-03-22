LOCAL_PATH := $(call my-dir)

local_ge_executor_src_files :=  \
    ge_executor.cc \
    ../common/profiling/profiling_manager.cc \
    ../common/dump/dump_properties.cc \
    ../common/dump/dump_manager.cc \
    ../common/dump/dump_op.cc \
    ../common/ge/plugin_manager.cc \
    ../common/ge/op_tiling_manager.cc \
    ../common/profiling/ge_profiling.cc \
    ../graph/load/graph_loader.cc \
    ../graph/execute/graph_execute.cc \
    ../graph/manager/graph_manager_utils.cc \
    ../graph/manager/graph_var_manager.cc \
    ../graph/manager/rdma_pool_allocator.cc \
    ../graph/manager/host_mem_allocator.cc \
    ../graph/manager/graph_mem_allocator.cc \
    ../graph/manager/graph_caching_allocator.cc \
    ../graph/manager/trans_var_data_utils.cc \
    ../graph/manager/util/debug.cc \
    ../model/ge_model.cc \
    ../model/ge_root_model.cc \
    ../graph/load/model_manager/davinci_model.cc \
    ../graph/load/model_manager/davinci_model_parser.cc \
    ../graph/load/model_manager/model_manager.cc \
    ../graph/load/model_manager/tbe_handle_store.cc \
    ../graph/load/model_manager/cpu_queue_schedule.cc \
    ../graph/load/model_manager/model_utils.cc \
    ../graph/load/model_manager/aipp_utils.cc \
    ../graph/load/model_manager/data_inputer.cc \
    ../graph/load/model_manager/data_dumper.cc \
    ../graph/load/model_manager/zero_copy_task.cc \
    ../graph/load/model_manager/zero_copy_offset.cc \
    ../graph/load/model_manager/task_info/task_info.cc                  \
    ../graph/load/model_manager/task_info/event_record_task_info.cc     \
    ../graph/load/model_manager/task_info/event_wait_task_info.cc       \
    ../graph/load/model_manager/task_info/fusion_start_task_info.cc     \
    ../graph/load/model_manager/task_info/fusion_stop_task_info.cc      \
    ../graph/load/model_manager/task_info/kernel_ex_task_info.cc        \
    ../graph/load/model_manager/task_info/kernel_task_info.cc           \
    ../graph/load/model_manager/task_info/label_set_task_info.cc        \
    ../graph/load/model_manager/task_info/label_switch_by_index_task_info.cc \
    ../graph/load/model_manager/task_info/label_goto_ex_task_info.cc    \
    ../graph/load/model_manager/task_info/memcpy_async_task_info.cc     \
    ../graph/load/model_manager/task_info/memcpy_addr_async_task_info.cc \
    ../graph/load/model_manager/task_info/profiler_trace_task_info.cc   \
    ../graph/load/model_manager/task_info/stream_active_task_info.cc    \
    ../graph/load/model_manager/task_info/stream_switch_task_info.cc    \
    ../graph/load/model_manager/task_info/stream_switchn_task_info.cc   \
    ../graph/load/model_manager/task_info/end_graph_task_info.cc        \
    ../graph/load/model_manager/task_info/model_exit_task_info.cc       \
    ../graph/load/model_manager/task_info/super_kernel/super_kernel_factory.cc   \
    ../graph/load/model_manager/task_info/super_kernel/super_kernel.cc  \
    ../opskernel_manager/ops_kernel_builder_manager.cc \
    ../single_op/single_op_manager.cc \
    ../single_op/single_op_model.cc \
    ../single_op/single_op.cc \
    ../single_op/stream_resource.cc \
    ../single_op/task/op_task.cc \
    ../single_op/task/build_task_utils.cc \
    ../single_op/task/tbe_task_builder.cc \
    ../single_op/task/aicpu_task_builder.cc \
    ../single_op/task/aicpu_kernel_task_builder.cc \
    ../hybrid/node_executor/aicpu/aicpu_ext_info.cc \
    ../common/local_context.cc \
    ../hybrid/common/tensor_value.cc                                        \
    ../hybrid/common/npu_memory_allocator.cc                                \
    ../hybrid/executor/rt_callback_manager.cc                               \
    ../hybrid/executor/node_state.cc                                        \
    ../hybrid/executor/node_done_manager.cc                                 \
    ../hybrid/executor/hybrid_profiler.cc                                   \
    ../hybrid/executor/hybrid_model_executor.cc                             \
    ../hybrid/executor/hybrid_model_async_executor.cc                       \
    ../hybrid/executor/hybrid_execution_context.cc                          \
    ../hybrid/executor/subgraph_context.cc                                  \
    ../hybrid/executor/subgraph_executor.cc                                 \
    ../hybrid/executor/worker/task_compile_engine.cc                        \
    ../hybrid/executor/worker/shape_inference_engine.cc                     \
    ../hybrid/executor/worker/execution_engine.cc                           \
    ../hybrid/model/hybrid_model.cc                                         \
    ../hybrid/model/hybrid_model_builder.cc                                 \
    ../hybrid/model/node_item.cc                                            \
    ../hybrid/model/graph_item.cc                                           \
    ../hybrid/node_executor/aicore/aicore_node_executor.cc                  \
    ../hybrid/node_executor/aicore/aicore_op_task.cc                        \
    ../hybrid/node_executor/aicore/aicore_task_builder.cc                   \
    ../hybrid/node_executor/aicpu/aicpu_node_executor.cc                    \
    ../hybrid/node_executor/compiledsubgraph/known_node_executor.cc         \
    ../hybrid/node_executor/ge_local/ge_local_node_executor.cc              \
    ../hybrid/node_executor/host_cpu/host_cpu_node_executor.cc              \
    ../hybrid/node_executor/host_cpu/kernel_factory.cc                      \
    ../hybrid/node_executor/host_cpu/kernel/no_op_kernel.cc                 \
    ../hybrid/node_executor/host_cpu/kernel/variable_kernel.cc              \
    ../hybrid/node_executor/host_cpu/kernel/assign_kernel.cc                \
    ../hybrid/node_executor/host_cpu/kernel/random_uniform_kernel.cc        \
    ../hybrid/node_executor/host_cpu/kernel/data_kernel.cc                  \
    ../hybrid/node_executor/controlop/control_op_executor.cc                \
    ../hybrid/node_executor/partitioned_call/partitioned_call_node_executor.cc \
    ../hybrid/node_executor/rts/rts_node_executor.cc                        \
    ../hybrid/node_executor/node_executor.cc                                \
    ../hybrid/node_executor/task_context.cc                                 \
    ../hybrid/hybrid_davinci_model.cc                                       \
    ../ge_local_engine/engine/host_cpu_engine.cc \
    ../common/omg_util.cc \
    ../graph/manager/host_mem_manager.cc \
    ../graph/build/memory/var_mem_assign_util.cc \
    ../host_kernels/transpose_kernel.cc \
    ../host_kernels/add_kernel.cc \
    ../host_kernels/broadcast_args_kernel.cc \
    ../host_kernels/broadcast_gradient_args_kernel.cc \
    ../host_kernels/cast_kernel.cc \
    ../host_kernels/concat_offset_kernel.cc \
    ../host_kernels/concat_v2_kernel.cc \
    ../host_kernels/dynamic_stitch_kernel.cc \
    ../host_kernels/identity_kernel.cc \
    ../host_kernels/empty_kernel.cc \
    ../host_kernels/expanddims_kernel.cc \
    ../host_kernels/fill_kernel.cc \
    ../host_kernels/floordiv_kernel.cc \
    ../host_kernels/floormod_kernel.cc \
    ../host_kernels/gather_v2_kernel.cc  \
    ../host_kernels/greater_kernel.cc \
    ../host_kernels/kernel_utils.cc \
    ../host_kernels/maximum_kernel.cc \
    ../host_kernels/mul_kernel.cc \
    ../host_kernels/pack_kernel.cc \
    ../host_kernels/permute_kernel.cc \
    ../host_kernels/range_kernel.cc \
    ../host_kernels/rank_kernel.cc \
    ../host_kernels/reduce_prod_kernel.cc \
    ../host_kernels/reshape_kernel.cc \
    ../host_kernels/rsqrt_kernel.cc \
    ../host_kernels/shape_kernel.cc \
    ../host_kernels/shape_n_kernel.cc \
    ../host_kernels/size_kernel.cc \
    ../host_kernels/slice_d_kernel.cc \
    ../host_kernels/slice_kernel.cc \
    ../host_kernels/squeeze_kernel.cc \
    ../host_kernels/unsqueeze_kernel.cc \
    ../host_kernels/ssd_prior_box_kernel.cc \
    ../host_kernels/strided_slice_kernel.cc \
    ../host_kernels/sub_kernel.cc \
    ../host_kernels/transdata_kernel.cc \
    ../host_kernels/unpack_kernel.cc \
    ../graph/passes/pass_utils.cc \
    ../common/bcast.cc \
    ../common/fp16_t.cc \
    ../common/formats/format_transfers/format_transfer_transpose.cc \
    ../common/formats/utils/formats_trans_utils.cc \

local_ge_executor_c_include :=             \
    proto/insert_op.proto                  \
    proto/op_mapping_info.proto            \
    proto/dump_task.proto                  \
    proto/ge_ir.proto                      \
    proto/task.proto                       \
    proto/om.proto                         \
    $(TOPDIR)inc/external                  \
    $(TOPDIR)metadef/inc/external                  \
    $(TOPDIR)graphengine/inc/external                  \
    $(TOPDIR)metadef/inc/external/graph            \
    $(TOPDIR)graphengine/inc/framework                 \
    $(TOPDIR)inc                           \
    $(TOPDIR)metadef/inc                           \
    $(TOPDIR)graphengine/inc                           \
    $(LOCAL_PATH)/../                      \
    $(TOPDIR)graphengine/ge                \
    $(TOPDIR)libc_sec/include              \
    third_party/protobuf/include           \
    third_party/json/include               \

local_ge_executor_shared_library :=        \
    libascend_protobuf                            \
    libc_sec                               \
    libge_common                           \
    libruntime                             \
    libslog                                \
    libmmpa                                \
    libgraph                               \
    libregister                            \
    liberror_manager                       \

local_ge_executor_ldflags := -lrt -ldl     \


#compile arm  device dynamic lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_executor
LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -O2 -DDAVINCI_SUPPORT_PROFILING -Dgoogle=ascend_private

LOCAL_SRC_FILES := $(local_ge_executor_src_files)
LOCAL_C_INCLUDES := $(local_ge_executor_c_include)

LOCAL_SHARED_LIBRARIES := $(local_ge_executor_shared_library)

LOCAL_SHARED_LIBRARIES += libascend_hal

LOCAL_STATIC_LIBRARIES := \
    libmsprofiler \

ifeq ($(device_os),android)
LOCAL_LDFLAGS += -ldl
LOCAL_LDLIBS += -L$(PWD)/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29 -llog
else
LOCAL_LDFLAGS += $(local_ge_executor_ldflags)
endif

include $(BUILD_SHARED_LIBRARY)

#compile x86 host dynamic lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_executor
LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DDAVINCI_SUPPORT_PROFILING -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
else
LOCAL_CFLAGS += -O2
endif

LOCAL_SRC_FILES := $(local_ge_executor_src_files)

LOCAL_C_INCLUDES := $(local_ge_executor_c_include)

LOCAL_SHARED_LIBRARIES :=                  \
    libascend_protobuf                            \
    libc_sec                               \
    libge_common                           \
    libruntime                             \
    libslog                                \
    libmmpa                                \
    libgraph                               \
    libregister                            \
    liberror_manager                       \
    stub/libascend_hal                     \

LOCAL_STATIC_LIBRARIES := \
    libmsprofiler \

LOCAL_LDFLAGS += $(local_ge_executor_ldflags)

include $(BUILD_HOST_SHARED_LIBRARY)

#compile for host static lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_executor
LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DDAVINCI_SUPPORT_PROFILING -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
else
LOCAL_CFLAGS += -O2
endif

LOCAL_SRC_FILES := $(local_ge_executor_src_files)

LOCAL_C_INCLUDES := $(local_ge_executor_c_include)

LOCAL_STATIC_LIBRARIES := \
    libge_common \
    libgraph     \
    libregister  \
    libascend_protobuf  \

LOCAL_SHARED_LIBRARIES :=                  \
    libc_sec                               \
    libruntime                             \
    libslog                                \
    libmmpa                                \

LOCAL_LDFLAGS += $(local_ge_executor_ldflags)

include $(BUILD_HOST_STATIC_LIBRARY)

#compile for device static lib
include $(CLEAR_VARS)

LOCAL_MODULE := libge_executor
LOCAL_CFLAGS += -Werror -Wno-deprecated-declarations
LOCAL_CFLAGS += -DPROTOBUF_INLINE_NOT_IN_HEADERS=0 -DDAVINCI_SUPPORT_PROFILING -Dgoogle=ascend_private
ifeq ($(DEBUG), 1)
LOCAL_CFLAGS += -g -O0
else
LOCAL_CFLAGS += -O2
endif

LOCAL_SRC_FILES := $(local_ge_executor_src_files)
LOCAL_C_INCLUDES := $(local_ge_executor_c_include)

LOCAL_STATIC_LIBRARIES := \
    libge_common \
    libgraph     \
    libregister  \
    libascend_protobuf  \

LOCAL_SHARED_LIBRARIES :=                  \
    libc_sec                               \
    libruntime                             \
    libslog                                \
    libmmpa                                \

ifeq ($(device_os),android)
LOCAL_LDFLAGS += -ldl
LOCAL_LDLIBS += -L$(PWD)/prebuilts/clang/linux-x86/aarch64/android-ndk-r21/sysroot/usr/lib/aarch64-linux-android/29 -llog
else
LOCAL_LDFLAGS += $(local_ge_executor_ldflags)
endif

include $(BUILD_STATIC_LIBRARY)
