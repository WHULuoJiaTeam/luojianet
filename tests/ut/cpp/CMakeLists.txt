message("build ut testcases...")

# virtual project for common include and library file path.
project(ut)

set(PROJECT_DIR "${PROJECT_SOURCE_DIR}/../../..")
if(ENABLE_DUMP_IR)
    add_compile_definitions(ENABLE_DUMP_IR)
endif()
if(ENABLE_D)
    add_compile_definitions(ENABLE_D)
endif()
if(NOT ENABLE_GLIBCXX)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
endif()

file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")

#add python lib and include for all ut executables;
message("PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
message("PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${MS_CCSRC_PATH})
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core)
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindrt/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/runtime/)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/graphengine_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/plugin/device/cpu/kernel)
MESSAGE("check  ut_test ${CMAKE_BINARY_DIR}")

link_directories(${MS_CCSRC_BUILD_PATH})

if(ENABLE_MINDDATA)
    add_definitions(-D ENABLE_MINDDATA)
    link_directories(${MS_CCSRC_BUILD_PATH}/minddata/dataset)
    link_directories(${MS_CCSRC_BUILD_PATH}/minddata/mindrecord)
endif()
# fetch ut test files
if(ENABLE_MINDDATA)
    include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset)
    include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/kernels/image)
    file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            ./stub/*.cc
            ./common/*.cc
            ./abstract/*.cc
            ./base/*.cc
            ./dataset/*.cc
            ./ir/dtype/*.cc
            ./device/*.cc
            ./ir/*.cc
            ./kernel/*.cc
            ./mindrecord/*.cc
            ./operator/*.cc
            ./optimizer/*.cc
            ./parallel/*.cc
            ./pipeline/*.cc
            ./pre_activate/*.cc
            ./pynative/*.cc
            ./session/*.cc
            ./transform/*.cc
            ./utils/*.cc
            ./vm/*.cc
            ./ps/*.cc
            ./fl/*.cc
            ./distributed/persistent/*.cc
            ./distributed/rpc/tcp/*.cc
            ./distributed/cluster/*.cc
            ./distributed/cluster/topology/*.cc
            ./distributed/recovery/*.cc
            ./cxx_api/*.cc
            ./tbe/*.cc
            ./mindapi/*.cc
            ./runtime/graph_scheduler/*.cc
            )
    if(NOT ENABLE_SECURITY)
        file(GLOB_RECURSE UT_SRCS_DEBUG RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
                ./debug/*.cc)
        list(APPEND UT_SRCS ${UT_SRCS_DEBUG})
    endif()
    if(NOT ENABLE_PYTHON)
        set(PYTHON_RELATED_SRCS
                dataset/filter_op_test.cc
                dataset/voc_op_test.cc
                dataset/manifest_op_test.cc
                dataset/sentence_piece_vocab_op_test.cc
                )
        list(REMOVE_ITEM UT_SRCS ${PYTHON_RELATED_SRCS})
    endif()

    if(NOT ENABLE_ACL)
        set(ASCEND310_RELATED_SRCS
                dataset/dvpp_decode_jpeg_test.cc
                )
        list(REMOVE_ITEM UT_SRCS ${ASCEND310_RELATED_SRCS})
    endif()
else()
    file(GLOB_RECURSE TEMP_UT_SRCS ./*.cc)
    foreach(OBJ ${TEMP_UT_SRCS})
        if(NOT ${OBJ} MATCHES "./dataset/" AND NOT ${OBJ} MATCHES "./mindrecord/")
            list(APPEND UT_SRCS ${OBJ})
        endif()
    endforeach()
endif()

file(GLOB_RECURSE LUOJIANET_MS_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "../../../luojianet_ms/ccsrc/pybind_api/*.cc"
        "../../../luojianet_ms/ccsrc/frontend/optimizer/*.cc"
        "../../../luojianet_ms/ccsrc/frontend/parallel/*.cc"
        "../../../luojianet_ms/ccsrc/frontend/operator/*.cc"
        # dont remove the 4 lines above
        "../../../luojianet_ms/ccsrc/debug/data_dump/dump_json_parser.cc"
        "../../../luojianet_ms/ccsrc/debug/common.cc"
        "../../../luojianet_ms/ccsrc/debug/utils.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/hccl_adapter/all_to_all_v_calc_param.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/kernel_runtime.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/memory_manager.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/memory_scheduler.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/memory_offload_strategy.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/kernel_runtime_manager.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/kernel_info.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/bucket.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/launch_kernel.cc"
        "../../../luojianet_ms/ccsrc/runtime/graph_scheduler/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/profiling/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ge_runtime/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/kernel_select_ascend.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_launch_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_launch_mul.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_launch_atomic_clean.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_launch_transdata.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/kernel_select_graph_kernel.cc"
        "../../../luojianet_ms/ccsrc/runtime/device/convert_tensor_utils.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_bucket.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_event.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/kernel_build_ascend.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_memory_adapter.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_memory_manager.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_device_address.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/ascend_memory_pool.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/lic_manager.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/hardware/ascend_device_context.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_optimization.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/factory/ms_factory.h"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/sparse_apply_adam_cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/sparse_apply_ftrl_cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/sparse_apply_lazy_adam_cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/sparse_apply_proximal_adagrad_cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/unique_cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/unique_with_pad_cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/adam_delta_cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/fused_ada_factor_cpu_kernel.cc"
        "../../../luojianet_ms/ccsrc/kernel/akg/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/akg/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/gpu/kernel/akg/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/akg/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/rts/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/hccl/*.cc"
        "../../../luojianet_ms/ccsrc/kernel/kernel_query.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_attr_to_input_registry.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_kernel_metadata.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/rts/rt_kernel_info.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/tbe/*.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/optimizer/*.cc"
        "../../../luojianet_ms/ccsrc/common/graph_kernel/*.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/anf_runtime_algorithm.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/ascend_session.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/ascend_auto_monad.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/ascend_control_parser.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/kernel_graph.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/session_basic.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/executor.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/executor_manager.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/session_factory.cc"
        "../../../luojianet_ms/ccsrc/backend/common/session/kernel_build_client.cc"
        "../../../luojianet_ms/ccsrc/ps/*.cc"
        "../../../luojianet_ms/ccsrc/fl/*.cc"
        "../../../luojianet_ms/ccsrc/distributed/cluster/actor_route_table_service.cc"
        "../../../luojianet_ms/ccsrc/distributed/cluster/actor_route_table_proxy.cc"
        "../../../luojianet_ms/ccsrc/distributed/persistent/*.cc"
        "../../../luojianet_ms/ccsrc/distributed/rpc/tcp/*.cc"
        "../../../luojianet_ms/ccsrc/distributed/cluster/topology/*.cc"
        "../../../luojianet_ms/ccsrc/profiler/device/ascend/*.cc"
        "../../../luojianet_ms/ccsrc/profiler/device/profiling.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.c"
        "../../../luojianet_ms/ccsrc/kernel/kernel.cc"
        "../../../luojianet_ms/ccsrc/kernel/ascend_kernel_mod.cc"
        "../../../luojianet_ms/ccsrc/backend/common/optimizer/helper.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/executor/aicpu_ext_info_handle.cc"
        "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/aicpu/aicpu_util.cc"
        )

list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST
  "../../../luojianet_ms/ccsrc/frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/util.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/scheduler.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/optimizer_info.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/optimizer_info_builder.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/worker.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/parameter_server.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/ps_cache/gpu/gpu_ps_cache.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/ps_cache/ascend/ascend_ps_cache.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/ps/ps_cache/ps_cache_manager.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/fl/server/kernel/sgd_kernel.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/fl/server/kernel/apply_momentum_kernel.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST
        "../../../luojianet_ms/ccsrc/plugin/device/gpu/optimizer/batch_norm_add_relu_fusion.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST
        "../../../luojianet_ms/ccsrc/plugin/device/gpu/optimizer/post_batch_norm_add_relu_fusion.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST
        "../../../luojianet_ms/ccsrc/plugin/device/gpu/optimizer/batch_norm_add_relu_grad_fusion.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/plugin/device/gpu/optimizer/batch_norm_relu_fusion.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST
        "../../../luojianet_ms/ccsrc/plugin/device/gpu/optimizer/batch_norm_relu_grad_fusion.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/common/graph_kernel/lite_adapter/akg_build.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/common/graph_kernel/lite_adapter/build_kernel.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST
  "../../../luojianet_ms/ccsrc/common/graph_kernel/lite_adapter/callback_impl.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST
  "../../../luojianet_ms/ccsrc/common/graph_kernel/lite_adapter/graph_kernel_optimization.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST
  "../../../luojianet_ms/ccsrc/common/graph_kernel/lite_adapter/graph_kernel_pass_manager.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/common/graph_kernel/lite_adapter/runtime/akg_kernel.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/common/graph_kernel/lite_adapter/common/infer_shape.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/tbe/tbe_kernel_compile.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/akg/akg_cpu_kernel_mod.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/plugin/device/cpu/kernel/akg/akg_cpu_kernel_build.cc")
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/runtime/graph_scheduler/rpc_node_scheduler.cc")

if(ENABLE_SECURITY)
    list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/profiling.cc")
    list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/ascend/memory_profiling.cc")
    list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/ascend/ascend_profiling.cc")
    list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/ascend/options.cc")
    list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/debug/data_dump/dump_json_parser.cc")
endif()
list(REMOVE_ITEM LUOJIANET_MS_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/ascend/parallel_strategy_profiling.cc")

add_library(_ut_luojianet_ms_obj OBJECT ${LUOJIANET_MS_SRC_LIST})
add_library(_ut_ut_obj OBJECT ${UT_SRCS})
add_dependencies(_ut_ut_obj engine-cache-server)
set(ut_objects $<TARGET_OBJECTS:_ut_ut_obj> $<TARGET_OBJECTS:_ut_luojianet_ms_obj>
        $<TARGET_OBJECTS:core_obj> $<TARGET_OBJECTS:core_proto_obj> $<TARGET_OBJECTS:mindrt_mid>
        $<TARGET_OBJECTS:luojianet_ms_shared_lib_obj> $<TARGET_OBJECTS:_luojianet_ms_utils_obj>
        $<TARGET_OBJECTS:_luojianet_ms_common_obj>)
if(ENABLE_MINDDATA)
    set(ut_objects ${ut_objects} ${dataengine_submodules} $<TARGET_OBJECTS:mindrecord_obj>)
endif()
add_executable(ut_tests ${ut_objects})

include_directories("${CMAKE_BINARY_DIR}/plugin/device/ascend/kernel/aicpu")
file(GLOB_RECURSE PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "../../../luojianet_ms/ccsrc/plugin/device/ascend/kernel/aicpu/proto/*.proto")
ms_protobuf_generate(PROTOSRCS PROTOHDRS ${PROTO_IN})

file(GLOB_RECURSE PROTO_DUMP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "../../../luojianet_ms/ccsrc/plugin/device/ascend/hal/device/dump/proto/*.proto")
ms_protobuf_generate(DUMP_PROTOSRCS PROTOHDRS ${PROTO_DUMP})

list(APPEND LUOJIANET_MS_PROTO_LIST ${PROTOSRCS})
list(APPEND LUOJIANET_MS_PROTO_LIST ${PREDICT_PROTOSRCS})
list(APPEND LUOJIANET_MS_PROTO_LIST ${DUMP_PROTOSRCS})

if(LUOJIANET_MS_PROTO_LIST)
    add_library(proto_input_ut STATIC ${LUOJIANET_MS_PROTO_LIST})
    set_target_properties(proto_input_ut PROPERTIES COMPILE_FLAGS "-Wno-unused-variable -Wno-array-bounds")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ut_tests PRIVATE luojianet_ms::gtest luojianet_ms::event luojianet_ms::event_pthreads
                          luojianet_ms::event_openssl luojianet_ms::ssl luojianet_ms::crypto ${PYTHON_LIBRARIES} pthread util dl)
    if(ENABLE_MINDDATA)
        target_link_libraries(ut_tests PRIVATE luojianet_ms::sqlite luojianet_ms::jpeg_turbo luojianet_ms::turbojpeg
                luojianet_ms::opencv_core luojianet_ms::opencv_imgcodecs luojianet_ms::opencv_imgproc luojianet_ms::tinyxml2
                luojianet_ms::sentencepiece luojianet_ms::sentencepiece_train luojianet_ms::icuuc luojianet_ms::icudata
                luojianet_ms::icui18n)
    endif()
else()
    target_link_libraries(ut_tests PRIVATE luojianet_ms::gtest ${PYTHON_LIBRARIES})
endif()
if(USE_GLOG)
    target_link_libraries(ut_tests PRIVATE luojianet_ms::glog)
endif()

add_library(backend_static STATIC
        $<TARGET_OBJECTS:_luojianet_ms_debug_obj>
        $<TARGET_OBJECTS:_luojianet_ms_common_debug_obj>
        $<TARGET_OBJECTS:_luojianet_ms_profiler_obj>
        $<TARGET_OBJECTS:_luojianet_ms_ps_obj>
        $<TARGET_OBJECTS:_luojianet_ms_distributed_obj>
        $<TARGET_OBJECTS:_luojianet_ms_kernel_obj>
        $<TARGET_OBJECTS:_luojianet_ms_common_mem_reuse_obj>
        $<TARGET_OBJECTS:_luojianet_ms_backend_common_optimizer_obj>
        $<TARGET_OBJECTS:_luojianet_ms_backend_common_pass_obj>
        $<TARGET_OBJECTS:_luojianet_ms_backend_common_session_obj>
        $<TARGET_OBJECTS:_luojianet_ms_backend_common_somas_obj>
        $<TARGET_OBJECTS:_luojianet_ms_backend_graph_compiler_obj>
        $<TARGET_OBJECTS:_luojianet_ms_runtime_collective_obj>
        $<TARGET_OBJECTS:_luojianet_ms_runtime_device_obj>
        $<TARGET_OBJECTS:_luojianet_ms_runtime_graph_scheduler_obj>
        $<TARGET_OBJECTS:_luojianet_ms_runtime_hardware_obj>
        $<TARGET_OBJECTS:_luojianet_ms_runtime_pynative_obj>)
target_link_libraries(ut_tests PRIVATE luojianet_ms securec -Wl,--start-group proto_input luojianet_ms::protobuf
        backend_static -Wl,--end-group)
target_link_libraries(ut_tests PRIVATE luojianet_ms::grpc++)
