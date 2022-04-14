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
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/backend/kernel_compiler/cpu)
MESSAGE("check  ut_test ${CMAKE_BINARY_DIR}")

link_directories(${MS_CCSRC_BUILD_PATH})

if(ENABLE_RS)
    remove_definitions(-DHAVE_STDLIB_H)
    if(WIN32)
        INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/include)
        INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/include/gdal)
        # message(FATAL_ERROR "${CMAKE_SOURCE_DIR}/third_party/GDAL/include/gdal")
        LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_win/lib)
    elseif(UNIX)
        INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/include)
        INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/include/gdal)
        # message(FATAL_ERROR "${CMAKE_SOURCE_DIR}/third_party/GDAL/include/gdal")
        LINK_DIRECTORIES(${CMAKE_SOURCE_DIR}/third_party/GDAL_linux/lib)
    endif()
endif()

if(ENABLE_MINDDATA)
    add_definitions(-DENABLE_MINDDATA)
    link_directories(${MS_CCSRC_BUILD_PATH}/minddata/dataset)
    link_directories(${MS_CCSRC_BUILD_PATH}/minddata/mindrecord)
endif()
# fetch ut test files
if(ENABLE_MINDDATA)
    include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset)
    include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/kernels/image)
    file(GLOB_RECURSE UT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            # ./stub/*.cc
            ./common/*.cc
            ./abstract/*.cc
            ./base/*.cc
            ./dataset/*.cc
            ./ir/dtype/*.cc
			${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/util/arena.cc
			${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/util/circular_pool.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/backend/kernel_compiler/cpu/nnacl/fp32/adam_fp32.c
			${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/backend/kernel_compiler/cpu/nnacl/fp32/adam_fp32.c
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/util.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/op_adapter.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/op_adapter_map.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/df_graph_manager.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/graph_runner.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/convert.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/op_adapter_util.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/*.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/op_declare/*.cc
			# ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/transform/graph_ir/op_declare/op_declare_macro.h
            # ${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/minddata/dataset/kernels/image/lite_cv/*.cc
            # ./device/*.cc
            ./ir/*.cc
            ./kernel/*.cc
            ./mindrecord/*.cc
            ./operator/*.cc
            ./optimizer/*.cc
            ./parallel/*.cc
            ./pipeline/*.cc
            # ./pre_activate/*.cc
            ./pynative/*.cc
            ./session/*.cc
            # ./transform/*.cc
            ./utils/*.cc
            ./vm/*.cc
            ./ps/*.cc
            ./fl/*.cc
            ./distributed/persistent/*.cc
            ./distributed/rpc/tcp/*.cc
            ./cxx_api/*.cc
            # ./tbe/*.cc
            ./mindapi/*.cc
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

file(GLOB_RECURSE LUOJIANET_GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
		"../../../luojianet_ms/runtime/device/gpu/gpu_memory_copy_manager.cc"
		# "../../../luojianet_ms/runtime/device/gpu/gpu_memory_copy_manager.cc"
		"../../../luojianet_ms/ccsrc/backend/optimizer/mem_reuse/mem_swap_manager.cc"
		"../../../luojianet_ms/ccsrc/backend/optimizer/mem_reuse/mem_reuse.cc"
		
		)

# file(GLOB_RECURSE LUOJIANET_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        # "../../../luojianet_ms/ccsrc/pybind_api/*.cc"
        # "../../../luojianet_ms/ccsrc/frontend/optimizer/*.cc"
        # "../../../luojianet_ms/ccsrc/frontend/parallel/*.cc"
        # "../../../luojianet_ms/ccsrc/frontend/operator/*.cc"
        # # dont remove the 4 lines above
        # "../../../luojianet_ms/ccsrc/debug/data_dump/dump_json_parser.cc"
        # "../../../luojianet_ms/ccsrc/debug/common.cc"
        # "../../../luojianet_ms/ccsrc/runtime/hccl_adapter/all_to_all_v_calc_param.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/kernel_runtime.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/memory_manager.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/memory_scheduler.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/memory_offload_strategy.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/kernel_runtime_manager.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/kernel_info.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/bucket.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/launch_kernel.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/profiling/*.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ge_runtime/*.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/kernel_select_ascend.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_launch_kernel.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_launch_mul.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_launch_atomic_clean.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_launch_transdata.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/kernel_select_graph_kernel.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/convert_tensor_utils.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_bucket.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_event.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/kernel_build_ascend.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_memory_adapter.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_memory_manager.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_device_address.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/ascend_memory_pool.cc"
        # "../../../luojianet_ms/ccsrc/runtime/device/ascend/lic_manager.cc"
        # "../../../luojianet_ms/ccsrc/runtime/hardware/ascend/ascend_device_context.cc"
        # "../../../luojianet_ms/ccsrc/runtime/hardware/ascend/ascend_graph_optimization.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/cpu_kernel.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/cpu_kernel_factory.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/sparse_apply_adam_cpu_kernel.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/sparse_apply_ftrl_cpu_kernel.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/sparse_apply_lazy_adam_cpu_kernel.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/sparse_apply_proximal_adagrad_cpu_kernel.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/unique_cpu_kernel.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/unique_with_pad_cpu_kernel.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/adam_delta_cpu_kernel.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/akg/*.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/rts/*.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/hccl/*.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/kernel_query.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/aicpu/aicpu_kernel_metadata.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/rts/rt_kernel_info.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/tbe/*.cc"
        # "../../../luojianet_ms/ccsrc/backend/optimizer/ascend/*.cc"
        # "../../../luojianet_ms/ccsrc/backend/optimizer/graph_kernel/*.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/anf_runtime_algorithm.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/ascend_session.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/ascend_auto_monad.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/ascend_control_parser.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/kernel_graph.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/session_basic.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/executor.cc"
        # "../../../luojianet_ms/core/ops/*.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/executor_manager.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/session_factory.cc"
        # "../../../luojianet_ms/ccsrc/backend/session/kernel_build_client.cc"
        # "../../../luojianet_ms/ccsrc/transform/graph_ir/*.cc"
        # "../../../luojianet_ms/ccsrc/transform/graph_ir/op_declare/*.cc"
        # "../../../luojianet_ms/ccsrc/ps/*.cc"
        # "../../../luojianet_ms/ccsrc/fl/*.cc"
        # "../../../luojianet_ms/ccsrc/distributed/persistent/*.cc"
        # "../../../luojianet_ms/ccsrc/distributed/rpc/tcp/*.cc"
        # "../../../luojianet_ms/ccsrc/profiler/device/ascend/*.cc"
        # "../../../luojianet_ms/ccsrc/profiler/device/profiling.cc"
        # "../../../luojianet_ms/ccsrc/backend/kernel_compiler/cpu/nnacl/fp32/adam_fp32.c"
        # )

# if(ENABLE_SECURITY)
    # file(GLOB_RECURSE _INFER_SUMMARY_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            # "../../../luojianet_ms/core/ops/*_summary.cc"
            # )
    # list(REMOVE_ITEM LUOJIANET_SRC_LIST ${_INFER_SUMMARY_FILES})
# endif()
# list(REMOVE_ITEM LUOJIANET_SRC_LIST
  # "../../../luojianet_ms/ccsrc/frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/util.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/scheduler.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/optimizer_info.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/optimizer_info_builder.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/worker.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/parameter_server.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/ps_cache/gpu/gpu_ps_cache.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/ps_cache/ascend/ascend_ps_cache.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/ps/ps_cache/ps_cache_manager.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/fl/server/kernel/sgd_kernel.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/fl/server/kernel/apply_momentum_kernel.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/optimizer/gpu/batch_norm_add_relu_fusion.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/optimizer/gpu/post_batch_norm_add_relu_fusion.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/optimizer/gpu/batch_norm_add_relu_grad_fusion.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/optimizer/gpu/batch_norm_relu_fusion.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/optimizer/gpu/batch_norm_relu_grad_fusion.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/optimizer/graph_kernel/lite_adapter/akg_build.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST
  # "../../../luojianet_ms/ccsrc/backend/optimizer/graph_kernel/lite_adapter/callback_impl.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/kernel_compiler/tbe/tbe_kernel_compile.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/kernel_compiler/akg/cpu/akg_cpu_kernel_mod.cc")
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/backend/kernel_compiler/akg/cpu/akg_cpu_kernel_build.cc")
# if(ENABLE_SECURITY)
    # list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/profiling.cc")
    # list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/ascend/memory_profiling.cc")
    # list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/ascend/ascend_profiling.cc")
    # list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/ascend/options.cc")
    # list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/debug/data_dump/dump_json_parser.cc")
# endif()
# list(REMOVE_ITEM LUOJIANET_SRC_LIST "../../../luojianet_ms/ccsrc/profiler/device/ascend/parallel_strategy_profiling.cc")

# add_library(_ut_luojianet_ms_obj OBJECT ${LUOJIANET_SRC_LIST})
# add_library(_ut_ut_obj OBJECT ${UT_SRCS})
# add_dependencies(_ut_luojianet_ms_obj graph)
# add_dependencies(_ut_ut_obj engine-cache-server graph)
# add_executable(ut_tests $<TARGET_OBJECTS:_ut_ut_obj>
        # $<TARGET_OBJECTS:_ut_luojianet_ms_obj>)

add_executable(ut_tests ${UT_SRCS} ${LUOJIANET_GPU_SRC_LIST})

# if(ENABLE_MPI)
	# target_link_libraries(luojianet_shared_lib nvidia_collective)
# endif()
if(ENABLE_GE)
    if(ENABLE_TRAIN)
        target_link_libraries(ut_tests PRIVATE graph ge_runner)
    else()
        target_link_libraries(ut_tests PRIVATE graph ge_client)
    endif()

    target_link_libraries(luojianet_ms PRIVATE tsdclient)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(ut_tests PRIVATE luojianet_ms::gtest luojianet_ms::event luojianet_ms::event_pthreads
                          luojianet_ms::event_openssl luojianet_gvar ${PYTHON_LIBRARIES} pthread util dl)
    if(ENABLE_MINDDATA)

        # AUX_SOURCE_DIRECTORY(LITE_CV_FILES)
        # message(STATUS "xxxxxxxxxxxxxxxxx"${LITE_CV_FILES} )
        # add_library(_live_cv OBJECT ${LITE_CV_FILES})

        target_link_libraries(ut_tests PRIVATE _c_dataengine _c_mindrecord)
    endif()
else()
    target_link_libraries(ut_tests PRIVATE luojianet_ms::gtest luojianet_gvar ${PYTHON_LIBRARIES})
endif()
if(USE_GLOG)
    target_link_libraries(ut_tests PRIVATE luojianet_ms::glog)
endif()

target_link_libraries(luojianet_ms luojianet_core)
if(ENABLE_GE)
    target_link_libraries(ut_tests PRIVATE luojianet_ms luojianet_shared_lib securec graph error_manager)
else()
    target_link_libraries(ut_tests PRIVATE luojianet_ms luojianet_shared_lib securec)
endif()

if(ENABLE_GPU)
	target_link_libraries(ut_tests PRIVATE gpu_cuda_lib gpu_queue)
endif()

if(ENALBE_CPU)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
    target_link_libraries(ut_tests PRIVATE  OpenMP::OpenMP_CXX)
endif()

#gdal
if(ENABLE_RS)
	if(UNIX)
		TARGET_LINK_LIBRARIES(ut_tests PRIVATE libgdal.so )
	elseif(MINGW)
		TARGET_LINK_LIBRARIES(ut_tests PRIVATE libgdal.dll.a)
	else()
		TARGET_LINK_LIBRARIES(ut_tests PRIVATE libgdal)
	endif()
endif()