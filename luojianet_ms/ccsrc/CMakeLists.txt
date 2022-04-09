## common setting
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/luojianet_ms/core/mindrt/src)

link_directories(${flatbuffers_LIBPATH})
#message(FATAL_ERROR "${flatbuffers_LIBPATH}")

set(SERVER_FLATBUFFER_OUTPUT "${CMAKE_BINARY_DIR}/schema")
set(FBS_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/../schema/cipher.fbs
        ${CMAKE_CURRENT_SOURCE_DIR}/../schema/fl_job.fbs
        )
ms_build_flatbuffers(FBS_FILES ${CMAKE_CURRENT_SOURCE_DIR}../../schema generated_fbs_files ${SERVER_FLATBUFFER_OUTPUT})

if(ENABLE_D)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/backend/kernel_compiler/aicpu/aicpu_ops)
    add_subdirectory(backend/kernel_compiler/aicpu/aicpu_ops)
endif()

if(ENABLE_CPU)
    if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
        set(PLATFORM_ARM64 "on")
        set(X86_64_SIMD "off")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(X86_64_SIMD "off")
    elseif("${X86_64_SIMD}" STREQUAL "off" AND NOT ${ENABLE_ASAN})
        set(X86_64_SIMD "avx")
    endif()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/backend/kernel_compiler/cpu)
    add_subdirectory(backend/kernel_compiler/cpu/nnacl)
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
    link_directories(${CMAKE_SOURCE_DIR}/build/luojianet_ms/graphengine)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 \
        -Wno-delete-non-abstract-non-virtual-dtor -Wno-unused-private-field -Wno-overloaded-virtual \
        -Wno-unused-const-variable -Wno-pessimizing-move -Wno-range-loop-analysis -Wno-mismatched-tags \
        -Wno-c++11-narrowing")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
    add_compile_definitions(BUILDING_DLL)
endif()

if(ENABLE_MPI)
    add_compile_definitions(ENABLE_MPI)
endif()

if(ENABLE_GPU)
    
   if(UNIX)
        SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    endif()

    find_package(CUDA REQUIRED)
    find_package(Threads)
    # if(${CUDA_VERSION} VERSION_LESS ${MS_REQUIRE_CUDA_VERSION})
        # message(FATAL_ERROR "The minimum CUDA version ${MS_REQUIRE_CUDA_VERSION} is required, \
              # but only CUDA ${CUDA_VERSION} found.")
    # endif()
    enable_language(CUDA)

    if(NOT CUDA_PATH OR CUDA_PATH STREQUAL "")
        if(DEFINED ENV{CUDA_HOME} AND NOT $ENV{CUDA_HOME} STREQUAL "")
            set(CUDA_PATH $ENV{CUDA_HOME})
        else()
            set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
        endif()
    endif()
   

    if(DEFINED ENV{CUDNN_HOME} AND NOT $ENV{CUDNN_HOME} STREQUAL "")
        set(CUDNN_INCLUDE_DIR $ENV{CUDNN_HOME}/include)
        set(CUDNN_LIBRARY_DIR $ENV{CUDNN_HOME}/lib64)
        find_path(CUDNN_INCLUDE_PATH cudnn.h HINTS ${CUDNN_INCLUDE_DIR} NO_DEFAULT_PATH)
        find_library(CUDNN_LIBRARY_PATH "cudnn" HINTS ${CUDNN_LIBRARY_DIR} NO_DEFAULT_PATH)
        if(CUDNN_INCLUDE_PATH STREQUAL CUDNN_INCLUDE_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn header file, please set environment variable CUDNN_HOME to \
                    cudnn installation position.")
        endif()
        if(CUDNN_LIBRARY_PATH STREQUAL CUDNN_LIBRARY_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn library file, please set environment variable CUDNN_HOME to \
                    cudnn installation position.")
        endif()
    else()
        list(APPEND CMAKE_PREFIX_PATH  ${CUDA_TOOLKIT_ROOT_DIR})
        find_path(CUDNN_INCLUDE_PATH cudnn.h PATH_SUFFIXES cuda/inclulde include cuda)
        find_library(CUDNN_LIBRARY_PATH "cudnn" PATH_SUFFIXES cuda/lib64 lib64 lib cuda/lib lib/x86_64-linux-gnu)
        if(CUDNN_INCLUDE_PATH STREQUAL CUDNN_INCLUDE_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn header file, if cudnn library is not installed, please put \
                    cudnn header file in cuda include path or user include path(eg. /usr/local/cuda/include; \
                    /usr/local/include; /usr/include), if cudnn library is installed in other position, please \
                    set environment variable CUDNN_HOME to cudnn installation position, there should be cudnn.h \
                    in {CUDNN_HOME}/include.")
        endif()
        if(CUDNN_LIBRARY_PATH STREQUAL CUDNN_LIBRARY_PATH-NOTFOUND)
            message(FATAL_ERROR "Failed to find cudnn library file, if cudnn library is not installed, please put \
                    cudnn library file in cuda library path or user library path(eg. /usr/local/cuda/lib64; \
                    /usr/local/lib64; /usr/lib64; /usr/local/lib; /usr/lib), if cudnn library is installed in other \
                    position, please set environment variable CUDNN_HOME to cudnn installation position, there should \
                    be cudnn library file in {CUDNN_HOME}/lib64.")
        endif()
    endif()

    if(NOT CUPTI_INCLUDE_DIRS OR CUPTI_INCLUDE_DIRS STREQUAL "")
        set(CUPTI_INCLUDE_DIRS  ${CUDA_PATH}/extras/CUPTI/include)
    endif()
    message("CUDA_PATH: ${CUDA_PATH}")
    message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message("CUDNN_INCLUDE_PATH: ${CUDNN_INCLUDE_PATH}")
    message("CUDNN_LIBRARY_PATH: ${CUDNN_LIBRARY_PATH}")
    message("CUPTI_INCLUDE_DIRS: ${CUPTI_INCLUDE_DIRS}")
    include_directories(${CUDNN_INCLUDE_PATH} ${CUDA_PATH} ${CUDA_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIRS})
     
    file(GLOB_RECURSE GPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "runtime/device/gpu/*.cc"
            "runtime/device/gpu/*.cu"
            # "backend/kernel_compiler/gpu/*.cu"
			"backend/kernel_compiler/gpu/*"
            )
    
	if(ENABLE_TESTCASES)
		if(ENABLE_MPI)
			set(NVIDIA_COLLECTIVE_SRCS 
									# "${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/runtime/hardware/gpu/nvidia_collective_comm_lib.cc"
									# "${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/runtime/hardware/gpu/nvidia_communication_group.cc"
									"${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/runtime/hardware/collective/collective_comm_lib_loader.cc"
									# "${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/runtime/hardware/collective/collective_communication_lib.cc"
									# "${CMAKE_SOURCE_DIR}/luojianet_ms/ccsrc/runtime/hardware/collective/communication_group.cc"
									)
		endif()
		file(GLOB_RECURSE GPU_SRC_LIST_APPEND RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
				# "backend/kernel_compiler/gpu/*"
				"runtime/hardware/gpu/gpu_device_context.cc"
				"backend/optimizer/gpu/*"
				"backend/optimizer/pass/*"
				"backend/optimizer/graph_kernel/*"
				"profiler/device/gpu/*"
				"utils/context/*"
				"backend/kernel_compiler/akg/gpu/*"
				"backend/kernel_compiler/akg/akg_kernel_build.cc"
				"backend/kernel_compiler/akg/akg_kernel_json_decoder.cc"
				"backend/kernel_compiler/akg/akg_kernel_json_generator.cc"
				"backend/kernel_compiler/akg/akg_kernel_metadata.cc"
				"runtime/device/launch_mul.cc"
				"runtime/device/bucket.cc"
				"runtime/device/convert_tensor_utils.cc"
				"runtime/device/kernel_adjust.cc"
				"runtime/device/kernel_info.cc"
				"runtime/device/kernel_runtime.cc"
				"runtime/device/kernel_runtime_manager.cc"
				"runtime/device/launch_kernel.cc"
				"runtime/device/memory_manager.cc"
				"runtime/device/memory_offload_strategy.cc"
				"runtime/device/memory_scheduler.cc"
				"runtime/device/tensor_array.cc"
				"backend/session/single_kernel_graph.cc"
				"backend/session/kernel_build_client.cc"
				"common/duplex_pipe.cc")
		list(APPEND GPU_SRC_LIST ${GPU_SRC_LIST_APPEND})
	endif()
	
    list(APPEND CUDA_NVCC_FLAGS -arch=sm_53 --expt-relaxed-constexpr)
    list(REMOVE_ITEM GPU_SRC_LIST "runtime/device/gpu/blocking_queue.cc" "runtime/device/gpu/gpu_buffer_mgr.cc")
    list(REMOVE_ITEM GPU_SRC_LIST "runtime/device/gpu/mpi/mpi_initializer.cc"
                                  "runtime/device/gpu/distribution/collective_wrapper.cc"
                                  "runtime/device/gpu/distribution/mpi_wrapper.cc"
                                  "runtime/device/gpu/distribution/nccl_wrapper.cc"
                                  "runtime/device/gpu/trt_loader.cc")
	list(REMOVE_ITEM GPU_SRC_LIST "backend/kernel_compiler/gpu/trt/trt_kernel.cc")

    if(NOT ${TENSORRT_HOME} STREQUAL "")
        find_path(TENSORRT_HOME_INCLUDE NvInfer.h HINTS ${TENSORRT_HOME}/include)
        if(TENSORRT_HOME_INCLUDE STREQUAL TENSORRT_HOME_INCLUDE-NOTFOUND)
          message(FATAL_ERROR "Tensor-RT dir not exist ${TENSORRT_HOME}")
        endif()
        message("Enable GPU inference. Tensor-RT include dir: ${TENSORRT_HOME_INCLUDE}")
        set(ENABLE_GPU_INFER TRUE)
        add_compile_definitions(ENABLE_GPU_INFER)
        include_directories(${TENSORRT_HOME_INCLUDE})
        list(APPEND GPU_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/runtime/device/gpu/trt_loader.cc)
    endif()

    set(NVCC_TMP_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    if(${CUDA_VERSION} VERSION_LESS 11.0)
        if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 9.0)
            string(REPLACE "-std=c++17" "-std=c++14" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        else()
            string(REPLACE "-std=c++17" "-std=c++11" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        endif()
    else()
        string(REPLACE "-std=c++17" "-std=c++14" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif()
    set_property(SOURCE ${GPU_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=luojianet_ms::SubModuleId::SM_DEVICE)
    cuda_add_library(gpu_cuda_lib STATIC ${GPU_SRC_LIST} ${NVIDIA_COLLECTIVE_SRCS})
	# target_link_libraries(gpu_cuda_lib nvidia_collective)
    set(CMAKE_CXX_FLAGS ${NVCC_TMP_CMAKE_CXX_FLAGS})
    add_compile_definitions(ENABLE_GPU)
endif()


## make protobuf files
file(GLOB ONNX_PROTO "" ${CMAKE_SOURCE_DIR}/third_party/proto/onnx/onnx.proto)
message("onnx proto path is :" ${ONNX_PROTO})
ms_protobuf_generate(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${ONNX_PROTO})
list(APPEND LUOJIANET_MS_PROTO_LIST ${ONNX_PROTO_SRCS})

include_directories("${CMAKE_BINARY_DIR}/ps/core")
file(GLOB_RECURSE COMM_PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ps/core/protos/*.proto")
ms_protobuf_generate(COMM_PROTO_SRCS COMM_PROTO_HDRS ${COMM_PROTO_IN})
list(APPEND LUOJIANET_MS_PROTO_LIST ${COMM_PROTO_SRCS})
if(NOT ENABLE_SECURITY)
    include_directories("${CMAKE_BINARY_DIR}/profiler/device/ascend")
    file(GLOB_RECURSE PROFILER_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      "profiler/device/ascend/memory_profiling.proto")
    ms_protobuf_generate_py(PROFILER_MEM_PROTO_PY PROFILER_MEM_PROTO_HDRS_PY PROFILER_MEM_PROTO_PYS
      ${PROFILER_PROTO_LIST})
    list(APPEND LUOJIANET_MS_PROTO_LIST ${PROFILER_MEM_PROTO_PY})
endif()

if(ENABLE_DEBUGGER)
    # debugger: compile proto files
    include_directories("${CMAKE_BINARY_DIR}/debug/debugger")
    file(GLOB_RECURSE DEBUGGER_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/debugger/debug_graph.proto")
    ms_protobuf_generate(DEBUGGER_PROTO_SRCS DEBUGGER_PROTO_HDRS ${DEBUGGER_PROTO_LIST})
    file(GLOB_RECURSE DEBUGGER_GRPC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/debugger/debug_grpc.proto")
    ms_grpc_generate(DEBUGGER_GRPC_SRCS DEBUGGER_GRPC_HDRS ${DEBUGGER_GRPC_LIST})
    file(GLOB_RECURSE DUMP_DATA_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/debugger/dump_data.proto")
    ms_protobuf_generate(DUMP_DATA_PROTO_SRCS DUMP_DATA_PROTO_HDRS ${DUMP_DATA_PROTO_LIST})
    list(APPEND LUOJIANET_MS_PROTO_LIST ${DEBUGGER_PROTO_SRCS})
    list(APPEND LUOJIANET_MS_PROTO_LIST ${DEBUGGER_GRPC_SRCS})
    list(APPEND LUOJIANET_MS_PROTO_LIST ${DUMP_DATA_PROTO_SRCS})
endif()

if(ENABLE_DUMP_PROTO)
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB_RECURSE PROTO_PY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "utils/anf_ir.proto"
        "utils/summary.proto"
        "utils/lineage.proto"
        "utils/checkpoint.proto"
        "utils/print.proto"
        "utils/node_strategy.proto"
        "utils/profiling_parallel.proto"
    )
    ms_protobuf_generate_py(PY_SRCS PY_HDRS PY_PYS ${PROTO_PY})

    list(APPEND LUOJIANET_MS_PROTO_LIST ${PROTO_SRCS})
    list(APPEND LUOJIANET_MS_PROTO_LIST ${PY_SRCS})
endif()

if(MODE_ASCEND_ALL)
    include_directories("${CMAKE_BINARY_DIR}/backend/kernel_compiler/aicpu")
    file(GLOB_RECURSE PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "backend/kernel_compiler/aicpu/proto/*.proto")
    ms_protobuf_generate(PROTOSRCS PROTOHDRS ${PROTO_IN})

    file(GLOB_RECURSE PROTO_DUMP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "runtime/device/ascend/dump/proto/*.proto")
    ms_protobuf_generate(DUMP_PROTOSRCS PROTOHDRS ${PROTO_DUMP})

    list(APPEND LUOJIANET_MS_PROTO_LIST ${PROTOSRCS})
    list(APPEND LUOJIANET_MS_PROTO_LIST ${PREDICT_PROTOSRCS})
    list(APPEND LUOJIANET_MS_PROTO_LIST ${DUMP_PROTOSRCS})

    add_compile_definitions(ENABLE_D)
endif()

if(LUOJIANET_MS_PROTO_LIST)
    add_library(proto_input STATIC ${LUOJIANET_MS_PROTO_LIST})
    set_target_properties(proto_input PROPERTIES COMPILE_FLAGS "-Wno-unused-variable")
    target_compile_options(proto_input PRIVATE "-Wno-array-bounds")
endif()

## make sub objects
set(SUB_COMP
        transform/graph_ir
        transform/express_ir
        backend/optimizer
        backend/kernel_compiler
        backend/session
        runtime/device
        runtime/framework
        runtime/hardware
        runtime/op_builder
        runtime/hccl_adapter
        frontend/optimizer
        frontend/parallel
        frontend/operator
        pipeline/jit
        pipeline/pynative
        common debug pybind_api utils vm profiler ps fl distributed
)

foreach(_comp ${SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _luojianet_${sub}_obj)
        list(APPEND SUB_OBJECTS_SRC $<TARGET_OBJECTS:_luojianet_${sub}_obj>)
        add_dependencies(_luojianet_${sub}_obj proto_input luojianet_core)
    endif()
endforeach()

set_property(SOURCE ${SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=luojianet_ms::SubModuleId::SM_ME)
# if(ENABLE_TESTCASES)
	# add_library(luojianet_ms STATIC ${SUB_OBJECTS_SRC} ${NVIDIA_COLLECTIVE_SRCS}) 
# else()
add_library(luojianet_ms STATIC ${SUB_OBJECTS_SRC})
# endif()

target_link_libraries(luojianet_ms luojianet_core)

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        target_link_libraries(luojianet_ms luojianet_ms::grpcpp)
    else()
        target_link_libraries(luojianet_ms -Wl,--no-as-needed luojianet_ms::grpcpp)
    endif()
endif()

target_link_libraries(luojianet_ms securec luojianet_ms::flatbuffers)

if(NOT WIN32)
  target_link_libraries(luojianet_ms dl)
endif()

if(MODE_ASCEND_ALL OR MODE_ASCEND_ACL)
    # common env paths
    if(DEFINED ENV{D_LINK_PATH})
        if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
            MESSAGE("system processor matches aarch64")
            set(D_LIB_PATH $ENV{D_LINK_PATH}/aarch64)
        elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
            MESSAGE("system processor matches x86_64")
            set(D_LIB_PATH $ENV{D_LINK_PATH}/x86_64)
        else()
            MESSAGE("system ${CMAKE_HOST_SYSTEM_PROCESSOR} not support")
        endif()
    else()
        MESSAGE("use system default lib")
        if(DEFINED ENV{ASCEND_CUSTOM_PATH})
            set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
        else()
            set(ASCEND_PATH /usr/local/Ascend)
        endif()
        set(ASCEND_DRIVER_PATH ${ASCEND_PATH}/driver/lib64/common)
        set(ASCEND_DRIVER_BACK_PATH ${ASCEND_PATH}/driver/lib64/driver)
        set(ASCEND_RUNTIME_PATH ${ASCEND_PATH}/fwkacllib/lib64)
        set(ASCEND_RUNTIME_NEW_PATH ${ASCEND_PATH}/lib64)
        set(ASCEND_OPP_PATH ${ASCEND_PATH}/opp/op_impl/built-in/ai_core/tbe/op_tiling)
        set(ASCEND_TOOLKIT_RUNTIME_PATH ${ASCEND_PATH}/ascend-toolkit/latest/fwkacllib/lib64)
        set(ASCEND_TOOLKIT_RUNTIME_NEW_PATH ${ASCEND_PATH}/ascend-toolkit/latest/lib64)
        set(ASCEND_TOOLKIT_OPP_PATH ${ASCEND_PATH}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
    endif()
endif()

if(ENABLE_GE)
    find_library(GE_RUNNER ge_runner ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GRAPH graph ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(HCCL hccl ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(luojianet_ms ${GE_RUNNER} ${GRAPH} ${HCCL})
endif()

if(MODE_ASCEND_ALL)
    MESSAGE("USE DAV LIB PATH: ${ASCEND_PATH}")
    find_library(ERROR_MANAGER error_manager ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH})
    find_library(RUNTIME_LIB runtime ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH})
    find_library(TSDCLIENT tsdclient HINTS ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH} ${ASCEND_DRIVER_BACK_PATH})
    find_library(DATATRANSFER datatransfer HINTS ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH} ${ASCEND_DRIVER_BACK_PATH})
    find_library(PROFILING msprofiler ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH})
    find_library(ACL ascendcl ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH})
    find_library(ACL_TDT_CHANNEL acl_tdt_channel ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH})
    find_library(PLATFORM platform ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH})
    find_library(OPTILING optiling ${ASCEND_OPP_PATH} ${ASCEND_TOOLKIT_OPP_PATH})
    find_library(OPT_FEATURE opt_feature ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH})

    target_link_libraries(luojianet_ms ${RUNTIME_LIB} ${TSDCLIENT} ${DATATRANSFER} ${ERROR_MANAGER} -Wl,--no-as-needed
      ${OPTILING} ${PLATFORM} ${ACL} ${ACL_TDT_CHANNEL} ${OPT_FEATURE} ${PROFILING})
    target_link_libraries(luojianet_ms -Wl,--start-group proto_input luojianet_ms::protobuf -Wl,--end-group)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(luojianet_ms -Wl,--start-group proto_input luojianet_ms::protobuf luojianet_ms::sentencepiece
                        -Wl,--end-group)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(luojianet_ms -Wl proto_input luojianet_ms::protobuf luojianet_ms::sentencepiece -Wl)
else()
    target_link_libraries(luojianet_ms -Wl,--start-group proto_input luojianet_ms::protobuf -Wl,--end-group)
endif()

# set c_expression building
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set_property(SOURCE "pipeline/jit/init.cc" PROPERTY
            COMPILE_DEFINITIONS SUBMODULE_ID=luojianet_ms::SubModuleId::SM_PIPELINE)
pybind11_add_module(_c_expression NO_EXTRAS "pipeline/jit/init.cc" NO_EXTRAS)

MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(_c_expression PRIVATE -Wl,-init,luojianet_log_init)
    set(ORIGIN_PATH $ORIGIN)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(_c_expression PROPERTIES MACOSX_RPATH ON)
    set(ORIGIN_PATH @loader_path)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ORIGIN_PATH $ORIGIN)
else()
    MESSAGE(FATAL_ERROR "other platform: ${CMAKE_SYSTEM_NAME}")
endif()

if(MODE_ASCEND_ALL)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/nnae/latest/fwkacllib/lib64)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/nnae/latest/lib64)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/lib64)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/lib64)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/fwkacllib/lib64)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/lib64)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/op_tiling)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
    set(LUOJIANET_MS_RPATH
        ${LUOJIANET_MS_RPATH}:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling)
elseif(ENABLE_GPU)
    set(LUOJIANET_MS_RPATH ${LUOJIANET_MS_RPATH}:${CUDA_PATH}/lib64)
endif()
set(LUOJIANET_MS_RPATH ${ORIGIN_PATH}/lib:${LUOJIANET_MS_RPATH})
set_target_properties(_c_expression PROPERTIES INSTALL_RPATH ${LUOJIANET_MS_RPATH})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(luojianet_ms luojianet_ms::pybind11_module)
    target_link_libraries(luojianet_ms luojianet_gvar)
    target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive luojianet_ms luojianet_core -Wl,--no-whole-archive)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(luojianet_ms proto_input luojianet_ms::protobuf
        luojianet_ms::event luojianet_ms::event_pthreads luojianet_ms::event_openssl luojianet_ms::eigen luojianet_ms::json)
    target_link_libraries(luojianet_ms luojianet_ms::event_core ps_cache)
    target_link_libraries(_c_expression PRIVATE -Wl,-all_load luojianet_ms luojianet_core proto_input -Wl,-noall_load)
    target_link_libraries(_c_expression PRIVATE luojianet_ms::pybind11_module)
    target_link_libraries(_c_expression PRIVATE luojianet_gvar)
else()
    if(ENABLE_CPU AND NOT WIN32)
        target_link_libraries(luojianet_ms proto_input luojianet_ms::protobuf
            luojianet_ms::event luojianet_ms::event_pthreads luojianet_ms::event_openssl luojianet_ms::eigen luojianet_ms::json)
        target_link_libraries(luojianet_ms -Wl,--no-as-needed luojianet_ms::event_core ps_cache)
        if(${ENABLE_IBVERBS} STREQUAL "ON")
            target_link_libraries(luojianet_ms ibverbs rdmacm)
        endif()
    endif()
    target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive luojianet_ms luojianet_core
        proto_input -Wl,--no-whole-archive)
    target_link_libraries(_c_expression PRIVATE luojianet_ms::pybind11_module)
    target_link_libraries(_c_expression PRIVATE luojianet_gvar)
endif()

if(USE_GLOG)
    target_link_libraries(_c_expression PRIVATE luojianet_ms::glog)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(luojianet_ms luojianet_ms::crypto luojianet_ms::ssl)
endif()

if(ENABLE_GPU)
    message("add gpu lib to c_expression")
    target_link_libraries(_c_expression PRIVATE gpu_cuda_lib gpu_queue cublas
                          ${CUDA_PATH}/lib64/libcurand.so
                          ${CUDNN_LIBRARY_PATH}
                          ${CUDA_PATH}/lib64/libcudart.so
                          ${CUDA_PATH}/lib64/stubs/libcuda.so
                          ${CUDA_PATH}/lib64/libcusolver.so
                          ${CUDA_PATH}/lib64/libcufft.so)
	# target_link_libraries(gpu_cuda_lib $<TARGET_OBJECTS:_luojianet_runtime_hardware_obj>)
	
	target_link_libraries(luojianet_ms gpu_cuda_lib gpu_queue cublas
                          ${CUDA_PATH}/lib64/libcurand.so
                          ${CUDNN_LIBRARY_PATH}
                          ${CUDA_PATH}/lib64/libcudart.so
                          ${CUDA_PATH}/lib64/stubs/libcuda.so
                          ${CUDA_PATH}/lib64/libcusolver.so
                          ${CUDA_PATH}/lib64/libcufft.so)
	
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        target_link_libraries(_c_expression PRIVATE luojianet_ms::crypto luojianet_ms::ssl)
    endif()
    if(ENABLE_MPI)
        set_target_properties(_ms_mpi PROPERTIES INSTALL_RPATH ${LUOJIANET_MS_RPATH})
        if(ENABLE_GPU)
			target_link_libraries(luojianet_ms nvidia_collective)
		endif()
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
    set_target_properties(_c_expression PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
endif()

if(ENABLE_CPU)
    target_link_libraries(_c_expression PRIVATE luojianet_ms::dnnl luojianet_ms::mkldnn nnacl)
endif()

if(ENABLE_MINDDATA)
    add_subdirectory(minddata/mindrecord)
    add_subdirectory(minddata/dataset)
endif()

if(MODE_ASCEND_ALL)
    find_library(adump_server libadump_server.a ${ASCEND_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH}
                ${ASCEND_RUNTIME_NEW_PATH} ${ASCEND_TOOLKIT_RUNTIME_NEW_PATH})
    target_link_libraries(_c_expression PRIVATE ${adump_server})
endif()

if(ENABLE_D)
    if(ENABLE_MPI)
        set_target_properties(_ascend_mpi PROPERTIES INSTALL_RPATH ${LUOJIANET_MS_RPATH})
    endif()
endif()

add_subdirectory(cxx_api)
