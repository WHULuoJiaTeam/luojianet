cmake_minimum_required(VERSION 3.14)
project (MetaDef[CXX])

set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR})

if (DEFINED ENV{D_PKG_SERVER})
    set(METADEF_PB_PKG $ENV{D_PKG_SERVER})
endif()

option(ENABLE_OPEN_SRC "Enable graphengine compile in opensource." FALSE)

if (ENABLE_OPEN_SRC)
    set(HI_PYTHON python3)
    include(cmake/external_libs/protobuf_shared.cmake)
    include(cmake/external_libs/protobuf_static.cmake)
    include(cmake/external_libs/protoc.cmake)
    include(cmake/external_libs/securec.cmake)
    include(cmake/external_libs/json.cmake)
    include(cmake/FindModule.cmake)
    include(cmake/intf_pub_linux.cmake)
endif()

set(METADEF_PROTO_LIST
    "${METADEF_DIR}/proto/task.proto"
    "${METADEF_DIR}/proto/om.proto"
    "${METADEF_DIR}/proto/ge_ir.proto"
    "${METADEF_DIR}/proto/insert_op.proto"
    "${METADEF_DIR}/proto/dump_task.proto"
    "${METADEF_DIR}/proto/fwk_adapter.proto"
    "${METADEF_DIR}/proto/op_mapping.proto"
    "${METADEF_DIR}/proto/onnx/ge_onnx.proto"
    "${METADEF_DIR}/proto/tensorflow/attr_value.proto"
    "${METADEF_DIR}/proto/tensorflow/function.proto"
    "${METADEF_DIR}/proto/tensorflow/graph.proto"
    "${METADEF_DIR}/proto/tensorflow/node_def.proto"
    "${METADEF_DIR}/proto/tensorflow/op_def.proto"
    "${METADEF_DIR}/proto/tensorflow/resource_handle.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor_shape.proto"
    "${METADEF_DIR}/proto/tensorflow/types.proto"
    "${METADEF_DIR}/proto/tensorflow/versions.proto"
    "${METADEF_DIR}/proto/var_manager.proto"
)

protobuf_generate(metadef_protos METADEF_PROTO_SRCS METADEF_PROTO_HDRS ${METADEF_PROTO_LIST} TARGET)

set(METADEF_GRAPH_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/om.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/ge_ir.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/insert_op.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/task.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/dump_task.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/fwk_adapter.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/op_mapping.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/onnx/ge_onnx.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/var_manager.pb.cc"
)

set(METADEF_TENSORFLOW_PROTO_SRCS
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/graph.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/node_def.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/function.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/versions.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/attr_value.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/op_def.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/tensor.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/tensor_shape.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/types.pb.cc"
    "${CMAKE_BINARY_DIR}/proto/metadef_protos/proto/tensorflow/resource_handle.pb.cc"
)

add_library(metadef_graph_protos_obj OBJECT ${METADEF_GRAPH_PROTO_SRCS})
add_dependencies(metadef_graph_protos_obj metadef_protos)
target_include_directories(metadef_graph_protos_obj PRIVATE
    #### blue zone ####
    ${PROTOBUF_SHARED_PKG_DIR}/include
    #### yellow zone ####
    ${ASCEND_PROTOBUF_SHARED_PKG_DIR}/include
)
target_compile_definitions(metadef_graph_protos_obj PRIVATE
    google=ascend_private
)
target_link_libraries(metadef_graph_protos_obj PRIVATE ascend_protobuf $<BUILD_INTERFACE:intf_pub>)
target_compile_options(metadef_graph_protos_obj PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

add_library(metadef_tensorflow_protos_obj OBJECT ${METADEF_TENSORFLOW_PROTO_SRCS})
add_dependencies(metadef_tensorflow_protos_obj metadef_protos)
target_include_directories(metadef_tensorflow_protos_obj PRIVATE
    #### blue zone ####
    ${PROTOBUF_SHARED_PKG_DIR}/include
    #### yellow zone ####
    ${ASCEND_PROTOBUF_SHARED_PKG_DIR}/include
)
target_compile_definitions(metadef_tensorflow_protos_obj PRIVATE
    google=ascend_private
)
target_link_libraries(metadef_tensorflow_protos_obj PRIVATE ascend_protobuf $<BUILD_INTERFACE:intf_pub>)
target_compile_options(metadef_tensorflow_protos_obj PRIVATE
    $<$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>:-O2 -fPIC>
    $<$<OR:$<STREQUAL:${PRODUCT_SIDE},host>,$<STREQUAL:${ENABLE_OPEN_SRC},True>>:-fexceptions>
    $<$<OR:$<STREQUAL:${TARGET_SYSTEM_NAME},Linux>,$<STREQUAL:${TARGET_SYSTEM_NAME},Android>>: -Wno-deprecated-declarations -fno-common>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Debug>>:/MTd>
    $<$<AND:$<STREQUAL:${TARGET_SYSTEM_NAME},Windows>,$<STREQUAL:${CMAKE_CONFIGURATION_TYPES},Release>>:/MT>
)

if (ENABLE_MS_TESTCASES)
    set(ENABLE_OPEN_SRC TRUE)
endif()
add_subdirectory(graph)
add_subdirectory(error_manager)
if (NOT ENABLE_D AND NOT ENABLE_ACL AND NOT ENABLE_MS_TESTCASES)
    add_subdirectory(register)
endif ()

if(ENABLE_GE_COV OR ENABLE_GE_UT)
    message(STATUS "Runing on llt mode, no need to depend other component")
elseif(ENABLE_METADEF_UT OR ENABLE_METADEF_COV)
    include(cmake/external_libs/gtest.cmake)
    include(cmake/init_test_depends.cmake)
    add_subdirectory(tests)
else()
    include(cmake/init_depends.cmake)
endif()

