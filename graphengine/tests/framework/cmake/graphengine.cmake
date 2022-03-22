# ---- Test coverage ----

if (ENABLE_GE_COV)
    set(COVERAGE_COMPILER_FLAGS "-g --coverage -fprofile-arcs -fPIC -O0 -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${COVERAGE_COMPILER_FLAGS}")
endif()

# ----metadef Proto generate ----
set(PROTO_LIST
        "${GE_CODE_DIR}/metadef/proto/om.proto"
        "${GE_CODE_DIR}/metadef/proto/ge_ir.proto"
        "${GE_CODE_DIR}/metadef/proto/insert_op.proto"
        "${GE_CODE_DIR}/metadef/proto/task.proto"
        "${GE_CODE_DIR}/metadef/proto/dump_task.proto"
        "${GE_CODE_DIR}/metadef/proto/fwk_adapter.proto"
        "${GE_CODE_DIR}/metadef/proto/op_mapping.proto"
        "${GE_CODE_DIR}/metadef/proto/ge_api.proto"
        "${GE_CODE_DIR}/metadef/proto/optimizer_priority.proto"
        "${GE_CODE_DIR}/metadef/proto/onnx/ge_onnx.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/attr_value.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/function.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/graph.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/node_def.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/op_def.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/resource_handle.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/tensor.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/tensor_shape.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/types.proto"
        "${GE_CODE_DIR}/metadef/proto/tensorflow/versions.proto"
        )

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# ---- File glob by group ----

file(GLOB_RECURSE METADEF_SRCS CONFIGURE_DEPENDS
    "${GE_CODE_DIR}/metadef/graph/*.cc"
    "${GE_CODE_DIR}/metadef/register/*.cc"
    "${GE_CODE_DIR}/metadef/register/*.cpp"
    "${GE_CODE_DIR}/metadef/ops/*.cc"
    "${GE_CODE_DIR}/metadef/third_party/transformer/src/*.cc"
)
file(GLOB_RECURSE METADEF_REGISTER_SRCS CONFIGURE_DEPENDS
        "${GE_CODE_DIR}/metadef/register/*.cc"
        "${GE_CODE_DIR}/metadef/register/*.cpp"
)

file(GLOB_RECURSE PARSER_SRCS CONFIGURE_DEPENDS 
    "${GE_CODE_DIR}/parser/parser/common/*.cc"
)

file(GLOB_RECURSE LOCAL_ENGINE_SRC CONFIGURE_DEPENDS
    "${GE_CODE_DIR}/ge/ge_local_engine/*.cc"
)
list(REMOVE_ITEM LOCAL_ENGINE_SRC
        "${GE_CODE_DIR}/ge/ge_local_engine/engine/host_cpu_engine.cc")


file(GLOB_RECURSE HOST_ENGINE_SRC CONFIGURE_DEPENDS
    "${GE_CODE_DIR}/ge/host_cpu_engine/*.cc"
)

file(GLOB_RECURSE NN_ENGINE_SRC CONFIGURE_DEPENDS
        "${GE_CODE_DIR}/ge/plugin/*.cc"
)

file(GLOB_RECURSE OFFLINE_SRC CONFIGURE_DEPENDS
    "${GE_CODE_DIR}/ge/offline/*.cc"
)

file(GLOB_RECURSE GE_SRCS CONFIGURE_DEPENDS
    "${GE_CODE_DIR}/ge/*.cc"
)
file(GLOB_RECURSE GE_SUB_ENGINE_SRCS CONFIGURE_DEPENDS
        "${GE_CODE_DIR}/ge/ge_local_engine/engine/host_cpu_engine.cc"
        )

list(REMOVE_ITEM GE_SRCS ${LOCAL_ENGINE_SRC} ${HOST_ENGINE_SRC} ${NN_ENGINE_SRC} ${OFFLINE_SRC})
list(APPEND GE_SRCS ${GE_SUB_ENGINE_SRCS})

list(APPEND INCLUDE_DIRECTORIES
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${GE_CODE_DIR}"
    "${GE_CODE_DIR}/inc"
    "${GE_CODE_DIR}/metadef/inc"
    "${GE_CODE_DIR}/ge"
    "${GE_CODE_DIR}/ge/inc"
    "${GE_CODE_DIR}/ge/ir_build"
    "${GE_CODE_DIR}/metadef"
    "${GE_CODE_DIR}/metadef/graph"
    "${GE_CODE_DIR}/inc/external"
    "${GE_CODE_DIR}/inc/framework/common"
    "${GE_CODE_DIR}/metadef/inc/external"
    "${GE_CODE_DIR}/metadef/inc/external/graph"
    "${GE_CODE_DIR}/metadef/inc/graph"
    "${GE_CODE_DIR}/inc/framework"
    "${GE_CODE_DIR}/metadef/inc/common"
    "${GE_CODE_DIR}/metadef/third_party"
    "${GE_CODE_DIR}/metadef/third_party/transformer/inc"
    "${GE_CODE_DIR}/parser"
    "${GE_CODE_DIR}/parser/parser"
    "${GE_CODE_DIR}/third_party/fwkacllib/inc"
    "${GE_CODE_DIR}/third_party/fwkacllib/inc/cce"
    "${GE_CODE_DIR}/third_party/fwkacllib/inc/ops"
    "${GE_CODE_DIR}/third_party/fwkacllib/inc/toolchain"
    "${GE_CODE_DIR}/third_party/fwkacllib/inc/opt_info"
    "${GE_CODE_DIR}/tests/ut/ge"
    "${GE_CODE_DIR}/tests/ut/common"
    "${CMAKE_BINARY_DIR}"
    "${CMAKE_BINARY_DIR}/proto/ge"
    "${CMAKE_BINARY_DIR}/proto/ge/proto"
)

list(APPEND STUB_LIBS
    c_sec
    slog_stub
    cce_ge_stub
    runtime_stub
    profiler_stub
    hccl_stub
    opt_feature_stub
    error_manager_stub
    ascend_protobuf
    json
)

# ---- Target : metadef graph ----

add_library(metadef_graph SHARED ${METADEF_SRCS} ${PROTO_SRCS})

target_include_directories(metadef_graph
        PUBLIC
        "${INCLUDE_DIRECTORIES}"
        )

target_compile_definitions(metadef_graph PRIVATE
        google=ascend_private
        FMK_SUPPORT_DUMP
        )

target_compile_options(metadef_graph PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -Werror=format
        )

target_link_libraries(metadef_graph PUBLIC
        $<BUILD_INTERFACE:intf_pub> ${STUB_LIBS}
        mmpa -L${GE_CODE_DIR}/third_party/prebuild/x86_64 -lrt -ldl -lpthread -lgcov
        )

set_target_properties(metadef_graph PROPERTIES CXX_STANDARD 11)

# ---- Target : Local engine ----

add_library(ge_local_engine SHARED ${LOCAL_ENGINE_SRC})

target_include_directories(ge_local_engine
        PUBLIC
        "${INCLUDE_DIRECTORIES}"
        "${GE_CODE_DIR}/ge/ge_local_engine"
        )

target_compile_definitions(ge_local_engine PRIVATE
        google=ascend_private
        )

target_compile_options(ge_local_engine PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -Werror=format
        )

target_link_libraries(ge_local_engine PUBLIC
        $<BUILD_INTERFACE:intf_pub> ${STUB_LIBS}
        -lrt -ldl -lpthread -lgcov
        )

set_target_properties(ge_local_engine PROPERTIES CXX_STANDARD 11)

# ---- Target : engine plugin----
#

add_library(nnengine SHARED ${NN_ENGINE_SRC})

target_include_directories(nnengine
        PUBLIC
        "${INCLUDE_DIRECTORIES}"
        "${GE_CODE_DIR}/ge/plugin/engine"
)

target_compile_definitions(nnengine PRIVATE
        google=ascend_private
)

target_compile_options(nnengine PRIVATE
        -g --coverage -fprofile-arcs -ftest-coverage
        -Werror=format
)

target_link_libraries(nnengine PUBLIC
        $<BUILD_INTERFACE:intf_pub> ${STUB_LIBS} -lrt -ldl -lpthread -lgcov
)

set_target_properties(nnengine PROPERTIES CXX_STANDARD 11)

#   Targe: engine_conf
add_custom_target(
        engine_conf.json ALL
        DEPENDS ${CMAKE_BINARY_DIR}/engine_conf.json
)
add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/engine_conf.json
        COMMAND cp ${GE_CODE_DIR}/ge/engine_manager/engine_conf.json ${CMAKE_BINARY_DIR}/
)
#   Targe: optimizer priority
add_custom_target(
        optimizer_priority.pbtxt ALL
        DEPENDS ${CMAKE_BINARY_DIR}/optimizer_priority.pbtxt
)
add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/optimizer_priority.pbtxt
        COMMAND cp ${GE_CODE_DIR}/ge/opskernel_manager/optimizer_priority.pbtxt ${CMAKE_BINARY_DIR}/
)

# ---- Target : Graph engine ----

add_library(graphengine STATIC ${PARSER_SRCS} ${GE_SRCS})

target_include_directories(graphengine
    PUBLIC 
    "${INCLUDE_DIRECTORIES}"
    "${GE_CODE_DIR}/ge/host_cpu_engine"
)

target_compile_definitions(graphengine PRIVATE
    google=ascend_private
    FMK_SUPPORT_DUMP
)

target_compile_options(graphengine PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(graphengine PUBLIC
    $<BUILD_INTERFACE:intf_pub> ${STUB_LIBS}
        metadef_graph
         -lrt -ldl -lpthread -lgcov
)

set_target_properties(graphengine PROPERTIES CXX_STANDARD 11)
add_dependencies(graphengine ge_local_engine nnengine engine_conf.json optimizer_priority.pbtxt)
