# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

project(ut_libgraph)

set(CMAKE_CXX_STANDARD 11)

set(PROTO_LIST
    "${GE_CODE_DIR}/metadef/proto/om.proto"
    "${GE_CODE_DIR}/metadef/proto/ge_ir.proto"
    "${GE_CODE_DIR}/metadef/proto/onnx/ge_onnx.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${GE_CODE_DIR})
include_directories(${GE_CODE_DIR}/metadef)
include_directories(${GE_CODE_DIR}/metadef/graph)
include_directories(${GE_CODE_DIR}/inc)
include_directories(${GE_CODE_DIR}/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external)
include_directories(${GE_CODE_DIR}/metadef/inc/external/graph)
include_directories(${GE_CODE_DIR}/metadef/inc)
include_directories(${GE_CODE_DIR}/metadef/inc/graph)
include_directories(${GE_CODE_DIR}/metadef/inc/common)
include_directories(${GE_CODE_DIR}/metadef/third_party)
include_directories(${GE_CODE_DIR}/metadef/third_party/transformer/inc)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc)
include_directories(${GE_CODE_DIR}/third_party/fwkacllib/inc/ops)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

set(UT_FILES
    "testcase/ge_graph/ge_anchor_utils_unittest.cc"
    "testcase/ge_graph/ge_def_type_unittest.cc"
    "testcase/ge_graph/ge_graph_anchor_unittest.cc"
    "testcase/ge_graph/ge_model_serialize_unittest.cc"
    "testcase/ge_graph/ge_node_unittest.cc"
    "testcase/ge_graph/ge_opdesc_unittest.cc"
    "testcase/ge_graph/ge_tensor_unittest.cc"
    "testcase/ge_graph/graph_builder_utils.cc"
    "testcase/ge_graph/ge_reg_unittest.cc"
    "testcase/ge_graph/ge_format_refiner_unittest.cc"
    "testcase/ge_graph/ge_opsproto_manager_unittest.cc"
    "testcase/ge_graph/ge_operator_unittest.cc"
    "testcase/ge_graph/ge_model_unittest.cc"
)

FILE(GLOB_RECURSE GRAPH_SRC_FILES_DEPTH0 ${GE_CODE_DIR}/metadef/graph/*.cc)
FILE(GLOB_RECURSE GRAPH_SRC_FILES_DEPTH1 ${GE_CODE_DIR}/metadef/graph/*/*.cc)
FILE(GLOB_RECURSE GRAPH_SRC_FILES_DEPTH2 ${GE_CODE_DIR}/metadef/graph/*/*/*.cc)

AUX_SOURCE_DIRECTORY(${GE_CODE_DIR}/metadef/ops GRAPH_OPS_SRC_FILES)
AUX_SOURCE_DIRECTORY(${GE_CODE_DIR}/metadef/third_party/transformer/src TRANSFORMER_SRC_FILES)

add_executable(ut_libgraph ${UT_FILES}
    ${GRAPH_SRC_FILES_DEPTH0}
    ${GRAPH_SRC_FILES_DEPTH1}
    ${GRAPH_SRC_FILES_DEPTH2}
    ${GRAPH_OPS_SRC_FILES}
    ${TRANSFORMER_SRC_FILES}
    ${PROTO_SRCS} ${PROTO_HDRS}
)

target_compile_options(ut_libgraph PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_libgraph PRIVATE
    google=ascend_private
)

target_link_libraries(ut_libgraph
    $<BUILD_INTERFACE:intf_pub>
    gtest
    gtest_main
    slog_stub
    ascend_protobuf
    c_sec
    error_manager_stub
    mmpa_stub
    -lrt
    -ldl
    -lgcov
)
