# Copyright 2019-2020 Huawei Technologies Co., Ltd
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

project(ut_graph)

set(CMAKE_CXX_STANDARD 11)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${METADEF_DIR}/graph)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
include_directories(${METADEF_DIR}/third_party)
include_directories(${METADEF_DIR}/third_party/graphengine/inc)
include_directories(${METADEF_DIR}/third_party/graphengine/inc/external)
include_directories(${METADEF_DIR}/third_party/graphengine/inc/external/ge)
include_directories(${METADEF_DIR}/third_party/fwkacllib/inc)
include_directories(${METADEF_DIR}/third_party/transformer/inc)
include_directories(${METADEF_DIR}/)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

set(UT_FILES
    "testcase/graph_unittest.cc"
    "testcase/ge_tensor_unittest.cc"
    "testcase/types_unittest.cc"
    "testcase/tensor_unittest.cc"
    "testcase/type_utils_unittest.cc"
    "testcase/aligned_ptr_unittest.cc"
    "testcase/object_pool_unittest.cc"
    "testcase/tensor_ut.cc"
    "testcase/node_utils_unittest.cc"
    "testcase/node_unittest.cc"
    "testcase/op_desc_utils_unittest.cc"
    "testcase/graph_builder_utils.cc"
    "testcase/operator_unittest.cc"
    "testcase/buffer_unittest.cc"
    "testcase/ge_attr_value_unittest.cc"
    "testcase/shape_refiner_unittest.cc"
    "testcase/graph_utils_unittest.cc"
    "testcase/file_utils_unittest.cc"
    "testcase/ffts_graph_utils_unittest.cc"
    "testcase/runtime_inference_context_unittest.cc"
    "testcase/op_desc_unittest.cc"
    "testcase/ref_relation_unittest.cc"
    "testcase/gnode_unittest.cc"
    "testcase/resource_context_mgr_unittest.cc"
    "testcase/inference_context_unittest.cc"
    "testcase/constant_utils_unittest.cc"
    "testcase/any_value_ut.cc"
    "testcase/ascend_string_unittest.cc"
    "testcase/attr_store_ut.cc"
    "testcase/attr_serializer_unittest.cc"
    "testcase/attr_serializer_registry_unittest.cc"
    "testcase/small_vector_ut.cc"
    "testcase/attr_utils_unittest.cc"
    "testcase/ge_ir_utils_unittest.cc"
    "testcase/func_counter.cc"
    "testcase/test_std_structs.cc"
    "testcase/transformer_expand_dims_ut.cc"
    "testcase/transformer_transfer_shape_ut.cc"
    "testcase/op_imp_unittest.cc"
    "testcase/operator_constuct_graph_unittest.cc"
    "testcase/format_refiner_unittes.cc"
    "testcase/model_serialize_unittest.cc"
    "testcase/ref_relation_unittes.cc"
    "testcase/tensor_utils_unittest.cc"
    "testcase/compile_cache_policy_hasher_unittest.cc"
    "testcase/compile_cache_policy_unittest.cc"
    "testcase/profiler_unittest.cc"
    "testcase/ge_graph_dumper_unittest.cc"
    "testcase/file_utils_unittest.cc"
    "testcase/ge_local_context_unittest.cc"
    "testcase/anchor_unittest.cc"
    "testcase/model_unittest.cc"
    "testcase/ge_context_unittest.cc"
    "testcase/anchor_utils_unittest.cc"
    "testcase/attributes_holder_unittest.cc"
)

add_executable(ut_graph ${UT_FILES})

target_compile_options(ut_graph PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_graph PRIVATE
    google=ascend_private
)

target_link_libraries(ut_graph
    $<BUILD_INTERFACE:intf_pub>
    graph_static
    gtest gtest_main slog_stub ascend_protobuf c_sec error_manager_stub mmpa_stub -lrt -ldl -lgcov
)
