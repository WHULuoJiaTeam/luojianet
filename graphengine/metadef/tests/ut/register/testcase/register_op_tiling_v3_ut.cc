/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#define private public
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "op_tiling.h"

using namespace std;
using namespace ge;

namespace optiling {
class RegisterOpTilingV3UT : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};
bool op_tiling_stub_v3(const Operator &op, const void* value, OpRunInfoV2 &run_info) {
  return true;
}

void* op_parse_stub_v3(const Operator &op, const ge::AscendString &compile_info_json) {
//  static void *p = new int(3);
  static int x = 1024;
  void *p = &x;
  return p;
}

void* op_parse_stub_null_v3(const Operator &op, const ge::AscendString &compile_info_json) {
  return nullptr;
}

void* op_parse_stub_dt1_v3(const Operator &op, const ge::AscendString &compile_info_json) {
//  static void *p = new int(3);
  static int x = 1024;
  void *p = &x;
  return p;
}

void* op_parse_stub_dt2_v3(const Operator &op, const ge::AscendString &compile_info_json) {
//  static void *p = new int(3);
  static int x = 1024;
  void *p = &x;
  return p;
}

void* op_parse_stub_dt3_v3(const Operator &op, const ge::AscendString &compile_info_json) {
//  static void *p = new int(3);
  static int x = 1024;
  void *p = &x;
  return p;
}

void* op_parse_stub_dt4_v3(const Operator &op, const ge::AscendString &compile_info_json) {
//  static void *p = new int(3);
  static int x = 1024;
  void *p = &x;
  return p;
}

REGISTER_OP_TILING_V3(ReluV3, op_tiling_stub_v3, op_parse_stub_v3);
REGISTER_OP_TILING_V3(ReluNullV3, op_tiling_stub_v3, op_parse_stub_null_v3);

TEST_F(RegisterOpTilingV3UT, op_para_calculate_v3_1) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV3");
  GeShape shape({4,3,14,14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  utils::OpRunInfo run_info;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingV3UT, op_para_calculate_v3_2) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluVV3");
  GeShape shape({4,3,14,14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  utils::OpRunInfo run_info;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);

  OpTilingFuncInfo op_func_info(OP_TYPE_AUTO_TILING);
  op_func_info.tiling_func_v3_ = op_tiling_stub_v3;
  op_func_info.parse_func_v3_ = op_parse_stub_v3;
  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  tiling_func_map.emplace(OP_TYPE_AUTO_TILING, op_func_info);
  ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  tiling_func_map.erase(OP_TYPE_AUTO_TILING);
}

TEST_F(RegisterOpTilingV3UT, op_para_calculate_v3_3) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV3");
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);

  vector<string> depend_names = {"x"};
  AttrUtils::SetListStr(op_desc, "_op_infer_depends", depend_names);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  utils::OpRunInfo run_info;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingV3UT, op_para_calculate_v3_4) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluNullV3");
  GeShape shape({4,3,14,14});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key_null";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  utils::OpRunInfo run_info;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  graphStatus ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(RegisterOpTilingV3UT, op_atomic_calculate_v3_1) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, compile_info_json);
  std::vector<int64_t> atomic_output_indices = {0};
  (void) ge::AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_func_info.tiling_func_v3_ = op_tiling_stub_v3;
  op_func_info.parse_func_v3_ = op_parse_stub_dt1_v3;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);

  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV3UT, op_atomic_calculate_v3_2) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, compile_info_json);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_func_info.tiling_func_v3_ = op_tiling_stub_v3;
  op_func_info.parse_func_v3_ = op_parse_stub_dt2_v3;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);

  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV3UT, op_atomic_calculate_v3_3) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, compile_info_json);
  std::vector<int64_t> atomic_output_indices = {1};
  (void) ge::AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_func_info.tiling_func_v3_ = op_tiling_stub_v3;
  op_func_info.parse_func_v3_ = op_parse_stub_dt3_v3;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);

  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV3UT, op_atomic_calculate_v3_4) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, compile_info_json);
  std::map<int64_t, int64_t> temp_map = {{1,1}, {2,2}, {3,3}};
  std::map<string, std::map<int64_t, int64_t>> atomic_workspace_info = {{"qwer", temp_map}};
  op_desc->SetExtAttr(ge::EXT_ATTR_ATOMIC_WORKSPACE_INFO, atomic_workspace_info);
  std::vector<int64_t> workspace_bytes = {1,2,3,4};
  op_desc->SetWorkspaceBytes(workspace_bytes);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_func_info.tiling_func_v3_ = op_tiling_stub_v3;
  op_func_info.parse_func_v3_ = op_parse_stub_dt4_v3;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);

  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV3UT, op_atomic_calculate_v3_5) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  string compile_info_key = "compile_info_key_null";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, compile_info_json);
  std::vector<int64_t> atomic_output_indices = {0};
  (void) ge::AttrUtils::SetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);

  ComputeGraphPtr graph = make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);

  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  OpTilingFuncInfo op_func_info(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  op_func_info.tiling_func_v3_ = op_tiling_stub_v3;
  op_func_info.parse_func_v3_ = op_parse_stub_null_v3;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);

  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);

  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

}
