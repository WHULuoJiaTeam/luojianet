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
class RegisterOpTilingV2UT : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};

bool op_tiling_stub_v2(const Operator &op, const utils::OpCompileInfo &compile_info, utils::OpRunInfo &run_info) {
  return true;
}

REGISTER_OP_TILING_V2(ReluV2, op_tiling_stub_v2);

TEST_F(RegisterOpTilingV2UT, AddNameToTensordesc_1) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
  GeShape shape({4,3,16,16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  vector<string> depend_names = {"x"};
  AttrUtils::SetListStr(op_desc, "_op_infer_depends", depend_names);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->GetName(), "");
  EXPECT_EQ(op_desc->MutableInputDesc(1)->GetName(), "");
  AddNameToTensordesc(op_desc);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->GetName(), "x");
  EXPECT_EQ(op_desc->MutableInputDesc(1)->GetName(), "y");
}

TEST_F(RegisterOpTilingV2UT, AddNameToTensordesc_2) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
  GeShape shape({4,3,16,16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);

  EXPECT_EQ(op_desc->MutableInputDesc(0)->GetName(), "");
  EXPECT_EQ(op_desc->MutableInputDesc(1)->GetName(), "");
  AddNameToTensordesc(op_desc);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->GetName(), "");
  EXPECT_EQ(op_desc->MutableInputDesc(1)->GetName(), "");
}

TEST_F(RegisterOpTilingV2UT, replace_and_recovery_tensor_1) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
  GeShape shape;
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);

  std::vector<int32_t> indexes;
  ReplaceEmptyShapeOfTensorDesc(op_desc, indexes);
  std::vector<int32_t> expect_indexes = {0, 1, -1};
  EXPECT_EQ(indexes, expect_indexes);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDimNum(), 1);
  EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDimNum(), 1);
  EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDimNum(), 1);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDim(0), 1);
  EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDim(0), 1);
  EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDim(0), 1);

  RecoveryEmptyShapeOfTensorDesc(op_desc, indexes);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDimNum(), 0);
  EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDimNum(), 0);
  EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDimNum(), 0);
}

TEST_F(RegisterOpTilingV2UT, replace_and_recovery_tensor_2) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
  GeShape shape({4,3,16,16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);

  std::vector<int32_t> indexes;
  ReplaceEmptyShapeOfTensorDesc(op_desc, indexes);
  EXPECT_EQ(indexes.size(), 0);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDimNum(), 4);
  EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDimNum(), 4);
  EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDimNum(), 4);

  RecoveryEmptyShapeOfTensorDesc(op_desc, indexes);
  EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDimNum(), 4);
  EXPECT_EQ(op_desc->MutableInputDesc(1)->MutableShape().GetDimNum(), 4);
  EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDimNum(), 4);
}

TEST_F(RegisterOpTilingV2UT, op_para_calculate_v2_1) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
  GeShape shape({4,3,16,16});
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
}

TEST_F(RegisterOpTilingV2UT, op_para_calculate_v2_2) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluVV");
  GeShape shape({4,3,16,16});
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
  op_func_info.tiling_func_v2_ = op_tiling_stub_v2;
  std::unordered_map<std::string, OpTilingFuncInfo> &tiling_func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  tiling_func_map.emplace(OP_TYPE_AUTO_TILING, op_func_info);
  ret = OpParaCalculateV2(op, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  tiling_func_map.erase(OP_TYPE_AUTO_TILING);
}

TEST_F(RegisterOpTilingV2UT, op_para_calculate_v2_3) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
  GeShape shape({4,3,16,16});
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
}

TEST_F(RegisterOpTilingV2UT, op_para_calculate_v2_4) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
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
}

TEST_F(RegisterOpTilingV2UT, op_atomic_calculate_v2_1) {
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
  op_func_info.tiling_func_v2_ = op_tiling_stub_v2;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);

  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV2UT, op_atomic_calculate_v2_2) {
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
  op_func_info.tiling_func_v2_ = op_tiling_stub_v2;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);
  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV2UT, op_atomic_calculate_v2_3) {
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
  op_func_info.tiling_func_v2_ = op_tiling_stub_v2;
  tiling_func_map.emplace(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN, op_func_info);
  utils::OpRunInfo run_info;
  graphStatus ret = OpAtomicCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_FAILED);
  tiling_func_map.erase(OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
}

TEST_F(RegisterOpTilingV2UT, op_ffts_calculate_v2_4) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", "ReluV2");
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
  std::vector<OpRunInfoV2> run_info;
  graphStatus ret = OpFftsCalculateV2(*node, run_info);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(run_info.size(), 2U);
}
}
