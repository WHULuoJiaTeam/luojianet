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

#define protected public
#define private public
#include "graph/passes/infer_value_range_pass.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"

#include "inc/external/graph/operator_reg.h"
#include "inc/external/graph/operator.h"
#include "inc/external/graph/operator_factory.h"
#include "inc/graph/operator_factory_impl.h"
#include "inc/kernel.h"
#include "inc/kernel_factory.h"

using namespace std;
using namespace testing;
namespace ge {
class UtestGraphInferValueRangePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

/*
 *   data1  const1
 *     \     /
 *      case1
 *        |
 *      relu10
 *        |
 *    netoutput
 */
ut::GraphBuilder ParentGraphBuilder() {
  ut::GraphBuilder builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  std::vector<int64_t> const_shape = {1};
  auto const1 = builder.AddNode("const1", "Const", 0, 1, FORMAT_NCHW, DT_INT32, const_shape);
  auto case1 = builder.AddNode("case1", CASE, 2, 1);
  auto relu1 = builder.AddNode("relu10", "Relu", 1, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor});
  auto case_in0_shape = GeShape({1, 1,-1, 224});
  auto case_in1_shape = GeShape({1,1});
  std::vector<std::pair<int64_t, int64_t>> in0_range = {make_pair(1, 1), make_pair(1, 1),
                                                       make_pair(1, -1),   make_pair(1, 224)};
  std::vector<std::pair<int64_t, int64_t>> in1_range = {make_pair(1, 100), make_pair(1, 10)};
  case1->GetOpDesc()->MutableInputDesc(0)->SetShape(case_in0_shape);
  case1->GetOpDesc()->MutableInputDesc(0)->SetValueRange(in0_range);
  case1->GetOpDesc()->MutableInputDesc(1)->SetShape(case_in1_shape);
  case1->GetOpDesc()->MutableInputDesc(1)->SetValueRange(in1_range);

  builder.AddDataEdge(data1, 0, case1, 0);
  builder.AddDataEdge(const1, 0, case1, 1);
  builder.AddDataEdge(case1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput, 0);
  return builder;
}

/*
 *   data1   data2
 *     \      /
 *      switch
 *     /      \
 *   relu1   relu2
 *     \      /
 *       merge
 *         |
 *     netoutput
 */
ut::GraphBuilder SwitchSubgraphBuilder(string graph_name, uint32_t num) {
  ut::GraphBuilder builder = ut::GraphBuilder(graph_name);

  std::vector<int64_t> shape1 = {2,2};
  string data1_name = "data1_" + std::to_string(num);
  auto data1 = builder.AddNode(data1_name, "Data", 1, 1, FORMAT_NCHW, DT_INT32, shape1);
  auto data1_desc = data1->GetOpDesc();
  EXPECT_NE(data1_desc, nullptr);
  AttrUtils::SetInt(data1_desc, "_parent_node_index", 0);

  std::vector<int64_t> shape2 = {3,3};
  string data2_name = "data2_" + std::to_string(num);
  auto data2 = builder.AddNode(data2_name, "Data", 1, 1, FORMAT_NCHW, DT_INT32, shape2);
  auto data2_desc = data2->GetOpDesc();
  EXPECT_NE(data2_desc, nullptr);
  AttrUtils::SetInt(data2_desc, "_parent_node_index", 1);

  string switch_name = "switch_" + std::to_string(num);
  auto switch1 = builder.AddNode(switch_name, "Switch", 2, 2);

  string relu1_name = "relu1_" + std::to_string(num);
  auto relu1 = builder.AddNode(relu1_name, "Relu", 1, 1);

  string relu2_name = "relu2_" + std::to_string(num);
  auto relu2 = builder.AddNode(relu2_name, "Relu", 1, 1);

  string merge_name = "merge_" + std::to_string(num);
  auto merge = builder.AddNode(merge_name, "Merge", 2, 1);

  std::vector<int64_t> shape7 = {8,8};
  string output_name = "output_" + std::to_string(num);
  auto netoutput = builder.AddNode(output_name, NETOUTPUT, 1, 0, FORMAT_NCHW, DT_INT32, shape7);
  auto input0_desc = netoutput->GetOpDesc()->MutableInputDesc(0);
  EXPECT_NE(input0_desc, nullptr);
  AttrUtils::SetInt(input0_desc, "_parent_node_index", 0);
  std::vector<std::pair<int64_t, int64_t>> range = {make_pair(1, -1), make_pair(1, -1)};
  input0_desc->SetValueRange(range);

  builder.AddDataEdge(data1, 0, switch1, 0);
  builder.AddDataEdge(data2, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, relu1, 0);
  builder.AddDataEdge(switch1, 1, relu2, 0);
  builder.AddDataEdge(relu1, 0, merge, 0);
  builder.AddDataEdge(relu2, 0, merge, 1);
  builder.AddDataEdge(merge, 0, netoutput, 0);

  return builder;
}

void AddCaseSubgraph(ComputeGraphPtr &parent_graph, uint32_t branch_num) {
  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);

  for (uint32_t i = 0; i < branch_num; ++i) {
    string name = "Branch_Graph_" + std::to_string(i);

    auto builder_subgraph = SwitchSubgraphBuilder(name, i);
    auto switch_subgraph = builder_subgraph.GetGraph();

    case_node->GetOpDesc()->AddSubgraphName(switch_subgraph->GetName());
    case_node->GetOpDesc()->SetSubgraphInstanceName(i, switch_subgraph->GetName());

    switch_subgraph->SetParentNode(case_node);
    switch_subgraph->SetParentGraph(parent_graph);
    EXPECT_EQ(parent_graph->AddSubgraph(switch_subgraph->GetName(), switch_subgraph), GRAPH_SUCCESS);
  }
}

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UnregisteredNodeType) {
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc ge_tensor_desc(GeShape({1, 1, 4, 192}), ge::FORMAT_NCHW, DT_FLOAT16);
  auto addn_op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  addn_op_desc->AddInputDesc(ge_tensor_desc);
  addn_op_desc->AddOutputDesc(ge_tensor_desc);
  auto addn_op_node = graph->AddNode(addn_op_desc);

  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(addn_op_node), SUCCESS);
}

auto ShapeValueInfer = [&](Operator &op) {
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_tensor_desc = op_desc->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> in_shape_range;
  op_desc->MutableInputDesc(0)->GetShapeRange(in_shape_range);
  if (!in_shape_range.empty()) {
    output_tensor_desc->SetValueRange(in_shape_range);
  }
  return SUCCESS;
};
REG_OP(Shape)
    .OP_END_FACTORY_REG(Shape)
IMPL_INFER_VALUE_RANGE_FUNC(Shape, ShapeValueRangeFunc){
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_tensor_desc = op_desc->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> in_shape_range;
  op_desc->MutableInputDesc(0)->GetShapeRange(in_shape_range);
  if (!in_shape_range.empty()) {
    output_tensor_desc->SetValueRange(in_shape_range);
  }
  return GRAPH_SUCCESS;
}

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UseRegistedFunc_NotInfer) {
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Shape, INPUT_IS_DYNAMIC, ShapeValueRangeFunc);
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc ge_tensor_desc(GeShape({1, 1, 4, 192}), ge::FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {make_pair(1, 1), make_pair(1, 1),
                                                          make_pair(4, 4),   make_pair(192, 192)};
  ge_tensor_desc.SetShapeRange(shape_range);
  GeTensorDesc output_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, DT_INT32);
  auto op_desc = std::make_shared<OpDesc>("Shape", "Shape");
  op_desc->AddInputDesc(ge_tensor_desc);
  op_desc->AddOutputDesc(output_tensor_desc);
  auto op_node = graph->AddNode(op_desc);

  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(op_node), SUCCESS);

  auto output_0_desc = op_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> value_range;
  output_0_desc.GetValueRange(value_range);
  EXPECT_EQ(value_range.empty(), true);
}

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UseRegistedFunc_DoInfer) {
  // sqrt -> shape -> Output
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Shape, INPUT_IS_DYNAMIC, ShapeValueRangeFunc);
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc sqrt_tensor_desc(GeShape({-1, -1, 4, 192}), ge::FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {make_pair(1, 100), make_pair(1, 240),
                                                          make_pair(4, 4),   make_pair(192, 192)};
  sqrt_tensor_desc.SetShapeRange(shape_range);
  auto sqrt_op_desc = std::make_shared<OpDesc>("Sqrt", "Sqrt");
  sqrt_op_desc->AddInputDesc(sqrt_tensor_desc);
  sqrt_op_desc->AddOutputDesc(sqrt_tensor_desc);
  auto sqrt_node = graph->AddNode(sqrt_op_desc);

  GeTensorDesc shape_output_desc(GeShape({4}), ge::FORMAT_NCHW, DT_INT32);
  auto shape_op_desc = std::make_shared<OpDesc>("Shape", "Shape");
  shape_op_desc->AddInputDesc(sqrt_tensor_desc);
  shape_op_desc->AddOutputDesc(shape_output_desc);
  auto shape_node = graph->AddNode(shape_op_desc);

  GeTensorDesc Output_in_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
  auto Output_op_desc = std::make_shared<OpDesc>("Output", "Output");
  Output_op_desc->AddInputDesc(Output_in_tensor_desc);
  auto Output_node = graph->AddNode(Output_op_desc);

  ge::GraphUtils::AddEdge(sqrt_node->GetOutDataAnchor(0), shape_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), Output_node->GetInDataAnchor(0));
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);


  InferValueRangePass infer_pass;
  auto ret = infer_pass.Run(shape_node);
  EXPECT_EQ(ret, SUCCESS);

  auto output_0_desc = shape_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> value_range;
  output_0_desc.GetValueRange(value_range);
  EXPECT_EQ(value_range.size(), 4);
  std::vector<int64_t> target_value_range = {1, 100, 1, 240, 4, 4, 192, 192};
  std::vector<int64_t> output_value_range;
  for (auto pair : value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(target_value_range, output_value_range);

  auto in_0_desc = Output_node->GetOpDesc()->GetInputDesc(0);
  value_range.clear();
  in_0_desc.GetValueRange(value_range);
  EXPECT_EQ(value_range.size(), 4);
  output_value_range.clear();
  for (auto pair : value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(target_value_range, output_value_range);

}

class AddKernel : public Kernel {
 public:
  Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr> &input,
                 std::vector<ge::GeTensorPtr> &v_output) override {
    if (input[0]->GetTensorDesc().GetDataType() == DT_INT64 || input[0]->GetTensorDesc().GetDataType() == DT_UINT64) {
      vector<int64_t> data_vec;
      auto data_num = input[0]->GetTensorDesc().GetShape().GetShapeSize();
      auto x1_data = reinterpret_cast<const int64_t *>(input[0]->GetData().data());
      auto x2_data = reinterpret_cast<const int64_t *>(input[1]->GetData().data());
      for (size_t i = 0; i < data_num; i++) {
        auto x_index = *(x1_data + i);
        auto y_index = *(x2_data + i);
        data_vec.push_back(x_index + y_index);
      }
      GeTensorPtr const_tensor = std::make_shared<ge::GeTensor>(input[0]->GetTensorDesc(), (uint8_t *)data_vec.data(),
                                                                data_num * sizeof(int64_t));
      v_output.emplace_back(const_tensor);
      return SUCCESS;
    } else if (input[0]->GetTensorDesc().GetDataType() == DT_INT32 || input[0]->GetTensorDesc().GetDataType() == DT_UINT32) {
      vector<int32_t> data_vec;
      auto data_num = input[0]->GetTensorDesc().GetShape().GetShapeSize();
      if (input[0]->GetTensorDesc().GetShape().IsScalar()) {
        data_num = 1;
      }
      auto x1_data = reinterpret_cast<const int32_t *>(input[0]->GetData().data());
      auto x2_data = reinterpret_cast<const int32_t *>(input[1]->GetData().data());
      for (size_t i = 0; i < data_num; i++) {
        auto x_index = *(x1_data + i);
        auto y_index = *(x2_data + i);
        data_vec.push_back(x_index + y_index);
      }
      GeTensorPtr const_tensor = std::make_shared<ge::GeTensor>(input[0]->GetTensorDesc(), (uint8_t *)data_vec.data(),
                                                                data_num * sizeof(int32_t));
      v_output.emplace_back(const_tensor);
      return SUCCESS;
    }
  }
};
REGISTER_KERNEL(ADD, AddKernel);
INFER_VALUE_RANGE_DEFAULT_REG(Add);
INFER_VALUE_RANGE_DEFAULT_REG(Sqrt);

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UseCpuKernel_InputsHaveUnKnownValueRange) {
  // shape --- add --- sqrt
  // constant /
  auto graph = std::make_shared<ComputeGraph>("test_graph");

  vector<int64_t> dims_vec = {4};
  vector<int64_t> data_vec = {1, 1, 1, 1};
  GeTensorDesc const_tensor_desc(ge::GeShape(dims_vec), ge::FORMAT_NCHW, ge::DT_INT64);
  GeTensorPtr const_tensor =
    std::make_shared<ge::GeTensor>(const_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int64_t));

  auto const_op_desc = std::make_shared<OpDesc>("Constant", "Constant");
  const_op_desc->AddOutputDesc(const_tensor_desc);
  EXPECT_EQ(OpDescUtils::SetWeights(const_op_desc, const_tensor), GRAPH_SUCCESS);
  auto const_node = graph->AddNode(const_op_desc);

  GeTensorDesc shape_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT64);
  std::vector<std::pair<int64_t, int64_t>> unknown_value_range = {make_pair(1, -1), make_pair(1, 240),
                                                                  make_pair(4, 4),  make_pair(192, 192)};
  shape_tensor_desc.SetValueRange(unknown_value_range);
  auto shape_op_desc = std::make_shared<OpDesc>("Shape", "Shape");
  shape_op_desc->AddOutputDesc(shape_tensor_desc);
  auto shape_node = graph->AddNode(shape_op_desc);

  GeTensorDesc add_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT64);
  auto add_op_desc = std::make_shared<OpDesc>("Add", "Add");
  add_op_desc->AddInputDesc(shape_tensor_desc);
  add_op_desc->AddInputDesc(const_tensor_desc);
  add_op_desc->AddOutputDesc(add_tensor_desc);
  auto add_node = graph->AddNode(add_op_desc);

  ge::GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));

  // test unknown value range
  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(add_node), SUCCESS);
  auto output_0_desc = add_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> out_value_range;
  output_0_desc.GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 4);

  std::vector<int64_t> unknown_target_value_range = {1, -1, 1, -1, 1, -1, 1, -1};
  std::vector<int64_t> output_value_range;
  for (auto pair : out_value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(unknown_target_value_range, output_value_range);
}

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UseCpuKernel_InputsHaveZeroInValueRange) {
  // shape --- add --- sqrt
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  GeTensorDesc shape_tensor_desc(GeShape({2}), ge::FORMAT_NCHW, ge::DT_INT64);
  std::vector<std::pair<int64_t, int64_t>> unknown_value_range = {make_pair(1, -1), make_pair(0, 240)};
  shape_tensor_desc.SetValueRange(unknown_value_range);
  auto shape_op_desc = std::make_shared<OpDesc>("Shape", "Shape");
  shape_op_desc->AddOutputDesc(shape_tensor_desc);
  auto shape_node = graph->AddNode(shape_op_desc);

  GeTensorDesc add_tensor_desc(GeShape({2}), ge::FORMAT_NCHW, ge::DT_INT64);
  auto add_op_desc = std::make_shared<OpDesc>("Add", "Add");
  add_op_desc->AddInputDesc(shape_tensor_desc);
  add_op_desc->AddInputDesc(shape_tensor_desc);
  add_op_desc->AddOutputDesc(add_tensor_desc);
  auto add_node = graph->AddNode(add_op_desc);

  ge::GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));

  // test unknown value range
  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(add_node), SUCCESS);
  auto output_0_desc = add_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> out_value_range;
  output_0_desc.GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 0);
}

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UseCpuKernel_InputsHaveUnKnownValueRange_ScalarOutput) {
  // shape --- add --- sqrt
  // constant /
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  vector<int64_t> data_vec = {1};
  GeTensorDesc const_tensor_desc(ge::GeShape(), ge::FORMAT_NCHW, ge::DT_INT64);
  GeTensorPtr const_tensor =
      std::make_shared<ge::GeTensor>(const_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int64_t));

  auto const_op_desc = std::make_shared<OpDesc>("Constant", "Constant");
  const_op_desc->AddOutputDesc(const_tensor_desc);
  EXPECT_EQ(OpDescUtils::SetWeights(const_op_desc, const_tensor), GRAPH_SUCCESS);
  auto const_node = graph->AddNode(const_op_desc);

  GeTensorDesc shape_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT64);
  std::vector<std::pair<int64_t, int64_t>> unknown_value_range = {make_pair(1, -1)};
  shape_tensor_desc.SetValueRange(unknown_value_range);
  auto shape_op_desc = std::make_shared<OpDesc>("Shape", "Shape");
  shape_op_desc->AddOutputDesc(shape_tensor_desc);
  auto shape_node = graph->AddNode(shape_op_desc);

  GeTensorDesc add_tensor_desc(GeShape(), ge::FORMAT_NCHW, ge::DT_INT64);
  auto add_op_desc = std::make_shared<OpDesc>("Add", "Add");
  add_op_desc->AddInputDesc(shape_tensor_desc);
  add_op_desc->AddInputDesc(const_tensor_desc);
  add_op_desc->AddOutputDesc(add_tensor_desc);
  auto add_node = graph->AddNode(add_op_desc);

  ge::GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));

  // test unknown value range
  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(add_node), SUCCESS);
  auto output_0_desc = add_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> out_value_range;
  output_0_desc.GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 1);

  std::vector<int64_t> unknown_target_value_range = {1, -1};
  std::vector<int64_t> output_value_range;
  for (auto pair : out_value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(unknown_target_value_range, output_value_range);
}

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UseCpuKernel_InputsAreKnownValueRange_ScalarOutput) {
  // shape --- add --- sqrt
  // constant /
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  vector<int32_t> data_vec = {2};
  GeTensorDesc const_td(ge::GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorPtr const_tensor = std::make_shared<ge::GeTensor>(const_td, (uint8_t *)data_vec.data(), sizeof(int32_t));
  auto const_op_desc = std::make_shared<OpDesc>("Constant", "Constant");
  const_op_desc->AddOutputDesc(const_td);
  EXPECT_EQ(OpDescUtils::SetWeights(const_op_desc, const_tensor), GRAPH_SUCCESS);
  auto const_node = graph->AddNode(const_op_desc);

  GeTensorDesc shape_td(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> known_value_range = {make_pair(1, 100)};
  shape_td.SetValueRange(known_value_range);
  auto shape_op_desc = std::make_shared<OpDesc>("Shape", "Shape");
  shape_op_desc->AddOutputDesc(shape_td);
  auto shape_node = graph->AddNode(shape_op_desc);

  GeTensorDesc add_td(GeShape(), ge::FORMAT_NCHW, ge::DT_INT32);
  auto add_op_desc = std::make_shared<OpDesc>("Add", "Add");
  add_op_desc->AddInputDesc(shape_td);
  add_op_desc->AddInputDesc(const_td);
  add_op_desc->AddOutputDesc(add_td);
  auto add_node = graph->AddNode(add_op_desc);

  ge::GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));

  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(add_node), SUCCESS);

  auto output_0_desc = add_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> out_value_range;
  output_0_desc.GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 1);

  std::vector<int64_t> target_value_range = {3, 102};
  std::vector<int64_t> output_value_range = {out_value_range[0].first, out_value_range[0].second};
  EXPECT_EQ(output_value_range, target_value_range);
}

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UseCpuKernel_InputsAreKnownValueRange_Int64) {
  // shape --- add --- sqrt
  // constant /
  auto graph = std::make_shared<ComputeGraph>("test_graph");

  vector<int64_t> dims_vec = {4};
  vector<int64_t> data_vec = {1, 1, 1, 1};
  GeTensorDesc const_tensor_desc(ge::GeShape(dims_vec), ge::FORMAT_NCHW, ge::DT_INT64);
  GeTensorPtr const_tensor =
    std::make_shared<ge::GeTensor>(const_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int64_t));

  auto const_op_desc = std::make_shared<OpDesc>("Constant", "Constant");
  const_op_desc->AddOutputDesc(const_tensor_desc);
  EXPECT_EQ(OpDescUtils::SetWeights(const_op_desc, const_tensor), GRAPH_SUCCESS);
  auto const_node = graph->AddNode(const_op_desc);

  GeTensorDesc shape_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT64);
  std::vector<std::pair<int64_t, int64_t>> unknown_value_range = {make_pair(1, 100), make_pair(1, 240),
                                                                  make_pair(4, 4),  make_pair(192, 192)};
  shape_tensor_desc.SetValueRange(unknown_value_range);
  auto shape_op_desc = std::make_shared<OpDesc>("Shape", "Shape");
  shape_op_desc->AddOutputDesc(shape_tensor_desc);
  auto shape_node = graph->AddNode(shape_op_desc);

  GeTensorDesc add_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT64);
  auto add_op_desc = std::make_shared<OpDesc>("Add", "Add");
  add_op_desc->AddInputDesc(shape_tensor_desc);
  add_op_desc->AddInputDesc(const_tensor_desc);
  add_op_desc->AddOutputDesc(add_tensor_desc);
  auto add_node = graph->AddNode(add_op_desc);

  auto sqrt_op_desc = std::make_shared<OpDesc>("Sqrt", "Sqrt");
  sqrt_op_desc->AddInputDesc(GeTensorDesc());
  auto sqrt_node = graph->AddNode(sqrt_op_desc);

  ge::GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), sqrt_node->GetInDataAnchor(1));

  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(sqrt_node), SUCCESS);

  // test known value range
  EXPECT_EQ(infer_pass.Run(add_node), SUCCESS);
  auto output_0_desc = add_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> out_value_range;
  output_0_desc.GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 4);

  std::vector<int64_t> target_value_range = {2, 101, 2, 241, 5, 5, 193, 193};
  std::vector<int64_t> output_value_range;
  for (auto pair : out_value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(target_value_range, output_value_range);
}

TEST_F(UtestGraphInferValueRangePass, CallRun_NoSubgraph_UseCpuKernel_InputsAreKnownValueRange_Int32) {
  // shape --- add --- sqrt
  // constant /
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  vector<int32_t> data_vec = {1, 100, 2, 200};
  GeTensorDesc const_tensor_desc(ge::GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
  GeTensorPtr const_tensor =
    std::make_shared<ge::GeTensor>(const_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  auto const_op_desc = std::make_shared<OpDesc>("Constant", "Constant");
  const_op_desc->AddOutputDesc(const_tensor_desc);
  EXPECT_EQ(OpDescUtils::SetWeights(const_op_desc, const_tensor), GRAPH_SUCCESS);
  auto const_node = graph->AddNode(const_op_desc);

  GeTensorDesc shape_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> known_value_range = {make_pair(1, 100), make_pair(1, 240),
                                                                make_pair(4, 4),   make_pair(192, 192)};
  shape_tensor_desc.SetValueRange(known_value_range);
  auto shape_op_desc = std::make_shared<OpDesc>("Shape", "Shape");
  shape_op_desc->AddOutputDesc(shape_tensor_desc);
  auto shape_node = graph->AddNode(shape_op_desc);

  GeTensorDesc add_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, ge::DT_INT32);
  auto add_op_desc = std::make_shared<OpDesc>("Add", "Add");
  add_op_desc->AddInputDesc(shape_tensor_desc);
  add_op_desc->AddInputDesc(const_tensor_desc);
  add_op_desc->AddOutputDesc(add_tensor_desc);
  auto add_node = graph->AddNode(add_op_desc);

  ge::GraphUtils::AddEdge(shape_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));

  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(add_node), SUCCESS);
  auto output_0_desc = add_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> out_value_range;
  output_0_desc.GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 4);

  std::vector<int64_t> target_value_range = {2, 101, 101, 340, 6, 6, 392, 392};
  std::vector<int64_t> output_value_range;
  for (auto pair : out_value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(target_value_range, output_value_range);
}

REG_OP(Case)
    .OP_END_FACTORY_REG(Case)
IMPL_INFER_VALUE_RANGE_FUNC(Case, ValueRangeFunc){
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_tensor_desc = op_desc->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> in_value_range;
  output_tensor_desc->GetValueRange(in_value_range);
  if (in_value_range.empty()) {
    std::vector<std::pair<int64_t, int64_t>> out_value_range = {make_pair(1, 2), make_pair(1, 3),
                                                                make_pair(1, 4),   make_pair(1, 5)};;
    output_tensor_desc->SetValueRange(out_value_range);
  }
  return GRAPH_SUCCESS;
}
INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Case, INPUT_HAS_VALUE_RANGE, ValueRangeFunc);

TEST_F(UtestGraphInferValueRangePass, CallRun_HasCaeSubgraph_WhenBeforeSubgraph) {
  auto builder = ParentGraphBuilder();
  auto parent_graph = builder.GetGraph();
  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  // check before subgraph
  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);
  InferValueRangePass infer_pass;
  EXPECT_EQ(infer_pass.Run(case_node), SUCCESS);

  auto case_out_0_desc = case_node->GetOpDesc()->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> out_value_range;
  case_out_0_desc->GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 4);
  std::vector<int64_t> target_value_range = {1,2,1,3,1,4,1,5};
  std::vector<int64_t> output_value_range_list;
  for (auto pair : out_value_range) {
    output_value_range_list.push_back(pair.first);
    output_value_range_list.push_back(pair.second);
  }
  EXPECT_EQ(target_value_range, output_value_range_list);

  auto data_node = subgraphs[0]->FindNode("data1_0");
  auto data_output_0_desc = data_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<int64_t> target_value_range_list = {1, 1, 1, 1, 1, -1, 1, 224};
  std::vector<std::pair<int64_t, int64_t>> output_value_range;
  data_output_0_desc.GetValueRange(output_value_range);
  EXPECT_EQ(output_value_range.size(), 4);
  std::vector<int64_t> data_value_range_list;
  for (auto pair : output_value_range) {
    data_value_range_list.push_back(pair.first);
    data_value_range_list.push_back(pair.second);
  }
  EXPECT_EQ(data_value_range_list, target_value_range_list);

  data_node = subgraphs[0]->FindNode("data2_0");
  auto data2_input_0_desc = data_node->GetOpDesc()->GetInputDesc(0);
  std::vector<int64_t> target_value_range_list2 = {1, 100, 1, 10};
  out_value_range.clear();
  data2_input_0_desc.GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 2);
  data_value_range_list.clear();
  for (auto pair : out_value_range) {
    data_value_range_list.push_back(pair.first);
    data_value_range_list.push_back(pair.second);
  }
  EXPECT_EQ(data_value_range_list, target_value_range_list2);
}

TEST_F(UtestGraphInferValueRangePass, CallRun_HasCaeSubgraph_WhenAfterSubgraph) {
  auto builder = ParentGraphBuilder();
  auto parent_graph = builder.GetGraph();
  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);
  InferValueRangePass infer_pass;
  // check after subgraph
  infer_pass.options_[kOptimizeAfterSubGraph] = "yes";
  EXPECT_EQ(infer_pass.Run(case_node), SUCCESS);

  std::vector<int64_t> out_target_dims = {1, -1, 1, -1};
  auto case_out = case_node->GetOpDesc()->GetOutputDescPtr(0);
  std::vector<std::pair<int64_t, int64_t>> out_value_range;
  case_out->GetValueRange(out_value_range);
  EXPECT_EQ(out_value_range.size(), 2);

  std::vector<int64_t> output_value_range_list;
  for (auto pair : out_value_range) {
    output_value_range_list.push_back(pair.first);
    output_value_range_list.push_back(pair.second);
  }
  EXPECT_EQ(out_target_dims, output_value_range_list);
}

TEST_F(UtestGraphInferValueRangePass, CallRun_HasSubgraph_WhenAfterSubgraph_ForMultiDims) {
  auto builder = ParentGraphBuilder();
  auto parent_graph = builder.GetGraph();
  AddCaseSubgraph(parent_graph, 2);
  auto subgraphs = parent_graph->GetAllSubgraphs();
  EXPECT_EQ(subgraphs.size(), 2);

  auto case_node = parent_graph->FindNode("case1");
  EXPECT_NE(case_node, nullptr);
  InferValueRangePass infer_pass;
  infer_pass.options_[kOptimizeAfterSubGraph] = "yes";

  // check after subgraph for multi-batch
  auto set_ret = AttrUtils::SetInt(case_node->GetOpDesc(), ATTR_NAME_BATCH_NUM, 2);
  EXPECT_EQ(set_ret, true);
  EXPECT_EQ(infer_pass.Run(case_node), GRAPH_FAILED);
}
}  // namespace ge
