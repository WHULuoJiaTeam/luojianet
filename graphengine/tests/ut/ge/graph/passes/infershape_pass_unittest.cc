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
#include "graph/passes/infershape_pass.h"

#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory.h"
#include "graph/operator_reg.h"
#include "graph_builder_utils.h"

using namespace std;
using namespace testing;
namespace ge {
class UtestGraphInfershapePass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static NodePtr CreateNode(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(1024);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(1024);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddInferFunc(stub_func);
  op_desc->AddInferFormatFunc(stub_func);
  op_desc->AddVerifierFunc(stub_func);

  return graph.AddNode(op_desc);
}

TEST_F(UtestGraphInfershapePass, infershape_pass_failed) {
  GeTensorDesc ge_tensor_desc(GeShape({-2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  string type = "AddN";
  auto addn_op_desc = std::make_shared<OpDesc>("AddN", type);
  addn_op_desc->AddInputDesc(ge_tensor_desc);
  addn_op_desc->AddOutputDesc(ge_tensor_desc);
  auto graph = std::make_shared<ComputeGraph>("test");
  auto addn_node = std::make_shared<Node>(addn_op_desc, graph);
  addn_node->Init();

  InferShapePass infershape_pass;
  EXPECT_EQ(infershape_pass.Run(addn_node), GRAPH_FAILED);
}

TEST_F(UtestGraphInfershapePass, stop_node_for_while_loop) {
/*******************************************************************************
 *      Exit         Identify
 *        \         /       \.
 *         \       /         \.
 *          Switch           Add
 *         /     |            |
 *        /      |            |
 *       /       |            |
 *  LoopCond     |            |
 *      \        |            |
 *       \       |            |
 *        \      |            |
 *       Less    |            |
 *          \    |       NextIteration
 *           \   |            |
 *            \  |            |
 *            Merge <---------|
 *              |
 *              |
 *            Enter
 ******************************************************************************/
  auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto data1 = CreateNode(*graph, "data", DATA, 1, 1);
  auto enter1 = CreateNode(*graph, "enter", ENTER, 1, 1);
  auto merge1 = CreateNode(*graph, "merge", MERGE, 2, 2);
  auto less1 = CreateNode(*graph, "less", LESS, 2, 1);
  auto loop1 = CreateNode(*graph, "loopcond", LOOPCOND, 1, 1);
  auto switch1 = CreateNode(*graph, "switch", SWITCH, 2, 2);
  auto ident1 = CreateNode(*graph, "identity", IDENTITY, 1, 1);
  auto add1 = CreateNode(*graph, "add", ADD, 2, 1);
  auto next1 = CreateNode(*graph, "next", NEXTITERATION, 1, 1);
  auto exit1 = CreateNode(*graph, "exit", EXIT, 1, 1);
  auto value0 = CreateNode(*graph, "const", CONSTANT, 0, 1);
  auto value1 = CreateNode(*graph, "const", CONSTANT, 0, 1);
  auto output1 = CreateNode(*graph, "net_output", NETOUTPUT, 1, 1);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), enter1->GetInDataAnchor(0));
  GraphUtils::AddEdge(enter1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), loop1->GetInDataAnchor(0));

  GraphUtils::AddEdge(loop1->GetOutDataAnchor(0), switch1->GetInDataAnchor(1));
  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), switch1->GetInDataAnchor(0));

  GraphUtils::AddEdge(switch1->GetOutDataAnchor(0), exit1->GetInDataAnchor(0));
  GraphUtils::AddEdge(switch1->GetOutDataAnchor(1), ident1->GetInDataAnchor(0));

  GraphUtils::AddEdge(ident1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
  GraphUtils::AddEdge(value1->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
  GraphUtils::AddEdge(add1->GetOutDataAnchor(0), next1->GetInDataAnchor(0));

  GraphUtils::AddEdge(next1->GetOutDataAnchor(0), merge1->GetInDataAnchor(1));
  GraphUtils::AddEdge(exit1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  GEPass ge_passes(graph);
  NamesToPass names_to_passes;
  InferShapePass infer_shape_pass;
  names_to_passes.emplace_back("InferShapePass", &infer_shape_pass);

  EXPECT_EQ(infer_shape_pass.Run(switch1), SUCCESS);
  auto suspend_nodes = infer_shape_pass.GetNodesSuspend();
  auto exit_node = graph->FindNode("exit");
  EXPECT_EQ(suspend_nodes.count(exit_node), 1);
  infer_shape_pass.OnSuspendNodesLeaked();
  auto resume_nodes = infer_shape_pass.GetNodesResume();
  EXPECT_EQ(resume_nodes.count(exit_node), 1);
}
TEST_F(UtestGraphInfershapePass, update_tensordesc_when_changed) {
  GeTensorDesc src_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dst_ge_tensor_desc(GeShape({2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDescPtr src_tensor_desc_ptr = std::make_shared<GeTensorDesc>(src_ge_tensor_desc);
  GeTensorDescPtr dst_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  bool changed = false;
  infershape_pass.UpdateTensorDesc(src_tensor_desc_ptr, dst_tensor_desc_ptr, changed);
  EXPECT_EQ(changed, true);
  EXPECT_EQ(dst_tensor_desc_ptr->GetShape().GetDims(), std::vector<int64_t>({1, 2, 3, 4}));
}

TEST_F(UtestGraphInfershapePass, update_tensordesc_when_not_changed) {
  GeTensorDesc src_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDescPtr src_tensor_desc_ptr = std::make_shared<GeTensorDesc>(src_ge_tensor_desc);
  GeTensorDescPtr dst_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  bool changed = false;
  infershape_pass.UpdateTensorDesc(src_tensor_desc_ptr, dst_tensor_desc_ptr, changed);
  EXPECT_EQ(changed, false);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_failed) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc ge_tensor_desc2(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphs({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr}, dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_get_unknown_rank) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphs({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr}, dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(dst_ge_tensor_desc_ptr->GetShape().GetDims(), UNKNOWN_RANK);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_get_unknown_shape) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphs({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr}, dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(dst_ge_tensor_desc_ptr->GetShape().GetDims(), std::vector<int64_t>({-1,2,3,4}));
  // todo shape range?
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_for_multiDims_failed) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  GeTensorDesc ge_tensor_desc2(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphsForMultiDims({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr},
                                                                   dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_for_multiDims_failed_shape_size_overflow) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({INT64_MAX, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphsForMultiDims({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr},
                                                                   dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestGraphInfershapePass, update_output_from_subgraphs_for_multiDims_success) {
  // ref output has different dtype
  GeTensorDesc ge_tensor_desc1(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc ge_tensor_desc2(GeShape({2, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc dst_ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT);
  GeTensorDescPtr ge_tensor_desc1_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc1);
  GeTensorDescPtr ge_tensor_desc2_ptr = std::make_shared<GeTensorDesc>(ge_tensor_desc2);
  GeTensorDescPtr dst_ge_tensor_desc_ptr = std::make_shared<GeTensorDesc>(dst_ge_tensor_desc);
  InferShapePass infershape_pass;
  auto ret = infershape_pass.UpdateOutputFromSubgraphsForMultiDims({ge_tensor_desc1_ptr, ge_tensor_desc2_ptr},
                                                                   dst_ge_tensor_desc_ptr);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(dst_ge_tensor_desc_ptr->GetShape().GetDims(), std::vector<int64_t>({2,2,3,4}));
}
}  // namespace ge
