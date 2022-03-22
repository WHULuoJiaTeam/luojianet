/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "graph/node.h"
#include "graph/node_impl.h"
#include "graph/any_value.h"
#include "graph/anchor.h"
#include "graph/op_desc.h"
#include "graph/op_desc_impl.h"
#include "graph_builder_utils.h"

#undef private
#undef protected

namespace ge {
class UtestNode : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

template<class T>
std::shared_ptr<T> make_nullptr(){
  return nullptr;
}


TEST_F(UtestNode, GetInDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  auto data_node = graph->FindNode("Data");
  auto in_data_anchor0 = data_node->GetInDataAnchor(0);
  EXPECT_NE(in_data_anchor0, nullptr);

  auto in_data_anchor1 = data_node->GetInDataAnchor(1);
  EXPECT_EQ(in_data_anchor1, nullptr);

  auto in_data_anchor2 = data_node->GetInDataAnchor(-1);
  EXPECT_EQ(in_data_anchor2, nullptr);
}
TEST_F(UtestNode, GetInAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  auto data_node = graph->FindNode("Data");
  auto in_anchor0 = data_node->GetInAnchor(-2);
  EXPECT_EQ(in_anchor0, nullptr);
}
TEST_F(UtestNode, GetOutAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  auto data_node = graph->FindNode("Data");
  auto out_anchor0 = data_node->GetOutAnchor(-2);
  EXPECT_EQ(out_anchor0, nullptr);
}

TEST_F(UtestNode, EqualCase) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  auto graph = builder.GetGraph();
  Node n();
  EXPECT_EQ(data_node->Init(), data_node->Init());
  EXPECT_EQ(data_node->SetOwnerComputeGraph(nullptr), GRAPH_PARAM_INVALID);
  EXPECT_EQ(data_node->ClearOwnerGraph(graph), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->GetAllInDataAnchors().size(), 1);
  EXPECT_EQ(data_node->GetAllOutDataAnchors().size(), 1);
  EXPECT_EQ(data_node->NodeAttrsAreEqual(*data_node), true);
  attr_node->impl_->attrs_["ex"] = AnyValue::CreateFrom<int>(100);
  EXPECT_EQ(data_node->NodeAttrsAreEqual(*attr_node), false);
  data_node->impl_->attrs_["ex2"] = AnyValue::CreateFrom<int>(1000);
  EXPECT_EQ(data_node->NodeAttrsAreEqual(*attr_node), false);
  EXPECT_EQ(data_node->NodeMembersAreEqual(*data_node), true);
  EXPECT_EQ(data_node->AddLinkFromForParse(attr_node), GRAPH_PARAM_INVALID);
  EXPECT_EQ(data_node->AddLinkFrom(attr_node), GRAPH_PARAM_INVALID);
  EXPECT_EQ(data_node->AddLinkFrom(2, attr_node), GRAPH_PARAM_INVALID);
  EXPECT_EQ(data_node->AddLinkFrom("Attr", attr_node), GRAPH_PARAM_INVALID);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, nullptr, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, out_anch, 1), true);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(inc_anch), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(out_anch, inc_anch, 1),false);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  EXPECT_EQ(attr_node->AddLinkFromForParse(data_node), GRAPH_SUCCESS);
  EXPECT_EQ(attr_node->AddLinkFrom(2, data_node),GRAPH_SUCCESS);
  EXPECT_EQ(attr_node->AddLinkFrom("Attr", data_node),GRAPH_SUCCESS);
  EXPECT_EQ(data_node->GetOutNodes().size(), 3);
  EXPECT_EQ(data_node->GetOutDataNodes().size(), 3);
  EXPECT_EQ(data_node->GetOutDataNodesSize(), 3);
  EXPECT_EQ(attr_node->GetInNodes().size(), 3);
}

TEST_F(UtestNode, GetCase) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  auto graph = builder.GetGraph();
  EXPECT_EQ(data_node->GetAllInAnchors().size(), 1);
  EXPECT_EQ(attr_node->GetAllOutAnchors().size(), 2);
  EXPECT_EQ(data_node->GetInNodes().size(), 0);
  EXPECT_EQ(attr_node->GetOutNodes().size(), 0);
  EXPECT_EQ(attr_node->GetOutDataNodes().size(), 0);
  EXPECT_EQ(attr_node->InferShapeAndType(), GRAPH_PARAM_INVALID);
  EXPECT_EQ(attr_node->impl_->InferShapeAndType(attr_node), GRAPH_PARAM_INVALID);
  EXPECT_EQ(attr_node->GetOutDataNodesAndAnchors().size(), 0);
  EXPECT_EQ(data_node->NodeInConnectsAreEqual(*attr_node), false);
  EXPECT_EQ(data_node->NodeOutConnectsAreEqual(*attr_node), false);
  EXPECT_EQ(attr_node->NodeInConnectsAreEqual(*data_node), false);
  EXPECT_EQ(attr_node->NodeOutConnectsAreEqual(*data_node), false);
  EXPECT_EQ((*data_node)==(*attr_node), false);
  std::unordered_set<Node *> us;
  us.insert(data_node.get());
  EXPECT_EQ(attr_node->IsAllInNodesSeen(us), true);
  data_node->AddSendEventId(10);
  data_node->AddRecvEventId(20);
  EXPECT_EQ(data_node->GetSendEventIdList().size(), 1);
  EXPECT_EQ(data_node->GetRecvEventIdList().size(), 1);
  kFusionDataFlowVec_t fusion_input_list;
  data_node->GetFusionInputFlowList(fusion_input_list);
  data_node->SetFusionInputFlowList(fusion_input_list);
  EXPECT_EQ(fusion_input_list.size(), 0);
  kFusionDataFlowVec_t fusion_output_list;
  data_node->GetFusionOutputFlowList(fusion_output_list);
  data_node->SetFusionOutputFlowList(fusion_output_list);
  EXPECT_EQ(fusion_output_list.size(), 0);
  EXPECT_EQ(data_node->GetHostNode(), false);
  data_node->SetOrigNode(attr_node);
  EXPECT_NE(data_node->GetOrigNode(), nullptr);
  OpDescPtr opd = std::make_shared<OpDesc>("Opdesc","OpdType");
  EXPECT_EQ(data_node->UpdateOpDesc(opd), GRAPH_PARAM_INVALID);
}

TEST_F(UtestNode, NodeInConnectsAreEqual) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  data_node->impl_->in_data_anchors_.push_back(in_anch);
  EXPECT_EQ(data_node->GetAllInDataAnchors().size(),2);
  EXPECT_EQ(attr_node->GetAllInDataAnchors().size(),2);
  EXPECT_EQ(data_node->NodeInConnectsAreEqual(*attr_node),true);
}

TEST_F(UtestNode, NodeOutConnectsAreEqual) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 111);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, out_anch, 1), false);
  data_node->impl_->out_data_anchors_.push_back(out_anch);
  EXPECT_EQ(data_node->GetAllOutDataAnchors().size(),2);
  EXPECT_EQ(attr_node->GetAllOutDataAnchors().size(),2);
  EXPECT_EQ(data_node->NodeOutConnectsAreEqual(*attr_node),true);
}

TEST_F(UtestNode, NodeAnchorIsEqual) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch1 = std::make_shared<InDataAnchor>(data_node, 111);
  InDataAnchorPtr in_anch2 = std::make_shared<InDataAnchor>(attr_node, 222);
  OutDataAnchorPtr out_anch1 = std::make_shared<OutDataAnchor>(data_node, 333);
  EXPECT_EQ(in_anch1->LinkFrom(out_anch1), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch1, in_anch2, 2), false);
  OutDataAnchorPtr out_anch2 = std::make_shared<OutDataAnchor>(make_nullptr<Node>(), 444);
  EXPECT_EQ(in_anch2->LinkFrom(out_anch2), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch1, in_anch2, 2), false);
}

TEST_F(UtestNode, AddLink) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  data_node->impl_->op_->impl_->input_name_idx_["input_name"] = 10;
  data_node->impl_->op_->impl_->outputs_desc_.push_back(make_nullptr<GeTensorDesc>());
  auto odesc = data_node->GetOpDesc()->GetOutputDesc(0);
  attr_node->impl_->op_->impl_->input_name_idx_["__input3"] = 20;
  EXPECT_NE(attr_node->impl_->op_->impl_->input_name_idx_.find("__input3"), attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_FAILED);
}

TEST_F(UtestNode, AddLinkByIndex) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, nullptr, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, out_anch, 1), true);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(inc_anch), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(out_anch, inc_anch, 1),false);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  data_node->impl_->op_->impl_->input_name_idx_["input_name"] = 10;
  data_node->impl_->op_->impl_->outputs_desc_.push_back(make_nullptr<GeTensorDesc>());
  auto odesc = data_node->GetOpDesc()->GetOutputDesc(0);
  attr_node->impl_->op_->impl_->input_name_idx_["__input3"] = 20;
  EXPECT_NE(attr_node->impl_->op_->impl_->input_name_idx_.find("__input3"), attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_EQ(attr_node->AddLinkFrom(11, data_node), GRAPH_FAILED);
}

TEST_F(UtestNode, AddLinkByString) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, nullptr, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, out_anch, 1), true);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(inc_anch), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(out_anch, inc_anch, 1),false);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  data_node->impl_->op_->impl_->input_name_idx_["input_name"] = 10;
  data_node->impl_->op_->impl_->outputs_desc_.push_back(make_nullptr<GeTensorDesc>());
  auto odesc = data_node->GetOpDesc()->GetOutputDesc(0);
  attr_node->impl_->op_->impl_->input_name_idx_["__input3"] = 20;
  EXPECT_NE(attr_node->impl_->op_->impl_->input_name_idx_.find("__input3"), attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_EQ(attr_node->AddLinkFrom("__input3", data_node), GRAPH_FAILED);
  attr_node->impl_->op_->impl_->input_name_idx_["__input_succ"] = 5;
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_NE(attr_node->impl_->op_->impl_->input_name_idx_.find("__input_succ"), attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(attr_node->AddLinkFrom("__input_succ", data_node), GRAPH_FAILED);
}

TEST_F(UtestNode, AddLinkByStringInputDescFailure) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(nullptr, in_anch, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, nullptr, 1), false);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(in_anch, out_anch, 1), true);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(out_anch->LinkTo(inc_anch), GRAPH_SUCCESS);
  EXPECT_EQ(data_node->NodeAnchorIsEqual(out_anch, inc_anch, 1),false);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  data_node->impl_->op_->impl_->input_name_idx_["input_name"] = 10;
  data_node->impl_->op_->impl_->outputs_desc_.push_back(make_nullptr<GeTensorDesc>());
  auto odesc = data_node->GetOpDesc()->GetOutputDesc(0);
  attr_node->impl_->op_->impl_->input_name_idx_["__input5"] = -1;
  auto it = attr_node->impl_->op_->impl_->input_name_idx_.find("__input5");
  EXPECT_NE(it, attr_node->impl_->op_->impl_->input_name_idx_.end());
  EXPECT_EQ(it->second, -1);
  EXPECT_EQ(attr_node->impl_->op_->impl_->inputs_desc_.size(), 3);
  EXPECT_EQ(attr_node->impl_->op_->impl_->AddInputDesc("__input5", odesc), GRAPH_FAILED);
  EXPECT_EQ(attr_node->AddLinkFrom("__input5", data_node), GRAPH_FAILED);
}

}
