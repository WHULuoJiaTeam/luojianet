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
#define protected public
#include "graph/partition/dynamic_shape_partition.h"
#include "compute_graph.h"
#include "graph/compute_graph_impl.h"
#include "inc/framework/common/types.h"
#include "utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "common/omg_util.h"

namespace ge {
namespace {
GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, Format format = FORMAT_NCHW,
                                 DataType data_type = DT_FLOAT) {
  GeShape ge_shape{vector<int64_t>(shape)};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  return tensor_desc;
}

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                            DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                             DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(GeTensorDescPtr tensor_desc) {
    op_desc_->AddOutputDesc(tensor_desc->Clone());
    return *this;
  }

  NodePtr Build(const ComputeGraphPtr &graph) {
    NodePtr node = graph->AddNode(op_desc_);
    return node;
  }

 private:
  OpDescPtr op_desc_;
};
}  // namespace

class UtestDynamicShapePartition : public testing::Test {
  protected:
    void SetUp() {}

    void TearDown() {}
};

TEST_F(UtestDynamicShapePartition, single_op_scene_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");

  NodePtr node1 =
      NodeBuilder("node1", CONSTANTOP).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr add_n_node =
      NodeBuilder("add_n_node", ADDN).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  NodePtr node2 =
      NodeBuilder("node2", RELU).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), add_n_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(add_n_node->GetOutDataAnchor(0), node2->GetInDataAnchor(0));

  (void)AttrUtils::SetBool(add_n_node->GetOpDesc(), ATTR_SINGLE_OP_SCENE, true);

  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
}

/*******************************************************************************
 *                |
 *              Merge1
 *      Active /      \ Active
 *            /        \.
 *           /          \.
 *        Merge2         \.
 *  Active/   \Active     \.
 *       /     \           \.
 *     Add      Sub       Relu
 *      |        |          |
 *      |        |          |
 * Switch_f2  Switch_t2     |
 *       \      /           |
 *        \    /            |
 *         Less2            |
 *           |              |
 *           |              |
 *       Switch_f      Switch_t
 *           |   \      /   |
 *           |    Active    |
 *           |       |      |
 *           |     Less1    |
 *           |     /   \    |
 *           |    /     \   |
 *            Data       Data
 ******************************************************************************/
TEST_F(UtestDynamicShapePartition, merge_control_flow_group) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  AttrUtils::SetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, "session_graph_id");

  auto data1 = NodeBuilder("data1", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto data2 = NodeBuilder("data2", DATA).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);

  auto less1 = NodeBuilder("less1", LESS).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto active1 = NodeBuilder("active1", STREAMACTIVE).Build(graph);
  auto switch_t = NodeBuilder("switch_t", STREAMSWITCH).AddInputDesc({1}).AddInputDesc({1}).Build(graph);
  auto switch_f = NodeBuilder("switch_f", STREAMSWITCH).AddInputDesc({1}).AddInputDesc({1}).Build(graph);
  auto const_01 = NodeBuilder("const_01", CONSTANT).AddOutputDesc({1}).Build(graph);
  auto const_11 = NodeBuilder("const_11", CONSTANT).AddOutputDesc({1}).Build(graph);


  auto less2 = NodeBuilder("less2", LESS).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto active2 = NodeBuilder("active2", STREAMACTIVE).Build(graph);
  auto switch_t2 = NodeBuilder("switch_t2", STREAMSWITCH).AddInputDesc({1}).AddInputDesc({1}).Build(graph);
  auto switch_f2 = NodeBuilder("switch_f2", STREAMSWITCH).AddInputDesc({1}).AddInputDesc({1}).Build(graph);
  auto const_02 = NodeBuilder("const_02", CONSTANT).AddOutputDesc({1}).Build(graph);
  auto const_12 = NodeBuilder("const_12", CONSTANT).AddOutputDesc({1}).Build(graph);

  auto add2 = NodeBuilder("add2", ADD).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto sub2 = NodeBuilder("sub2", SUB).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto merge2 = NodeBuilder("merge2", STREAMMERGE).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto active_f2 = NodeBuilder("active_f2", STREAMACTIVE).Build(graph);
  auto active_t2 = NodeBuilder("active_t2", STREAMACTIVE).Build(graph);

  auto relu1 = NodeBuilder("relu1", RELU).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto merge1 = NodeBuilder("merge1", STREAMMERGE).AddInputDesc({1}).AddInputDesc({1}).AddOutputDesc({1}).Build(graph);
  auto active_f1 = NodeBuilder("active_f1", STREAMACTIVE).Build(graph);
  auto active_t1 = NodeBuilder("active_t1", STREAMACTIVE).Build(graph);

  auto output1 = NodeBuilder("noutput1", NETOUTPUT).AddInputDesc({1}).Build(graph);

  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), less1->GetInDataAnchor(0));
  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), less1->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch_t->GetInDataAnchor(0));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), switch_f->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_01->GetOutDataAnchor(0), switch_t->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_11->GetOutDataAnchor(0), switch_f->GetInDataAnchor(1));
  GraphUtils::AddEdge(less1->GetOutControlAnchor(), active1->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), switch_t->GetInControlAnchor());
  GraphUtils::AddEdge(active1->GetOutControlAnchor(), switch_f->GetInControlAnchor());


  GraphUtils::AddEdge(data1->GetOutDataAnchor(0), less2->GetInDataAnchor(0));
  GraphUtils::AddEdge(less1->GetOutDataAnchor(0), less2->GetInDataAnchor(1));
  GraphUtils::AddEdge(less2->GetOutDataAnchor(0), switch_t2->GetInDataAnchor(0));
  GraphUtils::AddEdge(less2->GetOutDataAnchor(0), switch_f2->GetInDataAnchor(0));
  GraphUtils::AddEdge(const_02->GetOutDataAnchor(0), switch_t2->GetInDataAnchor(1));
  GraphUtils::AddEdge(const_12->GetOutDataAnchor(0), switch_f2->GetInDataAnchor(1));
  GraphUtils::AddEdge(less2->GetOutControlAnchor(), active2->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_t2->GetInControlAnchor());
  GraphUtils::AddEdge(active2->GetOutControlAnchor(), switch_f2->GetInControlAnchor());


  GraphUtils::AddEdge(switch_f2->GetOutControlAnchor(), add2->GetInControlAnchor());
  GraphUtils::AddEdge(less2->GetOutDataAnchor(0), add2->GetInDataAnchor(0));
  GraphUtils::AddEdge(add2->GetOutDataAnchor(0), merge2->GetInDataAnchor(0));
  GraphUtils::AddEdge(add2->GetOutControlAnchor(), active_f2->GetInControlAnchor());
  GraphUtils::AddEdge(active_f2->GetOutControlAnchor(), merge2->GetInControlAnchor());

  GraphUtils::AddEdge(switch_t2->GetOutControlAnchor(), sub2->GetInControlAnchor());
  GraphUtils::AddEdge(less2->GetOutDataAnchor(0), sub2->GetInDataAnchor(0));
  GraphUtils::AddEdge(sub2->GetOutDataAnchor(0), merge2->GetInDataAnchor(1));
  GraphUtils::AddEdge(sub2->GetOutControlAnchor(), active_t2->GetInControlAnchor());
  GraphUtils::AddEdge(active_t2->GetOutControlAnchor(), merge2->GetInControlAnchor());

  GraphUtils::AddEdge(switch_t->GetOutControlAnchor(), less2->GetInControlAnchor());
  GraphUtils::AddEdge(switch_f->GetOutControlAnchor(), relu1->GetInControlAnchor());


  GraphUtils::AddEdge(merge2->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(merge2->GetOutControlAnchor(), active_f1->GetInControlAnchor());
  GraphUtils::AddEdge(active_f1->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(data2->GetOutDataAnchor(0), relu1->GetInDataAnchor(1));
  GraphUtils::AddEdge(relu1->GetOutDataAnchor(0), merge1->GetInDataAnchor(0));
  GraphUtils::AddEdge(relu1->GetOutControlAnchor(), active_t1->GetInControlAnchor());
  GraphUtils::AddEdge(active_t1->GetOutControlAnchor(), merge1->GetInControlAnchor());

  GraphUtils::AddEdge(merge1->GetOutDataAnchor(0), output1->GetInDataAnchor(0));

  AttrUtils::SetBool(merge2->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, true);
  EXPECT_EQ(graph->TopologicalSorting(), GRAPH_SUCCESS);

  SetControlFlowGroup(merge2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_f2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_t2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(active2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(active_t2, merge2->GetOpDesc()->GetId());
  SetControlFlowGroup(active_f2, merge2->GetOpDesc()->GetId());

  SetControlFlowGroup(merge1, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_f, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(switch_t, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(active1, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(active_f1, merge1->GetOpDesc()->GetId());
  SetControlFlowGroup(active_t1, merge1->GetOpDesc()->GetId());

  EXPECT_EQ(graph->impl_->sub_graph_.size(), 0);
  DynamicShapePartitioner partitioner(graph);
  EXPECT_EQ(partitioner.Partition(), SUCCESS);
  EXPECT_EQ(graph->impl_->sub_graph_.size(), 3);   // input  less1  uknown
}
} // namespace ge