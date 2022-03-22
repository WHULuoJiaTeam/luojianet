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

#include "graph/passes/multi_batch_clone_pass.h"

#include <gtest/gtest.h>
#include <set>
#include <string>

#include "inc/pass_manager.h"
#include "graph/utils/tensor_utils.h"
#include "common/local_context.h"
#include "graph/passes/multi_batch_pass.h"
#include "graph/preprocess/multi_batch_copy_graph.h"
#include "graph/preprocess/insert_op/util_insert_aipp_op.h"
#include "framework/omg/omg_inner_types.h"
#include "register/op_registry.h"


namespace ge{
class UtestMultiBatchClonePass : public testing::Test {
protected:
  void SetUp() {
    SetLocalOmgContext(domi::GetContext());
    GetLocalOmgContext().dynamic_image_size.clear();
    GetLocalOmgContext().dynamic_batch_size.clear();
  }
  void TearDown() {
    GetLocalOmgContext().dynamic_image_size.clear();
    GetLocalOmgContext().dynamic_batch_size.clear();
    GetLocalOmgContext().dynamic_node_type.clear();
  }

public:
  NodePtr MakeNode(const ComputeGraphPtr &graph, int in_num, int out_num, string name, string type) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    return graph->AddNode(op_desc);
  }

  NodePtr MakeConstNode(const ComputeGraphPtr &graph) {
    static uint32_t index = 0;
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>("dynamic_const_" + std::to_string(index++), "Const");
    op_desc->AddOutputDesc(test_desc);
    return graph->AddNode(op_desc);
  }

  void make_original_graph(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 1, "data", "Data");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto bn_conv1 = MakeNode(graph, 4, 1, "bn_conv1", "BNInference");
    {
      GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), bn_conv1->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), bn_conv1->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), bn_conv1->GetInDataAnchor(2));
      auto const3= MakeConstNode(graph);
      GraphUtils::AddEdge(const3->GetOutDataAnchor(0), bn_conv1->GetInDataAnchor(3));
    }

    auto scale_conv1 = MakeNode(graph, 4, 1, "scale1", "Scale");
    {
      GraphUtils::AddEdge(bn_conv1->GetOutDataAnchor(0), scale_conv1->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), scale_conv1->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), scale_conv1->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(scale_conv1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void GraphWithJustData(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 1, "data", "Data");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void GraphWithGetNextNosink(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 1, "IteratorGetNext_data", "Data");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  // getnext has one data and has one out of shape
  void GraphWithGetNextSink(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 2, "data", "IteratorV2");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      GeTensorDesc shape_desc(GeShape({4,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(1, shape_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto identity = MakeNode(graph, 1, 0, "identity", "Identity");
      GraphUtils::AddEdge(data1->GetOutDataAnchor(1), identity->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }
};

// graph is nullptr
TEST_F(UtestMultiBatchClonePass, graph_nullptr) {
  PassManager pass_manager;
  pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass);
  ComputeGraphPtr graph;
  EXPECT_EQ(pass_manager.Run(graph), PARAM_INVALID);
}

// graph with subgraph
TEST_F(UtestMultiBatchClonePass, graph_with_subgraph) {
  PassManager pass_manager;
  pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_original_graph(graph);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);

  ComputeGraphPtr owner = std::make_shared<ComputeGraph>("test_owner");
  auto func_node = MakeNode(owner, 3, 1, "test_if", "If");
  graph->SetParentNode(func_node);
  graph->SetParentGraph(owner);
  owner->AddSubgraph(graph->GetName(), graph);
  size_t sub_graph_num = owner->GetAllSubgraphs().size();
  EXPECT_EQ(sub_graph_num, 1);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
}

//graph is uncompute graph, not need to do multi batch
TEST_F(UtestMultiBatchClonePass, uncompute_graph) {
  MultiBatchClonePass multi_batch_clone;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_original_graph(graph);
  GetLocalOmgContext().need_multi_batch = false;
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
}


//compute_graph with data from DATA
TEST_F(UtestMultiBatchClonePass, compute_graph_with_data) {
  MultiBatchClonePass multi_batch_clone;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithJustData(graph);
  GetLocalOmgContext().need_multi_batch = true;
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  GetLocalOmgContext().dynamic_node_type = DATA;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().data_nodes.size(), 1);
}

//compute_graph with data from GetNext_nosink
TEST_F(UtestMultiBatchClonePass, compute_graph_with_getnext_nosink) {
  MultiBatchClonePass multi_batch_clone;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithGetNextNosink(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = GETNEXT;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().getnext_nosink_nodes.size(), 1);
}

//compute_graph with data from GetNext_nosink
TEST_F(UtestMultiBatchClonePass, compute_graph_with_getnext_sink) {
  MultiBatchClonePass multi_batch_clone;
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithGetNextSink(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = GETNEXT;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().getnext_nosink_nodes.size(), 0);
}

}
