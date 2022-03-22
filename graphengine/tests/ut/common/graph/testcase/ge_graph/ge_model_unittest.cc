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

#include "graph/model.h"

#include <gtest/gtest.h>

#include "graph/compute_graph.h"
#include "graph/debug/graph_debug.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGeModelUnittest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

ge::ComputeGraphPtr CreateSaveGraph() {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  // variable1
  ge::OpDescPtr variable_op = std::make_shared<ge::OpDesc>();
  variable_op->SetType("Variable");
  variable_op->SetName("Variable1");
  variable_op->AddInputDesc(ge::GeTensorDesc());
  variable_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr variable_node = graph->AddNode(variable_op);
  // save1
  ge::OpDescPtr save_op = std::make_shared<ge::OpDesc>();
  save_op->SetType("Save");
  save_op->SetName("Save1");
  save_op->AddInputDesc(ge::GeTensorDesc());
  save_op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr save_node = graph->AddNode(save_op);

  // add edge
  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), save_node->GetInDataAnchor(0));

  return graph;
}

TEST_F(UtestGeModelUnittest, save_model_to_file_success) {
  ge::ComputeGraphPtr compute_graph = CreateSaveGraph();
  auto all_nodes = compute_graph->GetAllNodes();
  for (auto node : all_nodes) {
    auto op_desc = node->GetOpDesc();
    GeTensorDesc weight_desc;
    op_desc->AddOptionalInputDesc("test", weight_desc);
    for (auto in_anchor_ptr : node->GetAllInDataAnchors()) {
      bool is_optional = op_desc->IsOptionalInput(in_anchor_ptr->GetIdx());
    }
  }
  ge::Graph ge_graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  string file_name = "model_data.pb";
  setenv("DUMP_MODEL", "1", true);
  setenv("DUMP_MODEL", "0", true);
}

TEST_F(UtestGeModelUnittest, load_model_from_file_success) {
  ge::Graph ge_graph;
  string file_name = "model_data.pb";
}
