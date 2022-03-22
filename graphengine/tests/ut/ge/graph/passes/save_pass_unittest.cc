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

#include "graph/passes/save_pass.h"

#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "ge/ge_api.h"
#include "graph/compute_graph.h"
#include "graph/debug/graph_debug.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/pass_manager.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGraphPassesSavePass : public testing::Test {
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

  vector<ge::NodePtr> targets{save_node};
  graph->SetGraphTargetNodesInfo(targets);

  // add edge
  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), save_node->GetInDataAnchor(0));

  return graph;
}

TEST_F(UtestGraphPassesSavePass, cover_run_success) {
  ge::ComputeGraphPtr compute_graph = CreateSaveGraph();
  ge::PassManager pass_managers;
  pass_managers.AddPass("", new (std::nothrow) SavePass);
  Status status = pass_managers.Run(compute_graph);
  EXPECT_EQ(status, ge::SUCCESS);
}
