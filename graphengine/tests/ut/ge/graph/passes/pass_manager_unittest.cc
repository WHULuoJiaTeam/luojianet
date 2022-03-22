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
#include "inc/pass_manager.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

class SuccessGraphPass : public GraphPass {
  Status Run(ComputeGraphPtr graph) { return SUCCESS; }
};

class NotChangedGraphPass : public GraphPass {
  Status Run(ComputeGraphPtr graph) { return NOT_CHANGED; }
};

class ErrorGraphPass : public GraphPass {
  Status Run(ComputeGraphPtr graph) { return FAILED; }
};

class UtestGraphPassesPassManagerPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

NodePtr AddNode(ComputeGraphPtr graph) {
  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  OpDescPtr opdesc = make_shared<OpDesc>("test", "Add");
  opdesc->AddInputDesc(tensor_desc);
  opdesc->AddOutputDesc(tensor_desc);
  NodePtr node = graph->AddNode(opdesc);
  return node;
}

ComputeGraphPtr CreatePadGraph() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  return graph;
}

TEST_F(UtestGraphPassesPassManagerPass, all_pass_success) {
  PassManager manager;
  manager.AddPass("", new SuccessGraphPass);
  EXPECT_EQ(manager.GraphPasses().size(), 1);

  ComputeGraphPtr graph = CreatePadGraph();
  Status status = manager.Run(graph);
  EXPECT_EQ(SUCCESS, status);
}
/*
TEST_F(UtestGraphPassesPassManagerPass, graph_pass_success) {
  ComputeGraphPtr graph = CreatePadGraph();
  SuccessGraphPass pass;
  std::vector<std::pair<string, GraphPass*>> passes;
  Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(SUCCESS, status);
}
*/
TEST_F(UtestGraphPassesPassManagerPass, graph_pass_not_changed) {
  ComputeGraphPtr graph = CreatePadGraph();
  NotChangedGraphPass pass;
  std::vector<std::pair<string, GraphPass*>> passes;
  Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(NOT_CHANGED, status);
}
/*
TEST_F(UtestGraphPassesPassManagerPass, graph_pass_error) {
  ComputeGraphPtr graph = CreatePadGraph();
  ErrorGraphPass pass;
  std::vector<std::pair<string, GraphPass*>> passes;
  Status status = PassManager::Run(graph, passes);
  EXPECT_EQ(FAILED, status);
}
*/
