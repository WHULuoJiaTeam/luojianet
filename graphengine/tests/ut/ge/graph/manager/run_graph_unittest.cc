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
#include <memory>

#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"

#define protected public
#define private public
#include"graph/manager/graph_manager_utils.h"
#include "graph/manager/graph_manager.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;
using domi::GetContext;

class UtestGraphRunTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { GetContext().out_nodes_map.clear(); }
};

TEST_F(UtestGraphRunTest, RunGraphWithStreamAsync) {
  GraphManager graph_manager;
  GeTensor input0, input1;
  std::vector<GeTensor> inputs{input0, input1};
  std::vector<GeTensor> outputs;
  GraphNodePtr graph_node = std::make_shared<GraphNode>(1);
  graph_manager.AddGraphNode(1, graph_node);
  GraphPtr graph = std::make_shared<Graph>("test");
  graph_node->SetGraph(graph);
  graph_node->SetRunFlag(false);
  graph_node->SetBuildFlag(true);
  auto ret = graph_manager.RunGraphWithStreamAsync(1, nullptr, 0, inputs, outputs);
}
