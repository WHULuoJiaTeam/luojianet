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
#include "graph/passes/global_step_insert_pass.h"

#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/passes/base_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/tuning_utils.h"
#include "graph_builder_utils.h"
#include "graph/ge_context.h"
#include "inc/pass_manager.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;

class UtestGlobalStepInsertPass : public Test {
 protected:
};

static ComputeGraphPtr BuildGraph1() {
  ge::ut::GraphBuilder builder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto identity1 = builder.AddNode("identity1", "Identity", 1, 1);
  auto out = builder.AddNode("out", "NetOutput", 1, 1);

  builder.AddDataEdge(var1, 0, identity1, 0);
  builder.AddControlEdge(var2, identity1);
  builder.AddDataEdge(identity1, 0, out, 0);
  return builder.GetGraph();
}

TEST_F(UtestGlobalStepInsertPass, skip_insert) {
  auto graph = BuildGraph1();
  GlobalStepInsertPass pass;
  Status status = pass.Run(graph);
  EXPECT_EQ(status, SUCCESS);
  NodePtr found_node = graph->FindNode(NODE_NAME_GLOBAL_STEP);
  EXPECT_NE(found_node, nullptr);
}
