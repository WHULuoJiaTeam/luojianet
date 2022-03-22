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
#include "graph/passes/atomic_addr_clean_pass.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/pass_manager.h"
using namespace testing;

namespace ge {
class UtestGraphPassesAtomicAddrCleanPass : public Test {
public:
  UtestGraphPassesAtomicAddrCleanPass() {
    graph_ = std::make_shared<ComputeGraph>("test");
  }

  NodePtr NewNode(const string &name, const string &type, int input_cnt, int output_cnt) {
    OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
    for (int i = 0; i < input_cnt; ++i) {
      op_desc->AddInputDesc(GeTensorDesc());
    }
    for (int i = 0; i < output_cnt; ++i) {
      op_desc->AddOutputDesc(GeTensorDesc());
    }
    NodePtr node = graph_->AddNode(op_desc);
    return node;
  }

  int CountOfAtomicCleanNode() {
    int node_num = 0;
    for (NodePtr &node : graph_->GetDirectNode()) {
      if (node->GetType() == ATOMICADDRCLEAN) {
        ++node_num;
      }
    }
    return node_num;
  }

  ComputeGraphPtr graph_;
};

/*
 *     Data                       Data  Atomic_clean 
 *      |                           |   /  |
 *     relu                         relu   |
 *      |               ==>           |    |
 *    relu(atomic)               relu(atomic) 
 *      |                             |
 *   netoutput                    netoutput
 */
TEST_F(UtestGraphPassesAtomicAddrCleanPass, pass_run_success) {
  auto node1 = NewNode("node1", DATA, 0, 1);

  auto node2 = NewNode("node2", RELU, 1, 1);
  auto node3 = NewNode("node3", RELU, 1, 1);
  auto op_desc = node3->GetOpDesc();
  vector<int64_t> atomic_input_index = {123, 456};
  AttrUtils::SetListInt(op_desc, "atomic_input_index", atomic_input_index);

  auto node4 = NewNode("node4", NETOUTPUT, 1, 0);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  AtomicAddrCleanPass atomi_addr_clean_pass;
  Status ret = atomi_addr_clean_pass.Run(graph_);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(1, CountOfAtomicCleanNode());
  
  auto atomic_clean = graph_->FindNode("atomic_addr_clean");
  EXPECT_NE(atomic_clean, nullptr);
  auto out_ctrl_nodes = atomic_clean->GetOutControlNodes();
  EXPECT_EQ(out_ctrl_nodes.size(), 2);
}
}  // namespace ge
