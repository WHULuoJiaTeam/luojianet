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
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/pass_manager.h"
#include "graph_builder_utils.h"
#include "../utils/buffer_pool_graph_builder.h"
#include "graph/passes/buffer_pool_memory_pass.h"

namespace ge {
class UtestBufferPoolMemoryPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_normal_success_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;3");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add2;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;0");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add3;1");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add3");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_normal_graph_with_multi_buffer_pool_success_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraphWithMultiBufferPool");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraphWithMultiBufferPool();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;3");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;2");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add3;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add3");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;4");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_contain_one_node_success_test) {
  ut::BufferPoolGraphBuilder builder("SerialGraph");
  ge::ComputeGraphPtr graph = builder.BuildSerialGraph();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add1;2");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add2;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;0");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add3;1");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add3");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;1");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add4;2");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add4");
  }
}

TEST_F(UtestBufferPoolMemoryPass, calc_node_with_multi_buffer_pool_input_success_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiPrefetch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiPrefetch();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 0);
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 0);
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add1;2");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add2;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add2");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_in_different_subgraph_success_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithSubgraph");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithSubgraph();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  std::map<std::string, NodePtr> all_nodes;
  for (auto node : graph->GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    all_nodes[node->GetName()] = node;
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;3");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 0);
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;4");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 1);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "prefetch4");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_in_different_subgraph_with_inner_dependency_success_test) {
  ut::BufferPoolGraphBuilder builder("SubgraphWithInnerDependency");
  ge::ComputeGraphPtr graph = builder.BuildSubgraphWithInnerDependency();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  std::map<std::string, NodePtr> all_nodes;
  for (auto node : graph->GetAllNodes()) {
    EXPECT_NE(node, nullptr);
    all_nodes[node->GetName()] = node;
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add1;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add2;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add3;2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;add4;3");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 1);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "prefetch3");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = all_nodes.at("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;4");
    EXPECT_EQ(event_info.at(1), "RecvFrom;add3;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "add3");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_with_batch_label_success_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiBatch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiBatch();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add1;4");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add2;5");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add3;6");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add4;7");
    EXPECT_EQ(event_info.at(1), "RecvFrom;batch_label_256/add2;4");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "batch_label_256/add2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("batch_label_256/prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;batch_label_256/add5;4");
    EXPECT_EQ(event_info.at(1), "RecvFrom;batch_label_256/add3;5");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "batch_label_256/add3");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_node_has_multi_output_success_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiOutputPrefetch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiOutputPrefetch();

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch1");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;prefetch1_memcpy_async;0");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch2");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;prefetch2_memcpy_async;1");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch3");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 1);
    EXPECT_EQ(event_info.at(0), "SendTo;prefetch3_memcpy_async;2");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch4");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;prefetch4_memcpy_async;3");
    EXPECT_EQ(event_info.at(1), "RecvFrom;prefetch2_memcpy_async;0");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "prefetch2_memcpy_async");
  }

  {
    std::vector<std::string> event_info;
    auto prefetch = graph->FindNode("prefetch5");
    EXPECT_NE(prefetch, nullptr);
    (void) AttrUtils::GetListStr(prefetch->GetOpDesc(), "_event_multiplexing", event_info);
    EXPECT_EQ(event_info.size(), 2);
    EXPECT_EQ(event_info.at(0), "SendTo;add5;0");
    EXPECT_EQ(event_info.at(1), "RecvFrom;prefetch3_memcpy_async;1");
    auto in_ctrl_nodes = prefetch->GetInControlNodes();
    EXPECT_EQ(in_ctrl_nodes.size(), 2);
    EXPECT_EQ(in_ctrl_nodes.at(0)->GetName(), "prefetch3_memcpy_async");
  }
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_has_different_size_fail_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  const int64_t dummy_size = 256;
  auto prefetch = graph->FindNode("prefetch3");
  EXPECT_NE(prefetch, nullptr);
  (void) AttrUtils::SetInt(prefetch->GetOpDesc(), "_buffer_pool_size", dummy_size);

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_size_is_not_enough_fail_test) {
  ut::BufferPoolGraphBuilder builder("NormalGraph");
  ge::ComputeGraphPtr graph = builder.BuildNormalGraph();
  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;
  auto prefetch = graph->FindNode("prefetch3");
  EXPECT_NE(prefetch, nullptr);
  builder.SetPrefetchNodeInfo(prefetch, buffer_pool_id, buffer_pool_size, {buffer_pool_size + 512});

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_size_is_not_enough_for_multi_fail_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiPrefetch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiPrefetch();
  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;
  auto prefetch = graph->FindNode("prefetch3");
  EXPECT_NE(prefetch, nullptr);
  builder.SetPrefetchNodeInfo(prefetch, buffer_pool_id, buffer_pool_size, {buffer_pool_size});

  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, FAILED);
}

TEST_F(UtestBufferPoolMemoryPass, buffer_pool_node_has_multi_input_output_fail_test) {
  ut::BufferPoolGraphBuilder builder("GraphWithMultiInputOutputPrefetch");
  ge::ComputeGraphPtr graph = builder.BuildGraphWithMultiInputOutputPrefetch();
  BufferPoolMemoryPass buffer_pool_mem_pass;
  Status ret = buffer_pool_mem_pass.Run(graph);
  EXPECT_EQ(ret, FAILED);
}
}  // namespace ge
