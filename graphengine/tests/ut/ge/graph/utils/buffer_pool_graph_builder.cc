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
#include "buffer_pool_graph_builder.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"

namespace ge {
namespace ut {
BufferPoolGraphBuilder::BufferPoolGraphBuilder(const std::string &name) {
  graph_name_ = name;
}

BufferPoolGraphBuilder::InnerGraphBuilder::InnerGraphBuilder(const std::string &name) {
  graph_ = std::make_shared<ComputeGraph>(name);
  EXPECT_NE(graph_, nullptr);
}

NodePtr BufferPoolGraphBuilder::InnerGraphBuilder::AddNode(const std::string &name, const std::string &type,
                                                           int in_cnt, int out_cnt,
                                                           Format format, DataType data_type,
                                                           std::vector<int64_t> shape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  EXPECT_NE(tensor_desc, nullptr);
  tensor_desc->SetShape(GeShape(std::move(shape)));
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  auto op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_NE(op_desc, nullptr);
  for (int i = 0; i < in_cnt; ++i) {
    op_desc->AddInputDesc(tensor_desc->Clone());
  }
  for (int i = 0; i < out_cnt; ++i) {
    op_desc->AddOutputDesc(tensor_desc->Clone());
  }
  return graph_->AddNode(op_desc);
}

void BufferPoolGraphBuilder::InnerGraphBuilder::AddDataEdge(NodePtr &src_node, int src_idx,
                                                            NodePtr &dst_node, int dst_idx) {
  EXPECT_NE(src_node, nullptr);
  EXPECT_NE(dst_node, nullptr);
  GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
}

void BufferPoolGraphBuilder::InnerGraphBuilder::AddControlEdge(NodePtr &src_node, NodePtr &dst_node) {
  EXPECT_NE(src_node, nullptr);
  EXPECT_NE(dst_node, nullptr);
  GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
}

void BufferPoolGraphBuilder::SetBufferPool(NodePtr &node, int64_t pool_id, int64_t pool_size,
                                           const std::string &batch_label) {
  EXPECT_NE(node, nullptr);
  (void) AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_BUFFER_POOL_ID, pool_id);
  (void) AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_BUFFER_POOL_SIZE, pool_size);
  if (!batch_label.empty()) {
    (void) AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);
  }
}

void BufferPoolGraphBuilder::SetBatchLabel(NodePtr &node, const std::string &batch_label) {
  EXPECT_NE(node, nullptr);
  (void) AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, batch_label);

}

void BufferPoolGraphBuilder::SetOutputMemSize(NodePtr &node, const std::vector<int64_t> &mem_size) {
  EXPECT_NE(node, nullptr);
  EXPECT_NE(node->GetOpDesc(), nullptr);
  size_t output_size = node->GetOpDesc()->GetOutputsSize();
  EXPECT_EQ(output_size, mem_size.size());
  for (size_t i = 0; i < output_size; ++i) {
    auto output_op_desc = node->GetOpDesc()->MutableOutputDesc(i);
    ge::TensorUtils::SetSize(*output_op_desc, mem_size[i]);
  }
}

void BufferPoolGraphBuilder::SetWorkSpaceMemSize(NodePtr &node, const std::vector<int64_t> &ws_bytes) {
  EXPECT_NE(node, nullptr);
  EXPECT_NE(node->GetOpDesc(), nullptr);
  node->GetOpDesc()->SetWorkspaceBytes(ws_bytes);
}

void BufferPoolGraphBuilder::SetPrefetchNodeInfo(NodePtr &node, int64_t pool_id, int64_t pool_size,
                                                 const std::vector<int64_t> &mem_size,
                                                 const std::vector<int64_t> &ws_bytes,
                                                 const std::string &batch_label) {
  SetBufferPool(node, pool_id, pool_size, batch_label);
  SetOutputMemSize(node, mem_size);
  SetWorkSpaceMemSize(node, ws_bytes);
}

///
/// Normal graph
///
///             w1         w2         w3         w4         w5
///              \          \          \         \          \.
///          prefetch1  prefetch2  prefetch3  prefetch4  prefetch5
///               \          \          \         \          \.
/// const1 ----- add1 ----- add2 ----- add3 ----- add4 ----- add5 ----- net_output
///
///
///  Memory distribution:
///
///      |___w1__|__w2__|__w3__|__|
///
///      |_____w4_____|_____w5____|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildNormalGraph() {
  auto builder = InnerGraphBuilder(graph_name_);
  auto w1 = builder.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = builder.AddNode("w2", VARIABLE, 0, 1);
  auto w3 = builder.AddNode("w3", VARIABLE, 0, 1);
  auto w4 = builder.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = builder.AddNode("w5", VARIABLE, 0, 1);

  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;

  auto prefetch1 = builder.AddNode("prefetch1", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch2 = builder.AddNode("prefetch2", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch3 = builder.AddNode("prefetch3", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch4 = builder.AddNode("prefetch4", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024});
  auto prefetch5 = builder.AddNode("prefetch5", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024});

  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto add2 = builder.AddNode("add2", ADD, 2, 1);
  auto add3 = builder.AddNode("add3", ADD, 2, 1);
  auto add4 = builder.AddNode("add4", ADD, 2, 1);
  auto add5 = builder.AddNode("add5", ADD, 2, 1);
  auto const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(w1, 0, prefetch1, 0);
  builder.AddDataEdge(w2, 0, prefetch2, 0);
  builder.AddDataEdge(w3, 0, prefetch3, 0);
  builder.AddDataEdge(w4, 0, prefetch4, 0);
  builder.AddDataEdge(w5, 0, prefetch5, 0);

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(prefetch1, 0, add1, 1);

  builder.AddDataEdge(add1, 0, add2, 0);
  builder.AddDataEdge(prefetch2, 0, add2, 1);

  builder.AddDataEdge(add2, 0, add3, 0);
  builder.AddDataEdge(prefetch3, 0, add3, 1);

  builder.AddDataEdge(add3, 0, add4, 0);
  builder.AddDataEdge(prefetch4, 0, add4, 1);

  builder.AddDataEdge(add4, 0, add5, 0);
  builder.AddDataEdge(prefetch5, 0, add5, 1);

  builder.AddDataEdge(add5, 0, net_output, 0);

  auto compute_graph = builder.GetGraph();

  return compute_graph;
}

///
/// Normal graph with multi buffer pool
///
///             w1         w2         w3         w4         w5
///              \          \          \         \          \.
  ///          prefetch1  prefetch2  prefetch3  prefetch4  prefetch5
///            (pool0)    (pool1)    (pool0)   (pool0)    (pool1)
///               \          \          \         \          \.
  /// const1 ----- add1 ----- add2 ----- add3 ----- add4 ----- add5 ----- net_output
///
///
///  Memory distribution:
///
///      |___w1__|__w3__|_________|
///      |_____w4_____|___________|
///
///      |___w2__|_____w5___|_____|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildNormalGraphWithMultiBufferPool() {
  auto builder = InnerGraphBuilder(graph_name_);
  auto w1 = builder.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = builder.AddNode("w2", VARIABLE, 0, 1);
  auto w3 = builder.AddNode("w3", VARIABLE, 0, 1);
  auto w4 = builder.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = builder.AddNode("w5", VARIABLE, 0, 1);

  const int64_t buffer_pool_id_0 = 0;
  const int64_t buffer_pool_id_1 = 1;
  const int64_t buffer_pool_size = 5000;

  auto prefetch1 = builder.AddNode("prefetch1", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch1, buffer_pool_id_0, buffer_pool_size, {500});
  auto prefetch2 = builder.AddNode("prefetch2", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch2, buffer_pool_id_1, buffer_pool_size, {500});
  auto prefetch3 = builder.AddNode("prefetch3", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch3, buffer_pool_id_0, buffer_pool_size, {500});
  auto prefetch4 = builder.AddNode("prefetch4", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch4, buffer_pool_id_0, buffer_pool_size, {1024});
  auto prefetch5 = builder.AddNode("prefetch5", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch5, buffer_pool_id_1, buffer_pool_size, {1024});

  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto add2 = builder.AddNode("add2", ADD, 2, 1);
  auto add3 = builder.AddNode("add3", ADD, 2, 1);
  auto add4 = builder.AddNode("add4", ADD, 2, 1);
  auto add5 = builder.AddNode("add5", ADD, 2, 1);
  auto const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(w1, 0, prefetch1, 0);
  builder.AddDataEdge(w2, 0, prefetch2, 0);
  builder.AddDataEdge(w3, 0, prefetch3, 0);
  builder.AddDataEdge(w4, 0, prefetch4, 0);
  builder.AddDataEdge(w5, 0, prefetch5, 0);

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(prefetch1, 0, add1, 1);

  builder.AddDataEdge(add1, 0, add2, 0);
  builder.AddDataEdge(prefetch2, 0, add2, 1);

  builder.AddDataEdge(add2, 0, add3, 0);
  builder.AddDataEdge(prefetch3, 0, add3, 1);

  builder.AddDataEdge(add3, 0, add4, 0);
  builder.AddDataEdge(prefetch4, 0, add4, 1);

  builder.AddDataEdge(add4, 0, add5, 0);
  builder.AddDataEdge(prefetch5, 0, add5, 1);

  builder.AddDataEdge(add5, 0, net_output, 0);

  auto compute_graph = builder.GetGraph();

  return compute_graph;
}

///
/// SerialGraph: Buffer pool size only can contain one prefetch node
///
///             w1         w2         w3         w4         w5
///              \          \          \         \          \.
///          prefetch1  prefetch2  prefetch3  prefetch4  prefetch5
///               \          \          \         \          \.
/// const1 ----- add1 ----- add2 ----- add3 ----- add4 ----- add5 ----- net_output
///
///
///  Memory distribution:
///
///      |____w1_____|__|
///
///      |____w2_____|__|
///
///      |____w3_____|__|
///
///      |______w4______|
///
///      |______w5______|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildSerialGraph() {
  auto builder = InnerGraphBuilder(graph_name_);
  auto w1 = builder.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = builder.AddNode("w2", VARIABLE, 0, 1);
  auto w3 = builder.AddNode("w3", VARIABLE, 0, 1);
  auto w4 = builder.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = builder.AddNode("w5", VARIABLE, 0, 1);

  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 2048;

  auto prefetch1 = builder.AddNode("prefetch1", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch2 = builder.AddNode("prefetch2", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch3 = builder.AddNode("prefetch3", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch4 = builder.AddNode("prefetch4", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024});
  auto prefetch5 = builder.AddNode("prefetch5", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024});

  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto add2 = builder.AddNode("add2", ADD, 2, 1);
  auto add3 = builder.AddNode("add3", ADD, 2, 1);
  auto add4 = builder.AddNode("add4", ADD, 2, 1);
  auto add5 = builder.AddNode("add5", ADD, 2, 1);
  auto const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(w1, 0, prefetch1, 0);
  builder.AddDataEdge(w2, 0, prefetch2, 0);
  builder.AddDataEdge(w3, 0, prefetch3, 0);
  builder.AddDataEdge(w4, 0, prefetch4, 0);
  builder.AddDataEdge(w5, 0, prefetch5, 0);

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(prefetch1, 0, add1, 1);

  builder.AddDataEdge(add1, 0, add2, 0);
  builder.AddDataEdge(prefetch2, 0, add2, 1);

  builder.AddDataEdge(add2, 0, add3, 0);
  builder.AddDataEdge(prefetch3, 0, add3, 1);

  builder.AddDataEdge(add3, 0, add4, 0);
  builder.AddDataEdge(prefetch4, 0, add4, 1);

  builder.AddDataEdge(add4, 0, add5, 0);
  builder.AddDataEdge(prefetch5, 0, add5, 1);

  builder.AddDataEdge(add5, 0, net_output, 0);

  auto compute_graph = builder.GetGraph();

  return compute_graph;
}

///
/// GraphWithMultiPrefetch: Calc node with more prefetch node
///
///            w1          w2         w3         w4       w5
///             \           \          \          \        \.
///          prefetch1  prefetch2  prefetch3  prefetch4  prefetch5  const1
///               \         /           \         /          \       /
///                \       /             \       /            \     /
///                 \     /               \     /              \   /
///                  add1 ------ c ------- add2 ----- c -----  add3
///                   |                     |                   |
///                   |                     |                   |
///                    ---------------  net_output  ------------
///
///  Memory distribution:
///
///      |___w1__|__w2__|__w3__|__|
///
///      |_____w4_____|_____w5____|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildGraphWithMultiPrefetch() {
  auto builder = InnerGraphBuilder(graph_name_);
  auto w1 = builder.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = builder.AddNode("w2", VARIABLE, 0, 1);
  auto w3 = builder.AddNode("w3", VARIABLE, 0, 1);
  auto w4 = builder.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = builder.AddNode("w5", VARIABLE, 0, 1);

  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;

  auto prefetch1 = builder.AddNode("prefetch1", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch2 = builder.AddNode("prefetch2", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch3 = builder.AddNode("prefetch3", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch4 = builder.AddNode("prefetch4", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024});
  auto prefetch5 = builder.AddNode("prefetch5", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024});

  auto const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto add2 = builder.AddNode("add2", ADD, 2, 1);
  auto add3 = builder.AddNode("add3", ADD, 2, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 3, 0);

  builder.AddDataEdge(w1, 0, prefetch1, 0);
  builder.AddDataEdge(w2, 0, prefetch2, 0);
  builder.AddDataEdge(w3, 0, prefetch3, 0);
  builder.AddDataEdge(w4, 0, prefetch4, 0);
  builder.AddDataEdge(w5, 0, prefetch5, 0);

  builder.AddDataEdge(prefetch1, 0, add1, 0);
  builder.AddDataEdge(prefetch2, 0, add1, 1);

  builder.AddDataEdge(prefetch3, 0, add2, 0);
  builder.AddDataEdge(prefetch4, 0, add2, 1);

  builder.AddDataEdge(const1, 0, add3, 0);
  builder.AddDataEdge(prefetch5, 0, add3, 1);

  builder.AddDataEdge(add1, 0, net_output, 0);
  builder.AddDataEdge(add2, 0, net_output, 1);
  builder.AddDataEdge(add3, 0, net_output, 2);

  builder.AddControlEdge(add1, add2);
  builder.AddControlEdge(add2, add3);

  auto compute_graph = builder.GetGraph();

  return compute_graph;
}

///
/// GraphWithSubgraph: Calc node in different subgraph
///
///
///               call_node1(with Subgraph1) --------------- call_node2 (with Subgraph2)  --------------- net_output
///
///
///   Subgraph1:                                                    Subgraph2:
///
///             w1         w2         w3                                      w4         w5
///              \          \          \                                      \          \.
///          prefetch1  prefetch2  prefetch3                               prefetch4  prefetch5
///               \          \          \                                      \          \.
/// const1 ----- add1 ----- add2 ----- add3 ---- subgraph1_out      data1 ---- add4 ----- add5 ---- subgraph2_out
///
///
///  Memory distribution:
///
///      |___w1__|__w2__|__w3__|__|
///
///      |_____w4_____|_____w5____|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildGraphWithSubgraph() {
  auto builder = InnerGraphBuilder(graph_name_);

  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;

  // Subgraph1
  auto subgraph_builder1 = InnerGraphBuilder("Subgraph1");
  auto w1 = subgraph_builder1.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = subgraph_builder1.AddNode("w2", VARIABLE, 0, 1);
  auto w3 = subgraph_builder1.AddNode("w3", VARIABLE, 0, 1);

  auto prefetch1 = subgraph_builder1.AddNode("prefetch1", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch2 = subgraph_builder1.AddNode("prefetch2", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch3 = subgraph_builder1.AddNode("prefetch3", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500});
  auto subgraph1_out = subgraph_builder1.AddNode("subgraph1_out", NETOUTPUT, 1, 0);
  auto const1 = subgraph_builder1.AddNode("const1", CONSTANTOP, 0, 1);

  auto add1 = subgraph_builder1.AddNode("add1", ADD, 2, 1);
  auto add2 = subgraph_builder1.AddNode("add2", ADD, 2, 1);
  auto add3 = subgraph_builder1.AddNode("add3", ADD, 2, 1);

  subgraph_builder1.AddDataEdge(w1, 0, prefetch1, 0);
  subgraph_builder1.AddDataEdge(w2, 0, prefetch2, 0);
  subgraph_builder1.AddDataEdge(w3, 0, prefetch3, 0);
  subgraph_builder1.AddDataEdge(const1, 0, add1, 0);
  subgraph_builder1.AddDataEdge(prefetch1, 0, add1, 1);
  subgraph_builder1.AddDataEdge(add1, 0, add2, 0);
  subgraph_builder1.AddDataEdge(prefetch2, 0, add2, 1);
  subgraph_builder1.AddDataEdge(add2, 0, add3, 0);
  subgraph_builder1.AddDataEdge(prefetch3, 0, add3, 1);
  subgraph_builder1.AddDataEdge(add3, 0, subgraph1_out, 0);
  auto subgraph1 = subgraph_builder1.GetGraph();
  for (auto &node : subgraph1->GetDirectNode()) {
    node->SetOwnerComputeGraph(subgraph1);
  }

  // Subgraph2
  auto subgraph_builder2 = InnerGraphBuilder("Subgraph2");
  auto w4 = subgraph_builder2.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = subgraph_builder2.AddNode("w5", VARIABLE, 0, 1);

  auto prefetch4 = subgraph_builder2.AddNode("prefetch4", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024});
  auto prefetch5 = subgraph_builder2.AddNode("prefetch5", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024});

  auto add4 = subgraph_builder2.AddNode("add4", ADD, 2, 1);
  auto add5 = subgraph_builder2.AddNode("add5", ADD, 2, 1);
  auto data1 = subgraph_builder2.AddNode("data1", DATA, 0, 1);
  auto subgraph2_out = subgraph_builder2.AddNode("subgraph2_out", NETOUTPUT, 1, 1);

  subgraph_builder2.AddDataEdge(w4, 0, prefetch4, 0);
  subgraph_builder2.AddDataEdge(w5, 0, prefetch5, 0);
  subgraph_builder2.AddDataEdge(data1, 0, add4, 0);
  subgraph_builder2.AddDataEdge(prefetch4, 0, add4, 1);
  subgraph_builder2.AddDataEdge(add4, 0, add5, 0);
  subgraph_builder2.AddDataEdge(prefetch5, 0, add5, 1);
  subgraph_builder2.AddDataEdge(add5, 0, subgraph2_out, 0);

  auto subgraph2 = subgraph_builder2.GetGraph();
  for (auto &node : subgraph2->GetDirectNode()) {
    node->SetOwnerComputeGraph(subgraph2);
  }

  // root graph
  auto call_node1 = builder.AddNode("call_node1", PARTITIONEDCALL, 0, 1);
  auto call_node2 = builder.AddNode("call_node2", PARTITIONEDCALL, 1, 0);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(call_node1, 0, call_node2, 0);
  builder.AddDataEdge(call_node2, 0, net_output, 0);
  auto compute_graph = builder.GetGraph();
  call_node1->SetOwnerComputeGraph(compute_graph);
  call_node1->GetOpDesc()->AddSubgraphName(subgraph1->GetName());
  call_node1->GetOpDesc()->SetSubgraphInstanceName(0, subgraph1->GetName());
  call_node2->SetOwnerComputeGraph(compute_graph);
  call_node2->GetOpDesc()->AddSubgraphName(subgraph2->GetName());
  call_node2->GetOpDesc()->SetSubgraphInstanceName(0, subgraph2->GetName());

  subgraph1->SetParentNode(call_node1);
  subgraph1->SetParentGraph(compute_graph);
  subgraph2->SetParentNode(call_node2);
  subgraph2->SetParentGraph(compute_graph);
  compute_graph->AddSubGraph(subgraph1);
  compute_graph->AddSubGraph(subgraph2);

  return compute_graph;
}

///
/// SubgraphWithInnerDependency: Calc node in different subgraph with inner dependency
///
///
///              call_node1(with Subgraph1) --------------------- call_node2 (with Subgraph2)  ---------- net_output
///
///
///   Subgraph1:                                       Subgraph2:
///
///             w1         w2                                      w3         w4         w5
///              \          \                                       \         \          \.
///          prefetch1  prefetch2                               prefetch3  prefetch4  prefetch5
///               \          \                                       \         \          \.
/// const1 ----- add1 ----- add2 ----- subgraph1_out     data1 ---- add3 ---- add4 ----- add5 ---- subgraph2_out
///
///
///  Memory distribution:
///
///      |___w1__|__w2__|__w3__|__|
///
///      |_____w4_____|_____w5____|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildSubgraphWithInnerDependency() {
  auto builder = InnerGraphBuilder(graph_name_);

  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;

  // Subgraph1
  auto subgraph_builder1 = InnerGraphBuilder("Subgraph1");
  auto w1 = subgraph_builder1.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = subgraph_builder1.AddNode("w2", VARIABLE, 0, 1);

  auto prefetch1 = subgraph_builder1.AddNode("prefetch1", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch2 = subgraph_builder1.AddNode("prefetch2", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500});
  auto subgraph1_out = subgraph_builder1.AddNode("subgraph1_out", NETOUTPUT, 1, 0);
  auto const1 = subgraph_builder1.AddNode("const1", CONSTANTOP, 0, 1);

  auto add1 = subgraph_builder1.AddNode("add1", ADD, 2, 1);
  auto add2 = subgraph_builder1.AddNode("add2", ADD, 2, 1);

  subgraph_builder1.AddDataEdge(w1, 0, prefetch1, 0);
  subgraph_builder1.AddDataEdge(w2, 0, prefetch2, 0);
  subgraph_builder1.AddDataEdge(const1, 0, add1, 0);
  subgraph_builder1.AddDataEdge(prefetch1, 0, add1, 1);
  subgraph_builder1.AddDataEdge(add1, 0, add2, 0);
  subgraph_builder1.AddDataEdge(prefetch2, 0, add2, 1);
  subgraph_builder1.AddDataEdge(add2, 0, subgraph1_out, 0);
  auto subgraph1 = subgraph_builder1.GetGraph();
  for (auto &node : subgraph1->GetDirectNode()) {
    node->SetOwnerComputeGraph(subgraph1);
  }

  // Subgraph2
  auto subgraph_builder2 = InnerGraphBuilder("Subgraph2");
  auto w3 = subgraph_builder2.AddNode("w3", VARIABLE, 0, 1);
  auto w4 = subgraph_builder2.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = subgraph_builder2.AddNode("w5", VARIABLE, 0, 1);

  auto prefetch3 = subgraph_builder2.AddNode("prefetch3", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch4 = subgraph_builder2.AddNode("prefetch4", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024});
  auto prefetch5 = subgraph_builder2.AddNode("prefetch5", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024});

  auto add3 = subgraph_builder2.AddNode("add3", ADD, 2, 1);
  auto add4 = subgraph_builder2.AddNode("add4", ADD, 2, 1);
  auto add5 = subgraph_builder2.AddNode("add5", ADD, 2, 1);
  auto data1 = subgraph_builder2.AddNode("data1", DATA, 0, 1);
  auto subgraph2_out = subgraph_builder2.AddNode("subgraph2_out", NETOUTPUT, 1, 1);

  subgraph_builder2.AddDataEdge(w3, 0, prefetch3, 0);
  subgraph_builder2.AddDataEdge(w4, 0, prefetch4, 0);
  subgraph_builder2.AddDataEdge(w5, 0, prefetch5, 0);
  subgraph_builder2.AddDataEdge(data1, 0, add3, 0);
  subgraph_builder2.AddDataEdge(prefetch3, 0, add3, 1);
  subgraph_builder2.AddDataEdge(add3, 0, add4, 0);
  subgraph_builder2.AddDataEdge(prefetch4, 0, add4, 1);
  subgraph_builder2.AddDataEdge(add4, 0, add5, 0);
  subgraph_builder2.AddDataEdge(prefetch5, 0, add5, 1);
  subgraph_builder2.AddDataEdge(add5, 0, subgraph2_out, 0);

  auto subgraph2 = subgraph_builder2.GetGraph();
  for (auto &node : subgraph2->GetDirectNode()) {
    node->SetOwnerComputeGraph(subgraph2);
  }

  // root graph
  auto call_node1 = builder.AddNode("call_node1", PARTITIONEDCALL, 0, 1);
  auto call_node2 = builder.AddNode("call_node2", PARTITIONEDCALL, 1, 0);
  auto net_output = subgraph_builder2.AddNode("net_output", NETOUTPUT, 1, 0);
  builder.AddDataEdge(call_node1, 0, call_node2, 0);
  builder.AddDataEdge(call_node2, 0, net_output, 0);
  auto compute_graph = builder.GetGraph();
  call_node1->SetOwnerComputeGraph(compute_graph);
  call_node1->GetOpDesc()->AddSubgraphName(subgraph1->GetName());
  call_node1->GetOpDesc()->SetSubgraphInstanceName(0, subgraph1->GetName());
  call_node2->SetOwnerComputeGraph(compute_graph);
  call_node2->GetOpDesc()->AddSubgraphName(subgraph2->GetName());
  call_node2->GetOpDesc()->SetSubgraphInstanceName(0, subgraph2->GetName());

  subgraph1->SetParentNode(call_node1);
  subgraph1->SetParentGraph(compute_graph);
  subgraph2->SetParentNode(call_node2);
  subgraph2->SetParentGraph(compute_graph);
  compute_graph->AddSubGraph(subgraph1);
  compute_graph->AddSubGraph(subgraph2);

  return compute_graph;
}

///
/// BuildGraphWithMultiBatch: Different batch label
///
///
///                                                        batch_label_128
///
///                              const1 ----- add1 ----- add2 ----- add3 ----- add4 ----- add5 ---
///                             /              /          /          /         /          /       \.
///  						               /c        prefetch1  prefetch2  prefetch3  prefetch4  prefetch5     \.
///     const1        switch_false           /          /          /         /          /           \.
///         \         /                     /          /          /         /          /             \.
///           switch1                      w1         w2         w3        w4         w5           merge1 -- net_output
///  	     /          \                     \          \          \         \          \             /
///     const2        switch_true            \          \          \         \          \           /
///                            \c        prefetch1  prefetch2  prefetch3  prefetch4  prefetch5     /
///                             \              \          \          \         \          \       /
///                              const1 ----- add1 ----- add2 ----- add3 ----- add4 ----- add5 ---
///
///                                                       batch_label_256
///
///
///  Memory distribution:
///
///      |___w1__|__w2__|__w3__|__|
///
///      |_____w4_____|_____w5____|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildGraphWithMultiBatch() {
  auto builder = InnerGraphBuilder(graph_name_);
  auto w1 = builder.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = builder.AddNode("w2", VARIABLE, 0, 1);
  auto w3 = builder.AddNode("w3", VARIABLE, 0, 1);
  auto w4 = builder.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = builder.AddNode("w5", VARIABLE, 0, 1);

  auto const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  auto const2 = builder.AddNode("const2", CONSTANTOP, 0, 1);
  auto switch1 = builder.AddNode("switch1", SWITCH, 2, 2);
  auto switch_false = builder.AddNode("switch_false", IDENTITY, 1, 1);
  auto switch_true = builder.AddNode("switch_true", IDENTITY, 1, 1);
  auto merge1 = builder.AddNode("merge1", MERGE, 2, 2);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(const1, 0, switch1, 0);
  builder.AddDataEdge(const2, 0, switch1, 1);
  builder.AddDataEdge(switch1, 0, switch_false, 0);
  builder.AddDataEdge(switch1, 1, switch_true, 0);
  builder.AddDataEdge(merge1, 0, net_output, 0);

  std::string batch_label_128 = "batch_128";
  std::string batch_label_256 = "batch_256";

  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;

  {
    auto prefetch1 = builder.AddNode("batch_label_128/prefetch1", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500}, {500}, batch_label_128);
    auto prefetch2 = builder.AddNode("batch_label_128/prefetch2", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500}, {500}, batch_label_128);
    auto prefetch3 = builder.AddNode("batch_label_128/prefetch3", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500}, {500}, batch_label_128);
    auto prefetch4 = builder.AddNode("batch_label_128/prefetch4", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024}, {1024}, batch_label_128);
    auto prefetch5 = builder.AddNode("batch_label_128/prefetch5", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024}, {1024}, batch_label_128);

    auto add1 = builder.AddNode("batch_label_128/add1", ADD, 2, 1);
    SetBatchLabel(add1, batch_label_128);
    auto add2 = builder.AddNode("batch_label_128/add2", ADD, 2, 1);
    SetBatchLabel(add2, batch_label_128);
    auto add3 = builder.AddNode("batch_label_128/add3", ADD, 2, 1);
    SetBatchLabel(add3, batch_label_128);
    auto add4 = builder.AddNode("batch_label_128/add4", ADD, 2, 1);
    SetBatchLabel(add4, batch_label_128);
    auto add5 = builder.AddNode("batch_label_128/add5", ADD, 2, 1);
    SetBatchLabel(add5, batch_label_128);
    auto const1 = builder.AddNode("batch_label_128/const1", CONSTANTOP, 0, 1);
    SetBatchLabel(const1, batch_label_128);

    builder.AddDataEdge(w1, 0, prefetch1, 0);
    builder.AddDataEdge(w2, 0, prefetch2, 0);
    builder.AddDataEdge(w3, 0, prefetch3, 0);
    builder.AddDataEdge(w4, 0, prefetch4, 0);
    builder.AddDataEdge(w5, 0, prefetch5, 0);

    builder.AddDataEdge(const1, 0, add1, 0);
    builder.AddDataEdge(prefetch1, 0, add1, 1);

    builder.AddDataEdge(add1, 0, add2, 0);
    builder.AddDataEdge(prefetch2, 0, add2, 1);

    builder.AddDataEdge(add2, 0, add3, 0);
    builder.AddDataEdge(prefetch3, 0, add3, 1);

    builder.AddDataEdge(add3, 0, add4, 0);
    builder.AddDataEdge(prefetch4, 0, add4, 1);

    builder.AddDataEdge(add4, 0, add5, 0);
    builder.AddDataEdge(prefetch5, 0, add5, 1);

    builder.AddDataEdge(add5, 0, merge1, 0);
    builder.AddControlEdge(switch_false, const1);
  }

  {
    auto prefetch1 = builder.AddNode("batch_label_256/prefetch1", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500}, {500}, batch_label_256);
    auto prefetch2 = builder.AddNode("batch_label_256/prefetch2", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500}, {500}, batch_label_256);
    auto prefetch3 = builder.AddNode("batch_label_256/prefetch3", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500}, {500}, batch_label_256);
    auto prefetch4 = builder.AddNode("batch_label_256/prefetch4", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024}, {1024}, batch_label_256);
    auto prefetch5 = builder.AddNode("batch_label_256/prefetch5", HCOMALLGATHER, 1, 1);
    SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024}, {1024}, batch_label_256);

    auto add1 = builder.AddNode("batch_label_256/add1", ADD, 2, 1);
    SetBatchLabel(add1, batch_label_256);
    auto add2 = builder.AddNode("batch_label_256/add2", ADD, 2, 1);
    SetBatchLabel(add2, batch_label_256);
    auto add3 = builder.AddNode("batch_label_256/add3", ADD, 2, 1);
    SetBatchLabel(add3, batch_label_256);
    auto add4 = builder.AddNode("batch_label_256/add4", ADD, 2, 1);
    SetBatchLabel(add4, batch_label_256);
    auto add5 = builder.AddNode("batch_label_256/add5", ADD, 2, 1);
    SetBatchLabel(add5, batch_label_256);
    auto const1 = builder.AddNode("batch_label_256/const1", CONSTANTOP, 0, 1);
    SetBatchLabel(const1, batch_label_128);

    builder.AddDataEdge(w1, 0, prefetch1, 0);
    builder.AddDataEdge(w2, 0, prefetch2, 0);
    builder.AddDataEdge(w3, 0, prefetch3, 0);
    builder.AddDataEdge(w4, 0, prefetch4, 0);
    builder.AddDataEdge(w5, 0, prefetch5, 0);

    builder.AddDataEdge(const1, 0, add1, 0);
    builder.AddDataEdge(prefetch1, 0, add1, 1);

    builder.AddDataEdge(add1, 0, add2, 0);
    builder.AddDataEdge(prefetch2, 0, add2, 1);

    builder.AddDataEdge(add2, 0, add3, 0);
    builder.AddDataEdge(prefetch3, 0, add3, 1);

    builder.AddDataEdge(add3, 0, add4, 0);
    builder.AddDataEdge(prefetch4, 0, add4, 1);

    builder.AddDataEdge(add4, 0, add5, 0);
    builder.AddDataEdge(prefetch5, 0, add5, 1);

    builder.AddDataEdge(add5, 0, merge1, 1);

    builder.AddControlEdge(switch_true, const1);
  }

  auto compute_graph = builder.GetGraph();

  return compute_graph;
}

///
/// GraphWithMultiOutputPrefetch: Prefetch has more than one output
///
///                      w1         w2         w3         w4         w5
///                       \          \          \         \          \.
///                   prefetch1  prefetch2  prefetch3  prefetch4  prefetch5
///                     /   \      /   \      /   \      /   \      /
///                    /     \    /     \    /     \    /     \    /
///    const1 ----- add1      add2       add3       add4       add5
///                  |           \        |         /           |
///                  |            \       |        /            |
///                  |             \      |       /             |
///                  |              \     |      /              |
///                   --------------  net_output  ---------------
///
///  Memory distribution:
///
///      |___w1__|__w2__|__w3__|__|
///
///      |_____w4_____|_____w5____|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildGraphWithMultiOutputPrefetch() {
  auto builder = InnerGraphBuilder(graph_name_);
  auto w1 = builder.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = builder.AddNode("w2", VARIABLE, 0, 1);
  auto w3 = builder.AddNode("w3", VARIABLE, 0, 1);
  auto w4 = builder.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = builder.AddNode("w5", VARIABLE, 0, 1);

  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;

  auto prefetch1 = builder.AddNode("prefetch1", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch2 = builder.AddNode("prefetch2", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch3 = builder.AddNode("prefetch3", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500});
  auto prefetch4 = builder.AddNode("prefetch4", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024});
  auto prefetch5 = builder.AddNode("prefetch5", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024});

  auto const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto add2 = builder.AddNode("add2", ADD, 2, 1);
  auto add3 = builder.AddNode("add3", ADD, 2, 1);
  auto add4 = builder.AddNode("add4", ADD, 2, 1);
  auto add5 = builder.AddNode("add5", ADD, 2, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 5, 0);

  builder.AddDataEdge(w1, 0, prefetch1, 0);
  builder.AddDataEdge(w2, 0, prefetch2, 0);
  builder.AddDataEdge(w3, 0, prefetch3, 0);
  builder.AddDataEdge(w4, 0, prefetch4, 0);
  builder.AddDataEdge(w5, 0, prefetch5, 0);

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(prefetch1, 0, add1, 1);

  builder.AddDataEdge(prefetch1, 0, add2, 0);
  builder.AddDataEdge(prefetch2, 0, add2, 1);

  builder.AddDataEdge(prefetch2, 0, add3, 0);
  builder.AddDataEdge(prefetch3, 0, add3, 1);

  builder.AddDataEdge(prefetch3, 0, add4, 0);
  builder.AddDataEdge(prefetch4, 0, add4, 1);

  builder.AddDataEdge(prefetch4, 0, add5, 0);
  builder.AddDataEdge(prefetch5, 0, add5, 1);

  builder.AddDataEdge(add1, 0, net_output, 0);
  builder.AddDataEdge(add2, 0, net_output, 1);
  builder.AddDataEdge(add3, 0, net_output, 2);
  builder.AddDataEdge(add4, 0, net_output, 3);
  builder.AddDataEdge(add5, 0, net_output, 4);

  auto compute_graph = builder.GetGraph();

  return compute_graph;
}

///
/// GraphWithMultiOutputPrefetch: Prefetch has more than one output
///
///                     w1      w2        w3         w4         w5
///                      \    /   \      /  \      /   \      /    \.
  ///                   prefetch1  prefetch2  prefetch3  prefetch4  prefetch5
///                     /   \      /   \      /   \      /   \      /
///                    /     \    /     \    /     \    /     \    /
///    const1 ----- add1      add2       add3       add4       add5
///                  |           \        |         /           |
///                  |            \       |        /            |
///                  |             \      |       /             |
///                  |              \     |      /              |
///                   --------------  net_output  ---------------
///
///  Memory distribution:
///
///      |___w1__|__w2__|__w3__|__|
///
///      |_____w4_____|_____w5____|
///
ComputeGraphPtr BufferPoolGraphBuilder::BuildGraphWithMultiInputOutputPrefetch() {
  auto builder = InnerGraphBuilder(graph_name_);
  auto w1 = builder.AddNode("w1", VARIABLE, 0, 1);
  auto w2 = builder.AddNode("w2", VARIABLE, 0, 1);
  auto w3 = builder.AddNode("w3", VARIABLE, 0, 1);
  auto w4 = builder.AddNode("w4", VARIABLE, 0, 1);
  auto w5 = builder.AddNode("w5", VARIABLE, 0, 1);

  const int64_t buffer_pool_id = 0;
  const int64_t buffer_pool_size = 5600;

  auto prefetch1 = builder.AddNode("prefetch1", HCOMALLGATHER, 2, 2);
  SetPrefetchNodeInfo(prefetch1, buffer_pool_id, buffer_pool_size, {500, 500});
  auto prefetch2 = builder.AddNode("prefetch2", HCOMALLGATHER, 2, 2);
  SetPrefetchNodeInfo(prefetch2, buffer_pool_id, buffer_pool_size, {500, 500});
  auto prefetch3 = builder.AddNode("prefetch3", HCOMALLGATHER, 2, 2);
  SetPrefetchNodeInfo(prefetch3, buffer_pool_id, buffer_pool_size, {500, 1024});
  auto prefetch4 = builder.AddNode("prefetch4", HCOMALLGATHER, 2, 2);
  SetPrefetchNodeInfo(prefetch4, buffer_pool_id, buffer_pool_size, {1024, 1024});
  auto prefetch5 = builder.AddNode("prefetch5", HCOMALLGATHER, 1, 1);
  SetPrefetchNodeInfo(prefetch5, buffer_pool_id, buffer_pool_size, {1024});

  auto const1 = builder.AddNode("const1", CONSTANTOP, 0, 1);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto add2 = builder.AddNode("add2", ADD, 2, 1);
  auto add3 = builder.AddNode("add3", ADD, 2, 1);
  auto add4 = builder.AddNode("add4", ADD, 2, 1);
  auto add5 = builder.AddNode("add5", ADD, 2, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 5, 0);

  builder.AddDataEdge(w1, 0, prefetch1, 0);
  builder.AddDataEdge(w2, 0, prefetch1, 1);
  builder.AddDataEdge(w2, 0, prefetch2, 0);
  builder.AddDataEdge(w3, 0, prefetch2, 1);
  builder.AddDataEdge(w3, 0, prefetch3, 0);
  builder.AddDataEdge(w4, 0, prefetch3, 1);
  builder.AddDataEdge(w4, 0, prefetch4, 0);
  builder.AddDataEdge(w5, 0, prefetch4, 1);
  builder.AddDataEdge(w5, 0, prefetch5, 0);

  builder.AddDataEdge(const1, 0, add1, 0);
  builder.AddDataEdge(prefetch1, 0, add1, 1);

  builder.AddDataEdge(prefetch1, 1, add2, 0);
  builder.AddDataEdge(prefetch2, 0, add2, 1);

  builder.AddDataEdge(prefetch2, 1, add3, 0);
  builder.AddDataEdge(prefetch3, 0, add3, 1);

  builder.AddDataEdge(prefetch3, 1, add4, 0);
  builder.AddDataEdge(prefetch4, 0, add4, 1);

  builder.AddDataEdge(prefetch4, 1, add5, 0);
  builder.AddDataEdge(prefetch5, 0, add5, 1);

  builder.AddDataEdge(add1, 0, net_output, 0);
  builder.AddDataEdge(add2, 0, net_output, 1);
  builder.AddDataEdge(add3, 0, net_output, 2);
  builder.AddDataEdge(add4, 0, net_output, 3);
  builder.AddDataEdge(add5, 0, net_output, 4);

  auto compute_graph = builder.GetGraph();

  return compute_graph;
}
}  // namespace ut
}  // namespace ge
