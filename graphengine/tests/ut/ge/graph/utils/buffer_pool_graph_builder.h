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

#ifndef GRAPH_UTILS_BUFFER_POOL_GRAPH_BUILDER_H_
#define GRAPH_UTILS_BUFFER_POOL_GRAPH_BUILDER_H_

#include <string>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"

namespace ge {
namespace ut {
class BufferPoolGraphBuilder {
 public:
  explicit BufferPoolGraphBuilder(const std::string &name = "BufferPoolGraph");
  ~BufferPoolGraphBuilder() {}
  class InnerGraphBuilder {
   public:
    explicit InnerGraphBuilder(const std::string &name);
    ~InnerGraphBuilder() {}
    NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                    Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                    std::vector<int64_t> shape = {1, 1, 224, 224});

    void AddDataEdge(NodePtr &src_node, int src_idx, NodePtr &dst_node, int dst_idx);

    void AddControlEdge(NodePtr &src_node, NodePtr &dst_node);

    ComputeGraphPtr GetGraph() {
      graph_->TopologicalSorting();
      return graph_;
    }
   private:
    ComputeGraphPtr graph_;
  };

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
  ComputeGraphPtr BuildNormalGraph();

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
  ComputeGraphPtr BuildNormalGraphWithMultiBufferPool();

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
  ComputeGraphPtr BuildSerialGraph();

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
  ComputeGraphPtr BuildGraphWithMultiPrefetch();

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
  ComputeGraphPtr BuildGraphWithSubgraph();

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
  ComputeGraphPtr BuildSubgraphWithInnerDependency();

  ///
  /// BuildGraphWithMultiBatch: Different batch label
  ///
  ///
  ///                                                      batch_label_128
  ///
  ///                            const1 ----- add1 ----- add2 ----- add3 ----- add4 ----- add5 ---
  ///                           /              /          /          /         /          /       \.
  ///  				                 /c        prefetch1  prefetch2  prefetch3  prefetch4  prefetch5     \.
  ///   const1        switch_false           /          /          /         /          /           \.
  ///       \         /                     /          /          /         /          /             \.
  ///         switch1                      w1         w2         w3        w4         w5           merge1 -- net_output
  ///  	   /          \                     \          \          \         \          \             /
  ///   const2        switch_true            \          \          \         \          \           /
  ///                          \c        prefetch1  prefetch2  prefetch3  prefetch4  prefetch5     /
  ///                           \              \          \          \         \          \       /
  ///                            const1 ----- add1 ----- add2 ----- add3 ----- add4 ----- add5 ---
  ///
  ///                                                     batch_label_256
  ///
  ///
  ///  Memory distribution:
  ///
  ///      |___w1__|__w2__|__w3__|__|
  ///
  ///      |_____w4_____|_____w5____|
  ///
  ComputeGraphPtr BuildGraphWithMultiBatch();

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
  ComputeGraphPtr BuildGraphWithMultiOutputPrefetch();

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
  ComputeGraphPtr BuildGraphWithMultiInputOutputPrefetch();

  void SetBufferPool(NodePtr &node, int64_t pool_id, int64_t pool_size, const std::string &batch_label = "");

  void SetBatchLabel(NodePtr &node, const std::string &batch_label = "");

  void SetOutputMemSize(NodePtr &node, const std::vector<int64_t> &mem_size = {1024});

  void SetWorkSpaceMemSize(NodePtr &node, const std::vector<int64_t> &ws_bytes = {1024});

  void SetPrefetchNodeInfo(NodePtr &node, int64_t pool_id, int64_t pool_size,
                           const std::vector<int64_t> &mem_size = {1024},
                           const std::vector<int64_t> &ws_bytes = {1024},
                           const std::string &batch_label = "");

 private:
  std::string graph_name_;
};
}  // namespace ut
}  // namespace ge

#endif  // GRAPH_UTILS_BUFFER_POOL_GRAPH_BUILDER_H_
