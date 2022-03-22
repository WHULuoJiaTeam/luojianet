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
#ifndef INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_CONNECTION_MATRIX_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_CONNECTION_MATRIX_H_

#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "common/large_bm.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class ConnectionMatrix {
public:
  ConnectionMatrix();

  ~ConnectionMatrix();

  bool IsConnected(const ge::NodePtr &a, const ge::NodePtr &b) const;

  // inputs are all input nodes of parameter node.
  // if there is a path between A->B, then B will own A's
  // connectivity. The reason is ---
  // If some node can reach A, than it can also reach B.
  void SetConnectivity(const ge::Node::Vistor<ge::NodePtr> &inputs, const ge::NodePtr &node);

  /* Computes the connectivity between two nodes in the
   * computation. The returned ConnectivityMatrix is constructed such that
   * ConnectivityMatrix::IsConnected(a, b) returns true iff there exists a
   * directed path (from producer to consumer) from 'a' to 'b'. Both data
   * connection and control connection are considered for connectivity.
   * A node is connected to itself. */
  Status Generate(const ge::ComputeGraph &graph);

  // update reachablity map for fused nodes.
  void Update(ge::ComputeGraph &graph, std::vector<ge::NodePtr> &fusion_nodes);

private:
  int64_t GetIndex(const ge::NodePtr &node) const;

  const ge::LargeBitmap &GetBitMap(const ge::NodePtr &node) const;

  ge::LargeBitmap &GetBitMap(const ge::NodePtr &node);

  size_t size_;

  std::vector<ge::LargeBitmap> bit_maps;
};
}
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_CONNECTION_MATRIX_H_
