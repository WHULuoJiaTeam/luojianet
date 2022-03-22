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

#ifndef GE_GRAPH_PASSES_MEMCPY_ADDR_ASYNC_PASS_H_
#define GE_GRAPH_PASSES_MEMCPY_ADDR_ASYNC_PASS_H_

#include "inc/graph_pass.h"

namespace ge {

class MemcpyAddrAsyncPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph);

 private:
  Status AddMemcpyAddrAsyncNode(const ComputeGraphPtr &graph, const NodePtr &node);
  Status AddMemcpyAsyncNode(const NodePtr &node);
  void FindUserData(const NodePtr &node, uint32_t &parent_index);
  void FindUserDataForKnown(const NodePtr &parent_node, uint32_t &parent_index);
  void FindUserDataForNonDynamic(const ge::NodePtr &parent_node, uint32_t &parent_index);
  bool IsEmptyTenor(const GeShape &shape) const;

  NodePtr CreateMemcpyAddrAsyncNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor,
                                    const NodePtr &out_of_user_data);
  Status InsertMemcpyAddrAsyncNode(const OutDataAnchorPtr &out_anchor, const InDataAnchorPtr &in_anchor,
                                   const NodePtr &node);
  Status InsertMemAddrAsyncNodeBeforeNetoutput(const ComputeGraphPtr &graph, const NodePtr &node);

  NodePtr user_data_;
  NodePtr out_of_user_data_;
  OutDataAnchorPtr peer_out_anchor_;
  InDataAnchorPtr in_anchor_;
  bool find_user_data_ = false;
  NodePtr user_data_for_known_;
  NodePtr out_of_user_data_for_known_;
  OutDataAnchorPtr peer_out_anchor_for_known_;
  InDataAnchorPtr in_anchor_for_known_;
  bool find_user_data_for_known_ = false;
  bool known_sub_graph_ = false;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_MEMCPY_ADDR_ASYNC_PASS_H_
