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

#ifndef GE_GRAPH_PASSES_HCCL_CONTINUOUS_MEMCPY_PASS_H_
#define GE_GRAPH_PASSES_HCCL_CONTINUOUS_MEMCPY_PASS_H_

#include <string>
#include <unordered_map>

#include "external/graph/graph.h"
#include "inc/graph_pass.h"

namespace ge {
class HcclContinuousMemcpyPass : public GraphPass {
 public:
  Status Run(ge::ComputeGraphPtr graph);
  Status ClearStatus() override;

 private:
  NodePtr CreateIdentityNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor);

  NodePtr CreateAssignNode(const ComputeGraphPtr &graph, const OutDataAnchorPtr &out_data_anchor);

  std::string CheckDuplicateName(const std::string &node_name);

  Status ModifyEdgeConnection(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_out_anchor,
          const InDataAnchorPtr &hccl_in_anchor);

  Status InsertIdentityBeforeHccl(const ComputeGraphPtr &graph, const OutDataAnchorPtr &src_out_anchor,
                                  const InDataAnchorPtr &hccl_in_anchor);

  Status InsertAssignAfterBroadcastIfNeed(const ComputeGraphPtr &graph,
                                          const OutDataAnchorPtr &src_out_anchor,
                                          const InDataAnchorPtr &hccl_in_anchor);

  Status ContinuousInputProcess(const ComputeGraphPtr &graph, const NodePtr node);

  Status P2pmemInputProcess(const ComputeGraphPtr &graph, const NodePtr node);

  bool IsDataNode(const std::string& node_type);

  std::map<std::string, uint32_t> node_num_map_;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_HCCL_MEMCPY_PASS_H_
