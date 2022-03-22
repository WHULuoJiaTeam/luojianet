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

#ifndef GE_HYBRID_MODEL_SUBGRAPH_ITEM_H_
#define GE_HYBRID_MODEL_SUBGRAPH_ITEM_H_

#include "external/ge/ge_api_error_codes.h"
#include "hybrid/model/node_item.h"

namespace ge {
namespace hybrid {
class GraphItem {
 public:
  GraphItem() = default;
  ~GraphItem();
  Status GroupNodes();
  const vector<NodeItem *> &GetAllNodes() const;
  const vector<NodeItem *> &GetAllNodes(int group) const;
  const vector<NodeItem *> &GetRootNodes(int group) const;
  const vector<const NodeItem *> &GetInputNodes() const;
  Status GetOutputDescList(std::vector<ConstGeTensorDescPtr> &output_desc_list) const;
  const vector<std::pair<const NodeItem *, int>> &GetOutputEdges() const;
  int TotalInputs() const {
    return total_inputs_;
  }

  int TotalOutputs() const {
    return total_outputs_;
  }

  size_t GetNodeSize(int group) const;

  bool HasCtrlFlowOp() const {
    return has_ctrl_flow_op_;
  }

  const std::string& GetName() const {
    return name_;
  }

  void SetName(const string &name) {
    name_ = name;
  }

  size_t NumGroups() const {
    return grouped_node_items_.size();
  }

  const NodeItem *GetOutputNode() const;

  bool IsDynamic() const;
  int GetParentOutputIndex(size_t index) const;
  const vector<int> &GetInputIndexMapping() const;

 private:
  friend class HybridModelBuilder;
  Status GroupNodes(const std::vector<NodeItem *> &node_items,
                    std::vector<std::vector<NodeItem *>> &grouped_node_items) const;

  std::string name_;
  std::vector<NodeItem *> node_items_;
  std::vector<std::vector<NodeItem *>> grouped_node_items_;
  std::vector<NodeItem *> root_items_;
  std::vector<std::vector<NodeItem *>> grouped_root_items_;
  std::vector<const NodeItem *> input_nodes_;
  const NodeItem *output_node_ = nullptr;
  // <src_node, out_index>
  std::vector<std::pair<const NodeItem *, int>> output_edges_;
  int total_inputs_ = 0;
  int total_outputs_ = 0;

  bool is_dynamic_ = true;
  bool has_ctrl_flow_op_ = false;
  std::vector<int> input_index_mapping_;
  std::vector<int> output_index_mapping_;
};
}  // namespace hybrid
}  // namespace ge
#endif // GE_HYBRID_MODEL_SUBGRAPH_ITEM_H_
