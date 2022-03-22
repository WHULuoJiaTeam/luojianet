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
#ifndef GE_GRAPH_PASSES_FOR_PASS_H
#define GE_GRAPH_PASSES_FOR_PASS_H

#include "graph/passes/base_pass.h"

struct ForInfo {
  ForInfo() : for_node(nullptr), start(nullptr), limit(nullptr), delta(nullptr), for_body(nullptr) {}
  ge::NodePtr for_node;
  ge::OutDataAnchorPtr start;
  ge::OutDataAnchorPtr limit;
  ge::OutDataAnchorPtr delta;
  std::string body_name;
  ge::ComputeGraphPtr for_body;
  std::vector<ge::OutDataAnchorPtr> data_inputs;
  std::vector<std::vector<ge::InDataAnchorPtr>> data_outputs;
  std::vector<ge::OutControlAnchorPtr> ctrl_inputs;
  std::vector<ge::InControlAnchorPtr> ctrl_outputs;
};

struct WhileInfo {
  WhileInfo()
      : while_node(nullptr), sub_graph_node(nullptr), i(nullptr), abs_delta(nullptr), range(nullptr),
        start(nullptr), delta(nullptr), for_body(nullptr), while_cond(nullptr), while_body(nullptr) {}
  ge::NodePtr while_node;
  ge::NodePtr sub_graph_node;
  ge::OutDataAnchorPtr i;
  ge::OutDataAnchorPtr abs_delta;
  ge::OutDataAnchorPtr range;
  ge::OutDataAnchorPtr start;
  ge::OutDataAnchorPtr delta;
  std::string for_body_name;
  ge::ComputeGraphPtr for_body;
  ge::ComputeGraphPtr while_cond;
  ge::ComputeGraphPtr while_body;
  std::vector<ge::OutDataAnchorPtr> data_inputs;
  std::vector<std::vector<ge::InDataAnchorPtr>> data_outputs;
  std::vector<ge::OutControlAnchorPtr> ctrl_inputs;
  std::vector<ge::InControlAnchorPtr> ctrl_outputs;
};

namespace ge {
class ForPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;

 private:
  ///
  /// @brief Build for_info
  /// @param [in] root_graph
  /// @param [in] node
  /// @param [out] for_info
  /// @return Status
  ///
  static Status BuildForInfo(const ComputeGraphPtr &root_graph, const NodePtr &node, ForInfo &for_info);

  ///
  /// @brief Transfer while_info from for_info
  /// @param [in] graph
  /// @param [in] for_info
  /// @param [out] while_info
  /// @return Status
  ///
  Status TranWhileInfo(const ComputeGraphPtr &graph, const ForInfo &for_info, WhileInfo &while_info);

  ///
  /// @brief Build cond_graph for while_node
  /// @param [in&out] while_info
  /// @return ComputeGraphPtr
  ///
  static ComputeGraphPtr BuildCondGraph(WhileInfo &while_info);

  ///
  /// @brief Build body_graph for while_node
  /// @param [in&out] while_info
  /// @return ComputeGraphPtr
  ///
  static ComputeGraphPtr BuildBodyGraph(WhileInfo &while_info);

  ///
  /// @brief Update InputMapping for for-body-graph
  /// @param [in] while_info
  /// @return Status
  ///
  static Status UpdateForBodyInputMapping(const WhileInfo &while_info);

  ///
  /// @brief Find input with index for For node
  /// @param [in] node
  /// @param [in] index
  /// @return OutDataAnchorPtr
  ///
  static OutDataAnchorPtr FindInputWithIndex(const NodePtr &node, uint32_t index);

  ///
  /// @brief Find inputs / outputs for for node
  /// @param [in] node
  /// @param [out] data_inputs
  /// @param [out] data_outputs
  /// @param [out] ctrl_inputs
  /// @param [out] ctrl_outputs
  /// @return Status
  ///
  static Status FindInputsAndOutputs(const NodePtr &node, std::vector<OutDataAnchorPtr> &data_inputs,
                                     std::vector<std::vector<ge::InDataAnchorPtr>> &data_outputs,
                                     std::vector<ge::OutControlAnchorPtr> &ctrl_inputs,
                                     std::vector<ge::InControlAnchorPtr> &ctrl_outputs);

  ///
  /// @brief Create const op_desc
  /// @param [in] name
  /// @param [in] value
  /// @return OpDescPtr
  ///
  static OpDescPtr CreateConstDesc(const std::string &name, int32_t value);

  ///
  /// @brief Create loop input
  /// @param [in] graph
  /// @param [in] for_info
  /// @param [out] range_input
  /// @param [out] abs_delta_input
  /// @return Status
  ///
  Status CreateLoopInput(const ComputeGraphPtr &graph, const ForInfo &for_info, OutDataAnchorPtr &range_input,
                         OutDataAnchorPtr &abs_delta_input);

  ///
  /// @brief Create op_desc
  /// @param [in] name
  /// @param [in] type
  /// @param [in] io_equal_flag
  /// @return OpDescPtr
  ///
  static OpDescPtr CreateOpDesc(const std::string &name, const std::string &type, bool io_equal_flag);

  ///
  /// @brief Build while-info
  /// @param [in] for_info
  /// @param [in] i_input
  /// @param [in] range_input
  /// @param [in] abs_delta_input
  /// @param [out] while_info
  /// @return void
  ///
  static void BuildWhileInfo(const ForInfo &for_info, const OutDataAnchorPtr &i_input,
                             const OutDataAnchorPtr &range_input, const OutDataAnchorPtr &abs_delta_input,
                             WhileInfo &while_info);

  ///
  /// @brief Insert while_node
  /// @param [in] graph
  /// @param [in] name
  /// @param [in] while_info
  /// @return Status
  ///
  Status InsertWhileNode(const ComputeGraphPtr &graph, const std::string &name, WhileInfo &while_info);

  ///
  /// @brief Build while link-edge
  /// @param [in] while_info
  /// @return Status
  ///
  static Status BuildWhileLink(const WhileInfo &while_info);

  ///
  /// @brief Create op_desc for subgraph node
  /// @param [in] name
  /// @param [in] input_num
  /// @param [in] output_num
  /// @return OpDescPtr
  ///
  static OpDescPtr CreateSubgraphOpDesc(const std::string &name, uint32_t input_num, uint32_t output_num);
};
}  // namespace ge
#endif //GE_GRAPH_PASSES_FOR_PASS_H
