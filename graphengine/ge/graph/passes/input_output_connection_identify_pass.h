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

#ifndef GE_GRAPH_PASSES_INPUT_OUTPUT_CONNECTION_IDENTIFY_PASS_H_
#define GE_GRAPH_PASSES_INPUT_OUTPUT_CONNECTION_IDENTIFY_PASS_H_

#include <map>
#include <vector>
#include "external/graph/graph.h"
#include "inc/graph_pass.h"

namespace ge {
class InputOutputConnectionIdentifyPass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;

 private:
  ///
  /// Find all nodes that connect to input node.
  /// @param [in] input node
  /// @param [out] map of nodes and anchor index that connect to input
  /// @param [out] map of nodes and anchor index that connect to output
  /// @return Status
  ///
  Status ProcessInputNode(const NodePtr &node, std::map<NodePtr, std::vector<uint32_t>> &connect_input_node_idx,
                          std::map<NodePtr, std::vector<uint32_t>> &connect_output_node_idx);

  ///
  /// Find all nodes that connect to output node.
  /// @param [in] output node
  /// @param [out] map of nodes and anchor index that connect to input
  /// @param [out] map of nodes and anchor index that connect to output
  /// @return Status
  ///
  Status ProcessOutputNode(const NodePtr &node, std::map<NodePtr, std::vector<uint32_t>> &connect_input_node_idx,
                           std::map<NodePtr, std::vector<uint32_t>> &connect_output_node_idx);

  ///
  /// Update all nodes that have shared memory.
  /// @param [in] symbol string
  /// @param [out] map of nodes and anchor index that connect to input
  /// @param [out] map of nodes and anchor index that connect to output
  /// @return Status
  ///
  Status UpdateNodeIdxMap(const string &symbol_string, std::map<NodePtr, std::vector<uint32_t>> &connect_input_node_idx,
                          std::map<NodePtr, std::vector<uint32_t>> &connect_output_node_idx);

  ///
  /// Set attr for all nodes that connect to input and output.
  /// @param [in] map of nodes and anchor index that connect to input
  /// @param [in] map of nodes and anchor index that connect to output
  /// @return Status
  ///
  Status SetNodeAttrOfConnectingInputOutput(const std::map<NodePtr, std::vector<uint32_t>> &connect_input_node_idx,
                                            const std::map<NodePtr, std::vector<uint32_t>> &connect_output_node_idx);

  // Members for ref mapping
  std::map<std::string, std::list<NodeIndexIO>> symbol_to_anchors_;
  std::map<std::string, std::string> anchor_to_symbol_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_INPUT_OUTPUT_CONNECTION_IDENTIFY_PASS_H_