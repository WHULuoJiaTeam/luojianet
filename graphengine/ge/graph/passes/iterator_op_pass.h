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

#ifndef GE_GRAPH_PASSES_ITERATOR_OP_PASS_H_
#define GE_GRAPH_PASSES_ITERATOR_OP_PASS_H_

#include "external/graph/graph.h"
#include "inc/graph_pass.h"

namespace ge {
class IteratorOpPass : public GraphPass {
 public:
  IteratorOpPass() {}

  virtual ~IteratorOpPass() {}

  Status Run(ge::ComputeGraphPtr graph);

  Status GetVariableValue(uint64_t session_id, const ge::GeTensorDesc &opdesc, const std::string &var_name, void *dest);

 private:
  ///
  /// @brief Insert EndOfSequence node
  ///
  /// @param preNode
  /// @param graph
  /// @return ge::NodePtr
  ///
  ge::NodePtr InsertEndOfSequenceNode(const ge::NodePtr &pre_node, const ge::NodePtr &memcpy_node,
                                      const ge::ComputeGraphPtr &graph);
  ///
  /// @brief Create a EndOfSequence Op object
  ///
  /// @param preNode
  /// @return ge::OpDescPtr
  ///
  ge::OpDescPtr CreateEndOfSequenceOp(const ge::NodePtr &pre_node);
  ///
  /// @brief Insert memcpy node
  ///
  /// @param preNode
  /// @param graph
  /// @return ge::NodePtr
  ///
  ge::NodePtr InsertMemcpyAsyncNode(const ge::NodePtr &pre_node, const ge::ComputeGraphPtr &graph);
  ///
  /// @brief Create a Memcpy Async Op object
  ///
  /// @param preNode
  /// @return ge::OpDescPtr
  ///
  ge::OpDescPtr CreateMemcpyAsyncOp(const ge::NodePtr &pre_node);

  Status SetRtContext(uint64_t session_id, uint32_t graph_id, rtContext_t rt_context, rtCtxMode_t mode);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_ITERATOR_OP_PASS_H_
