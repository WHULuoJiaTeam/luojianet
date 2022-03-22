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

#ifndef GE_GRAPH_PASSES_NO_USE_RESHAPE_REMOVE_PASS_H_
#define GE_GRAPH_PASSES_NO_USE_RESHAPE_REMOVE_PASS_H_

#include "graph/passes/base_pass.h"

namespace ge {
class NoUseReshapeRemovePass : public BaseNodePass {
 public:
  ///
  /// Entry of the NoUseReshapeRemovePass optimizer
  /// To satisfy fusion rule of FE, remove reshape op which input & output format is same
  /// @param [in] node: Input Node
  /// @return SUCCESS: Dont find need to delete node
  /// @return NOT_CHANGED: find need to delete node
  /// @return OTHERS:  Execution failed
  /// @author
  ///
  Status Run(ge::NodePtr &node) override;

 private:
  Status TryRemoveConstShapeInput(NodePtr &reshape_node);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_NO_USE_RESHAPE_REMOVE_PASS_H_
