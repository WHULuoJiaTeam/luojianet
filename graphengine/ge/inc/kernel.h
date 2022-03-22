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

#ifndef GE_INC_KERNEL_H_
#define GE_INC_KERNEL_H_

#include <vector>

#include "framework/common/op/ge_op_utils.h"
#include "graph/compute_graph.h"
#include "external/graph/graph.h"
#include "graph/op_desc.h"

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

namespace ge {
///
/// @ingroup domi_omg
/// @brief Kernel interface
/// @author
///
class Kernel {
 public:
  ///
  /// Constant calculation interface, the result is appended to output
  /// @param [in] op_desc_ptr Operator related parameters
  /// @param [in] input Constant to be calculated
  /// @param [inout] output Save calculation results
  /// @author
  ///
  virtual Status Compute(const ge::OpDescPtr op_desc_ptr, const std::vector<ge::ConstGeTensorPtr>& input,
                         std::vector<ge::GeTensorPtr>& v_output) {
    (void)op_desc_ptr;
    (void)input;
    (void)v_output;
    return NOT_CHANGED;
  }

  virtual Status Compute(const NodePtr& node, std::vector<GeTensorPtr>& v_output) {
    (void)node;
    (void)v_output;
    return NOT_CHANGED;
  }

  virtual Status Compute(const NodePtr& node_ptr) {
    (void)node_ptr;
    return NOT_CHANGED;
  }

  ///
  /// Destructor
  ///
  virtual ~Kernel() {}
};
}  // namespace ge
#endif  // GE_INC_KERNEL_H_
