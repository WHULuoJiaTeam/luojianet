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

#ifndef COMMON_GRAPH_UTILS_CONSTANT_UTILS_H_
#define COMMON_GRAPH_UTILS_CONSTANT_UTILS_H_
#include "graph/node.h"
#include "graph/operator.h"
#include "graph/op_desc.h"

namespace ge {
class ConstantUtils {
 public:
  // check is constant
  static bool IsConstant(const NodePtr &node);
  static bool IsConstant(const Operator &op);
  static bool IsConstant(const OpDescPtr &op_desc);
  static bool IsPotentialConst(const OpDescPtr &op_desc);
  static bool IsRealConst(const OpDescPtr &op_desc);
  // get/set  weight
  static bool GetWeight(const OpDescPtr &op_desc, const uint32_t index, ConstGeTensorPtr &weight);
  static bool GetWeight(const Operator &op, const uint32_t index, Tensor &weight);
  static bool MutableWeight(const OpDescPtr &op_desc, const uint32_t index, GeTensorPtr &weight);
  static bool SetWeight(const OpDescPtr &op_desc, const uint32_t index, const GeTensorPtr weight);
  static bool MarkPotentialConst(const OpDescPtr &op_desc, const std::vector<int> indices, const std::vector<GeTensorPtr> weights);
  static bool UnMarkPotentialConst(const OpDescPtr &op_desc);
 private:
  static bool GetPotentialWeight(const OpDescPtr &op_desc, std::vector<uint32_t> &weight_indices,
                                 std::vector<ConstGeTensorPtr> &weights);
  static bool MutablePotentialWeight(const OpDescPtr &op_desc, std::vector<uint32_t> &weight_indices,
                                     std::vector<GeTensorPtr> &weights);
};
}

#endif // COMMON_GRAPH_UTILS_CONSTANT_UTILS_H_
