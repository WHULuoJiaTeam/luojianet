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

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_H_

#include <climits>
#include <string>
#include <vector>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/node.h"

namespace ge {
struct RunContext;
namespace ge_local {
/**
 * The base class for all op.
 */
class GE_FUNC_VISIBILITY Op {
 public:
  Op(const Node &node, RunContext &run_context);

  virtual ~Op() = default;

  virtual Status Run() = 0;

 protected:
  const RunContext &run_context_;
  const Node &node_;
  std::string name_;
  std::string type_;
};
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_STORE_OP_OP_H_
