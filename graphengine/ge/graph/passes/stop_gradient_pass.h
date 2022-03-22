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

#ifndef GE_GRAPH_PASSES_STOP_GRADIENT_PASS_H_
#define GE_GRAPH_PASSES_STOP_GRADIENT_PASS_H_

#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/omg_util.h"
#include "graph/passes/base_pass.h"

namespace ge {
class StopGradientPass : public BaseNodePass {
 public:
  Status Run(NodePtr &node) override;
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_STOP_GRADIENT_PASS_H_
