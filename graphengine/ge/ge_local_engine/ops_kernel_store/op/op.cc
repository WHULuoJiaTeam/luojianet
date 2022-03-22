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

#include "ge_local_engine/ops_kernel_store/op/op.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/op_desc.h"
#include "graph/utils/anchor_utils.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
namespace ge_local {
Op::Op(const Node &node, RunContext &run_context)
    : run_context_(run_context), node_(node), name_(node.GetName()), type_(node.GetType()) {}
}  // namespace ge_local
}  // namespace ge
