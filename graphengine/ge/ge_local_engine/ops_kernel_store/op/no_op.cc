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

#include "ge_local_engine/ops_kernel_store/op/no_op.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge_local_engine/ops_kernel_store/op/op_factory.h"

namespace ge {
namespace ge_local {
NoOp::NoOp(const Node &node, RunContext &run_context) : Op(node, run_context) {}

Status NoOp::Run() {
  GELOGD("Node:%s type is %s, no need generate task.", name_.c_str(), type_.c_str());
  // Do nothing
  return SUCCESS;
}

REGISTER_OP_CREATOR(Data, NoOp);

REGISTER_OP_CREATOR(AippData, NoOp);

REGISTER_OP_CREATOR(NoOp, NoOp);

REGISTER_OP_CREATOR(Variable, NoOp);

REGISTER_OP_CREATOR(Constant, NoOp);

REGISTER_OP_CREATOR(Const, NoOp);

REGISTER_OP_CREATOR(ControlTrigger, NoOp);

REGISTER_OP_CREATOR(Merge, NoOp);

// Functional Op.
REGISTER_OP_CREATOR(If, NoOp);
REGISTER_OP_CREATOR(_If, NoOp);
REGISTER_OP_CREATOR(StatelessIf, NoOp);
REGISTER_OP_CREATOR(Case, NoOp);
REGISTER_OP_CREATOR(While, NoOp);
REGISTER_OP_CREATOR(_While, NoOp);
REGISTER_OP_CREATOR(StatelessWhile, NoOp);
REGISTER_OP_CREATOR(For, NoOp);
REGISTER_OP_CREATOR(PartitionedCall, NoOp);
REGISTER_OP_CREATOR(StatefulPartitionedCall, NoOp);
}  // namespace ge_local
}  // namespace ge
