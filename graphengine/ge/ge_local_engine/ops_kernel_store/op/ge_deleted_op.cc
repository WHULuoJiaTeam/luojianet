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

#include "ge_local_engine/ops_kernel_store/op/ge_deleted_op.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge_local_engine/ops_kernel_store/op/op_factory.h"

namespace ge {
namespace ge_local {
GeDeletedOp::GeDeletedOp(const Node &node, RunContext &run_context) : Op(node, run_context) {}

Status GeDeletedOp::Run() {
  REPORT_INNER_ERROR("E19999", "Node:%s type is %s, should be deleted by ge.", name_.c_str(), type_.c_str());
  GELOGE(FAILED, "[Delelte][Node] Node:%s type is %s, should be deleted by ge.", name_.c_str(), type_.c_str());
  // Do nothing
  return FAILED;
}

REGISTER_OP_CREATOR(TemporaryVariable, GeDeletedOp);
REGISTER_OP_CREATOR(DestroyTemporaryVariable, GeDeletedOp);
REGISTER_OP_CREATOR(GuaranteeConst, GeDeletedOp);
REGISTER_OP_CREATOR(PreventGradient, GeDeletedOp);
REGISTER_OP_CREATOR(StopGradient, GeDeletedOp);
REGISTER_OP_CREATOR(ExpandDims, GeDeletedOp);
REGISTER_OP_CREATOR(Reshape, GeDeletedOp);
REGISTER_OP_CREATOR(ReFormat, GeDeletedOp);
REGISTER_OP_CREATOR(Squeeze, GeDeletedOp);
REGISTER_OP_CREATOR(Unsqueeze, GeDeletedOp);
REGISTER_OP_CREATOR(Size, GeDeletedOp);
REGISTER_OP_CREATOR(Shape, GeDeletedOp);
REGISTER_OP_CREATOR(ShapeN, GeDeletedOp);
REGISTER_OP_CREATOR(Rank, GeDeletedOp);
REGISTER_OP_CREATOR(_Retval, GeDeletedOp);
REGISTER_OP_CREATOR(ReadVariableOp, GeDeletedOp);
REGISTER_OP_CREATOR(VarHandleOp, GeDeletedOp);
REGISTER_OP_CREATOR(VarIsInitializedOp, GeDeletedOp);
REGISTER_OP_CREATOR(Snapshot, GeDeletedOp);
REGISTER_OP_CREATOR(Identity, GeDeletedOp);
REGISTER_OP_CREATOR(IdentityN, GeDeletedOp);
REGISTER_OP_CREATOR(VariableV2, GeDeletedOp);
REGISTER_OP_CREATOR(Empty, GeDeletedOp);
REGISTER_OP_CREATOR(PlaceholderWithDefault, GeDeletedOp);
REGISTER_OP_CREATOR(IsVariableInitialized, GeDeletedOp);
REGISTER_OP_CREATOR(PlaceholderV2, GeDeletedOp);
REGISTER_OP_CREATOR(Placeholder, GeDeletedOp);
REGISTER_OP_CREATOR(End, GeDeletedOp);
REGISTER_OP_CREATOR(Switch, GeDeletedOp);
REGISTER_OP_CREATOR(SwitchN, GeDeletedOp);
REGISTER_OP_CREATOR(RefMerge, GeDeletedOp);
REGISTER_OP_CREATOR(RefSwitch, GeDeletedOp);
REGISTER_OP_CREATOR(TransShape, GeDeletedOp);
REGISTER_OP_CREATOR(Bitcast, GeDeletedOp);
}  // namespace ge_local
}  // namespace ge
