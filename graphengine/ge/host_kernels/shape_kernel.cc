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

#include "host_kernels/shape_kernel.h"

#include <memory>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "host_kernels/kernel_utils.h"
#include "graph/passes/pass_utils.h"
#include "inc/kernel_factory.h"
#include "framework/common/types.h"

namespace ge {
namespace {
const size_t kShapeInputSize = 1;
const size_t kShapeOutputSize = 1;
}  // namespace
Status ShapeKernel::Compute(const NodePtr &node, std::vector<GeTensorPtr> &v_output) {
  GELOGD("ShapeKernel in");
  if (node == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  bool size_check = ((op_desc->GetInputsSize() != kShapeInputSize) || (op_desc->GetOutputsSize() != kShapeOutputSize));
  if (size_check) {
    GELOGW("Size check fail, inputs size:%zu, outputs size:%zu", op_desc->GetInputsSize(), op_desc->GetOutputsSize());
    return NOT_CHANGED;
  }
  const auto &input_desc = op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  if (KernelUtils::IsUnknownShape(input_desc->GetShape())) {
    GELOGW("Input shape is unknown, ignore shape kernel.");
    return NOT_CHANGED;
  }
  vector<int64_t> dims = input_desc->GetShape().GetDims();
  Status ret = PassUtils::ConstructTensorDescWithData(op_desc->GetOutputDesc(0), dims, v_output);
  if (ret != SUCCESS) {
    GELOGE(ret, "Shape kernel construct tensor desc failed!");
    return ret;
  }
  GELOGD("Shape kernel success");
  return SUCCESS;
}

REGISTER_KERNEL(SHAPE, ShapeKernel);
}  // namespace ge
