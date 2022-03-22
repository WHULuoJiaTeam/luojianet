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

#include "host_kernels/squeeze_kernel.h"

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "inc/kernel_factory.h"


namespace {
constexpr uint32_t kInputDescIndex = 0;
constexpr uint32_t kOutputDescIndex = 0;
constexpr size_t kSqueezeInputSize = 1;
constexpr size_t kSqueezeOutputSize = 1;
}

namespace ge {
Status SqueezeKernel::Compute(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "parameter is nullptr");
    return PARAM_INVALID;
  }
  if (!KernelUtils::CheckFormatSupported(node_ptr)) {
    GELOGW("CheckFormatSupported failed");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status SqueezeKernel::Compute(const ge::OpDescPtr op_desc, const std::vector<ge::ConstGeTensorPtr> &input,
                              std::vector<ge::GeTensorPtr> &v_output) {
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "SqueezeKernel op_desc is null.");
    return PARAM_INVALID;
  }
  GELOGD("SqueezeKernel in: node[%s]", op_desc->GetName().c_str());

  bool size_check = ((op_desc->GetInputsSize() != kSqueezeInputSize) ||
                     (op_desc->GetOutputsSize() != kSqueezeOutputSize) || (input.size() != kSqueezeInputSize));
  if (size_check) {
    GELOGW("Size check fail, node[%s] inputs size:%zu, outputs size:%zu", op_desc->GetName().c_str(),
           op_desc->GetInputsSize(), op_desc->GetOutputsSize());
    return NOT_CHANGED;
  }

  auto tensor_desc = op_desc->GetOutputDesc(kOutputDescIndex);
  GeTensorPtr output_ptr = MakeShared<ge::GeTensor>(tensor_desc);
  if (output_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "node [%s] make shared failed.", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  auto ge_tensor = input.at(kInputDescIndex);
  if (ge_tensor == nullptr) {
    GELOGE(PARAM_INVALID, "node [%s] get input failed.", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  if (output_ptr->SetData(ge_tensor->GetData()) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
  }
  v_output.emplace_back(output_ptr);
  GELOGI("SqueezeKernel success: node[%s]", op_desc->GetName().c_str());

  return SUCCESS;
}
REGISTER_KERNEL(SQUEEZE, SqueezeKernel);
}  // namespace ge
