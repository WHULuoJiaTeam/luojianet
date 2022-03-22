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

#include "host_kernels/identity_kernel.h"
#include "inc/kernel_factory.h"
#include "framework/common/types.h"

namespace {
constexpr uint32_t kInputDescIndex = 0;
constexpr uint32_t kOutputDescIndex = 0;
}  // namespace

namespace ge {
Status IdentityKernel::Compute(const ge::OpDescPtr op_desc, const std::vector<ge::ConstGeTensorPtr> &input,
                               std::vector<ge::GeTensorPtr> &v_output) {
  if (op_desc == nullptr) {
    GELOGE(PARAM_INVALID, "IdentityKernel op_desc is null.");
    return NOT_CHANGED;
  }
  if (input.empty()) {
    GELOGE(PARAM_INVALID, "Node [%s] inputs is empty.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  if (op_desc->GetOutputsSize() < 1) {
    GELOGE(PARAM_INVALID, "Node [%s] output is empty.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  GELOGD("IdentityKernel in: node[%s]", op_desc->GetName().c_str());

  auto out_tensor_desc = op_desc->GetOutputDesc(kOutputDescIndex);
  GeTensorPtr output_ptr = MakeShared<ge::GeTensor>(out_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGE(OUT_OF_MEMORY, "Node [%s] make shared failed.", op_desc->GetName().c_str());
    return OUT_OF_MEMORY;
  }
  auto input_tensor_ptr = input.at(kInputDescIndex);
  if (input_tensor_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Node [%s] get input failed.", op_desc->GetName().c_str());
    return NOT_CHANGED;
  }
  if (output_ptr->SetData(input_tensor_ptr->GetData()) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
    return NOT_CHANGED;
  }
  v_output.emplace_back(output_ptr);
  GELOGD("IdentityKernel success: node[%s]", op_desc->GetName().c_str());

  return SUCCESS;
}
REGISTER_KERNEL(IDENTITY, IdentityKernel);
REGISTER_KERNEL(PLACEHOLDERWITHDEFAULT, IdentityKernel);
}  // namespace ge
