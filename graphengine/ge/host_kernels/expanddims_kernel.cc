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

#include "host_kernels/expanddims_kernel.h"

#include <memory>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const int kExpandDimsIndexZero = 0;
const size_t kExpandDimsOutputDescNum = 1;
const size_t kExpandDimsInputNum = 2;
}  // namespace
Status ExpanddimsKernel::Compute(const NodePtr &node_ptr) {
  GELOGI("Expanddims dimension kernel in.");
  if (node_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "parameter is nullptr");
    return PARAM_INVALID;
  }
  Status ret = KernelUtils::CheckDimensionNodeInfo(node_ptr);
  if (ret != SUCCESS) {
    GELOGW("GetDimensionNodeInfo failed");
    return ret;
  }

  if (!KernelUtils::CheckFormatSupported(node_ptr)) {
    GELOGW("CheckFormatSupported failed");
    return NOT_CHANGED;
  }
  GELOGI("Expanddims dimension kernel success.");
  return SUCCESS;
}
Status ExpanddimsKernel::Compute(const ge::OpDescPtr op_desc_ptr,
                                 const std::vector<ge::ConstGeTensorPtr> &input,
                                 std::vector<ge::GeTensorPtr> &v_output) {
  GELOGI("Expanddims folding kernel in.");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  if ((input.size() != kExpandDimsInputNum) || (op_desc_ptr->GetOutputsSize() != kExpandDimsOutputDescNum)) {
    GELOGW("Unexpected ExpandDims node, node input size: %zu, node output size: %zu, node name: %s", input.size(),
           op_desc_ptr->GetOutputsSize(), op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(kExpandDimsIndexZero);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("Failed to fold node %s, out of memory", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  // print output tensor information, and will be deleted
  GELOGI("Expanddims op %s output tensor data size is %zu", op_desc_ptr->GetName().c_str(),
         output_ptr->GetData().size());
  size_t data_dim_size = output_ptr->GetTensorDesc().GetShape().GetDims().size();
  GELOGI("Expanddims op %s output tensor dim size is %zu", op_desc_ptr->GetName().c_str(), data_dim_size);

  if (output_ptr->SetData(input.at(kExpandDimsIndexZero)->GetData()) != GRAPH_SUCCESS) {
    GELOGW("Compute: SetData failed");
  }
  v_output.emplace_back(output_ptr);
  GELOGI("Expanddims folding kernel success.");
  return SUCCESS;
}
REGISTER_KERNEL(EXPANDDIMS, ExpanddimsKernel);
}  // namespace ge
