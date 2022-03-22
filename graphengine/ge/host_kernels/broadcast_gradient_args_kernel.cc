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
#include "host_kernels/broadcast_gradient_args_kernel.h"

#include <vector>

#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/bcast.h"
#include "graph/passes/pass_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kBCastGradArgsInputsSize = 2;
const size_t kBCastGradArgsOutputsSize = 2;
}  // namespace

Status BroadcastGradientArgsKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                                            std::vector<GeTensorPtr> &v_output) {
  GELOGD("BroadcastGradientArgs kernel in");
  if (op_desc_ptr == nullptr) {
    GELOGE(PARAM_INVALID, "Parameter's invalid, Input opDescPtr is nullptr.");
    return PARAM_INVALID;
  }
  // check input size
  bool size_check_fail =
    (op_desc_ptr->GetAllInputsDesc().size() != kBCastGradArgsInputsSize || input.size() != kBCastGradArgsInputsSize ||
     op_desc_ptr->GetAllOutputsDesc().size() != kBCastGradArgsOutputsSize);
  if (size_check_fail) {
    GELOGW(
      "input/output size error. InDesc size:%zu,"
      "OutDesc size:%zu, in size:%zu ",
      op_desc_ptr->GetAllInputsDesc().size(), op_desc_ptr->GetAllOutputsDesc().size(), input.size());
    return NOT_CHANGED;
  }

  vector<int64_t> x1_dims;
  vector<int64_t> x2_dims;
  DataType x1_data_type = op_desc_ptr->GetInputDesc(0).GetDataType();
  DataType x2_data_type = op_desc_ptr->GetInputDesc(1).GetDataType();
  bool result = (OpUtils::GetShapeDataFromConstTensor(input[0], x1_data_type, x1_dims) == SUCCESS) &&
                (OpUtils::GetShapeDataFromConstTensor(input[1], x2_data_type, x2_dims) == SUCCESS);
  if (!result) {
    GELOGE(PARAM_INVALID, "Get shape data from const tensor fail.");
    return PARAM_INVALID;
  }

  BCast bcast;
  Status ret = bcast.GenerateBcastInfo(x1_dims, x2_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "Generate bcast info fail.");
    return ret;
  }

  vector<vector<int64_t>> grad_reduce_idx;
  grad_reduce_idx.push_back(bcast.GetGradXReduceIdx());
  grad_reduce_idx.push_back(bcast.GetGradYReduceIdx());

  for (size_t i = 0; i < grad_reduce_idx.size(); i++) {
    ret = PassUtils::ConstructTensorDescWithData(op_desc_ptr->GetOutputDesc(i), grad_reduce_idx[i], v_output);
    if (ret != SUCCESS) {
      GELOGE(ret, "BroadcastGradientArgs kernel construct tensor desc fail");
      return ret;
    }
  }

  return SUCCESS;
}

REGISTER_KERNEL(BROADCASTGRADIENTARGS, BroadcastGradientArgsKernel);
}  // namespace ge
