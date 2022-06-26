/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/activation_grad_infer.h"
#include "nnacl/infer/infer_register.h"

int ActivationGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                             OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }
  const TensorC *input = inputs[0];
  const TensorC *input_grad = inputs[1];
  if (input->shape_size_ != input_grad->shape_size_) {
    return NNACL_ERR;
  }
  for (size_t i = 0; i < input->shape_size_; i++) {
    if (input->shape_[i] != input_grad->shape_[i]) {
      return NNACL_ERR;
    }
  }

  SetDataTypeFormat(outputs[0], inputs[0]);
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  SetShapeTensor(outputs[0], inputs[0]);
  return NNACL_OK;
}

REG_INFER(ActivationGrad, PrimType_ActivationGrad, ActivationGradInferShape)
