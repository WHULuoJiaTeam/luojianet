/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "nnacl/infer/group_conv2d_grad_input_infer.h"

int GroupConv2dGradInputInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                   size_t outputs_size, OpParameter *parameter) {
  int ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (ret != NNACL_OK) {
    return ret;
  }
  if (inputs_size < 2 || outputs_size != 1) {
    return NNACL_ERR;
  }

  const TensorC *in0 = inputs[0];
  TensorC *out = outputs[0];

  SetDataTypeFormat(out, in0);

  size_t shape_size = in0->shape_size_;
  if (shape_size > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int shape_[MAX_SHAPE_SIZE];
  for (size_t i = 0; i < shape_size; i++) {
    shape_[i] = in0->shape_[i];
  }
  SetShapeArray(out, shape_, shape_size);

  return NNACL_OK;
}
