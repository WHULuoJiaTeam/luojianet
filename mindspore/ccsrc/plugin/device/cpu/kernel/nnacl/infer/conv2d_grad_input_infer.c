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

#include "nnacl/infer/conv2d_grad_input_infer.h"
#include "nnacl/infer/infer_register.h"

int Conv2dGradInputInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                              OpParameter *parameter) {
  int ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (ret != NNACL_OK) {
    return ret;
  }
  if (inputs_size < 3 || outputs_size != 1) {
    return NNACL_ERR;
  }
  const TensorC *in0 = inputs[0];
  TensorC *out = outputs[0];

  if (in0 == NULL || out == NULL) {
    return NNACL_NULL_PTR;
  }
  if (in0->format_ != Format_NHWC) {
    return NNACL_FORMAT_ERROR;
  }
  SetDataTypeFormat(out, in0);

  if (inputs[2]->shape_size_ < 1 || inputs[2]->data_ == NULL) {
    return NNACL_ERR;
  }
  size_t data_size = (size_t)inputs[2]->shape_[0];
  if (data_size != 4) {
    return NNACL_ERR;
  }
  int shape[MAX_SHAPE_SIZE];
  const int nchw2nhwc[4] = {0, 2, 3, 1};
  for (size_t i = 0; i < data_size; i++) {
    shape[i] = *((int *)(inputs[2]->data_) + nchw2nhwc[i]);
  }
  SetShapeArray(out, shape, data_size);

  return NNACL_OK;
}

REG_INFER(Conv2DBackpropInputFusion, PrimType_Conv2DBackpropInputFusion, Conv2dGradInputInferShape)
