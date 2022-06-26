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

#include "nnacl/infer/pad_infer.h"
#include "nnacl/infer/infer_register.h"

int PadInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                  OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  PadParameter *param = (PadParameter *)parameter;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (input->shape_size_ > DEFAULT_PAD_NDIMS) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  const TensorC *paddings = inputs[1];
  int size = GetElementNum(paddings);
  if (size > MAX_PAD_SIZE) {
    return NNACL_PARAM_INVALID;
  }
  if (paddings->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  param->padding_length = size;
  for (int i = 0; i < size; ++i) {
    MS_CHECK_TRUE_RET(((int *)paddings->data_)[i] >= 0, NNACL_INFER_INVALID);
    param->paddings_[i] = ((int *)paddings->data_)[i];
  }

  int output_shape[DEFAULT_PAD_NDIMS] = {0};
  size_t output_shape_size = 0;
  for (size_t i = 0; i < input->shape_size_; i++) {
    int shape = input->shape_[i] + param->paddings_[2 * i] + param->paddings_[2 * i + 1];
    ShapePush(output_shape, &output_shape_size, shape);
  }

  SetShapeArray(output, output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(Pad, PrimType_PadFusion, PadInferShape)
