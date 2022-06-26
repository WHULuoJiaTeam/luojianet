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

#include "nnacl/infer/uniform_real_infer.h"
#include "nnacl/infer/infer_register.h"

int UniformRealInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                          OpParameter *parameter) {
  int ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 1, 1);
  if (ret != NNACL_OK) {
    return ret;
  }
  outputs[0]->data_type_ = kNumberTypeFloat32;
  outputs[0]->format_ = inputs[0]->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  int32_t *input_data = (int32_t *)(inputs[0]->data_);
  if (input_data == NULL) {
    return NNACL_INFER_INVALID;
  }
  int input_num = GetElementNum(inputs[0]);
  if (input_num > MAX_SHAPE_SIZE || input_num < 0) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int output_shape[MAX_SHAPE_SIZE];
  size_t output_shape_size = (size_t)(input_num);
  for (int i = 0; i < input_num; i++) {
    output_shape[i] = input_data[i];
  }
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(UniformReal, PrimType_UniformReal, UniformRealInferShape)
