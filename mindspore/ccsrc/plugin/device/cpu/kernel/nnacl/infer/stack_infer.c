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

#include "nnacl/infer/stack_infer.h"
#include "nnacl/infer/infer_register.h"

int StackInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                    OpParameter *parameter) {
  int ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (ret != NNACL_OK) {
    return ret;
  }
  if (outputs_size != 1) {
    return NNACL_PARAM_INVALID;
  }
  if (inputs_size < 1) {
    return NNACL_PARAM_INVALID;
  }
  const TensorC *input = inputs[0];
  SetDataTypeFormat(outputs[0], input);
  StackParameter *param = (StackParameter *)parameter;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  int32_t output_shape[MAX_SHAPE_SIZE] = {0};
  size_t output_shape_size = 0;
  ShapeSet(output_shape, &output_shape_size, input->shape_, input->shape_size_);
  int axis = param->axis_ < 0 ? (int)(param->axis_) + (int)(input->shape_size_) + 1 : param->axis_;
  if (axis < 0 || axis > (int)(input->shape_size_)) {
    return NNACL_PARAM_INVALID;
  }

  for (size_t i = 1; i < inputs_size; ++i) {
    if (inputs[i]->shape_size_ != input->shape_size_) {
      return NNACL_PARAM_INVALID;
    }
    for (size_t j = 0; j < input->shape_size_; ++j) {
      if (inputs[i]->shape_[j] != input->shape_[j]) {
        return NNACL_PARAM_INVALID;
      }
    }
    if (inputs[i]->data_type_ != input->data_type_) {
      return NNACL_PARAM_INVALID;
    }
  }
  int insert_ret = ShapeInsert(output_shape, &output_shape_size, axis, inputs_size);
  if (insert_ret != NNACL_OK) {
    return NNACL_ERR;
  }
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(Stack, PrimType_Stack, StackInferShape)
