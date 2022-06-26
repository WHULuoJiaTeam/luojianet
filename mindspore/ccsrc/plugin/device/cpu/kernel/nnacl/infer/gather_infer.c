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

#include "nnacl/infer/gather_infer.h"
#include "nnacl/infer/infer_register.h"

int GatherInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                     OpParameter *parameter) {
  int ret = CheckAugmentNull(inputs, inputs_size, outputs, outputs_size, parameter);
  if (ret != NNACL_OK) {
    return ret;
  }
  const size_t kMinimumGradInputsNum = 3;
  if (inputs_size < kMinimumGradInputsNum || outputs_size != 1) {
    return NNACL_ERR;
  }
  const TensorC *input = inputs[0];
  const TensorC *indices = inputs[1];
  TensorC *output = outputs[0];
  output->data_type_ = input->data_type_;
  if (parameter->quant_type_ == QuantType_QUANT_WEIGHT || parameter->quant_type_ == QuantType_QUANT_DYNAMIC) {
    output->data_type_ = kNumberTypeFloat32;
  }
  output->format_ = input->format_;
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }
  if (input->shape_size_ > MAX_SHAPE_SIZE || indices->shape_size_ > MAX_SHAPE_SIZE) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (inputs[2]->data_ == NULL) {
    return NNACL_NULL_PTR;
  }
  if (GetElementNum(inputs[2]) < 1) {
    return NNACL_ERR;
  }
  int axis = *((int *)inputs[2]->data_);
  if (axis < 0) {
    axis += input->shape_size_;
  }
  int indices_shape[MAX_SHAPE_SIZE];
  size_t indices_shape_size = 0;
  ShapeSet(indices_shape, &indices_shape_size, indices->shape_, indices->shape_size_);
  size_t indices_rank = indices_shape_size;
  int in_shape[MAX_SHAPE_SIZE] = {0};
  size_t in_shape_size = 0;
  ShapeSet(in_shape, &in_shape_size, input->shape_, input->shape_size_);
  if ((int)(in_shape_size) < axis + 1) {
    return NNACL_ERR;
  }
  int out_shape[MAX_SHAPE_SIZE] = {0};
  size_t out_shape_size = 0;
  ShapeSet(out_shape, &out_shape_size, in_shape, in_shape_size);
  int erase_ret = ShapeErase(out_shape, &out_shape_size, axis);
  if (erase_ret != NNACL_OK) {
    return NNACL_ERR;
  }
  for (int i = (int)(indices_rank - 1); i >= 0; --i) {
    ret = ShapeInsert(out_shape, &out_shape_size, axis, indices_shape[i]);
    if (ret != NNACL_OK) {
      return NNACL_ERR;
    }
  }
  SetShapeArray(output, out_shape, out_shape_size);
  return NNACL_OK;
}

REG_INFER(Gather, PrimType_Gather, GatherInferShape)
