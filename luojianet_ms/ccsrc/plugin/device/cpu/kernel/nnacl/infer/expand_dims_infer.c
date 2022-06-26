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

#include "nnacl/infer/expand_dims_infer.h"
#include "nnacl/infer/infer_register.h"

int ExpandDimsInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                         OpParameter *parameter) {
  int check_ret = CheckAugmentNullOutputSize(inputs, inputs_size, outputs, outputs_size, parameter, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  TensorC *output = outputs[0];
  SetDataTypeFormat(output, input);
  if (parameter->quant_type_ == QuantType_QUANT_WEIGHT) {
    output->data_type_ = kNumberTypeFloat32;
  }
  if (!InferFlag(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  if (inputs_size < C2NUM) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  if (inputs[1]->data_ == NULL) {
    return NNACL_INPUT_TENSOR_ERROR;
  }
  if (GetElementNum(inputs[1]) < 1) {
    return NNACL_ERR;
  }
  int dim = ((int32_t *)(inputs[1]->data_))[0];
  if (dim < 0) {
    dim += (int)(input->shape_size_) + 1;
  }
  if (dim > (int)(input->shape_size_)) {
    return NNACL_INPUT_TENSOR_ERROR;
  }

  ShapeSet(output->shape_, &(output->shape_size_), input->shape_, input->shape_size_);
  int ret = ShapeInsert(output->shape_, &(output->shape_size_), dim, 1);
  if (ret != NNACL_OK) {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

REG_INFER(ExpandDims, PrimType_ExpandDims, ExpandDimsInferShape)
