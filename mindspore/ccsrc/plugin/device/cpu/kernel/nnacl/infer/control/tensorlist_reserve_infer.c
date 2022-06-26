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

#include "nnacl/infer/control/tensorlist_reserve_infer.h"
#include "nnacl/infer/infer_register.h"
#include "nnacl/tensorlist_parameter.h"

int TensorListReserveInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs,
                                size_t outputs_size, OpParameter *parameter) {
  int check_ret = CheckAugmentWithMinSize(inputs, inputs_size, outputs, outputs_size, parameter, 2, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  TensorListParameter *reserve_param = (TensorListParameter *)parameter;
  const TensorC *input0 = inputs[0];
  int ele_shape_type = input0->data_type_;
  if (ele_shape_type != kNumberTypeInt && ele_shape_type != kNumberTypeInt32) {
    return NNACL_ERR;
  }

  TensorListC *output = (TensorListC *)(outputs[0]);
  output->data_type_ = kObjectTypeTensorType;
  output->format_ = Format_NHWC;
  output->tensors_data_type_ = reserve_param->element_dtype_;

  if (input0->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  int *ele_shape_ptr = (int *)(input0->data_);

  const TensorC *input1 = inputs[1];
  int num_ele_type = input1->data_type_;
  if (num_ele_type != kNumberTypeInt && ele_shape_type != kNumberTypeInt32) {
    return NNACL_ERR;
  }
  if (input1->data_ == NULL) {
    return NNACL_INFER_INVALID;
  }
  if (GetElementNum(input1) != 1) {
    return NNACL_ERR;
  }
  int num_elements = ((int *)(input1->data_))[0];
  ShapeSet(output->element_shape_, &(output->element_shape_size_), ele_shape_ptr, (size_t)GetElementNum(input0));
  output->element_num_ = (size_t)(num_elements);

  vvector tmp_shape;
  tmp_shape.size_ = (size_t)(num_elements);
  tmp_shape.shape_ = (int **)malloc(tmp_shape.size_ * sizeof(int *));
  if (tmp_shape.shape_ == NULL) {
    return NNACL_NULL_PTR;
  }
  tmp_shape.shape_size_ = (int *)malloc(tmp_shape.size_ * sizeof(int));
  if (tmp_shape.shape_size_ == NULL) {
    free(tmp_shape.shape_);
    return NNACL_NULL_PTR;
  }

  for (size_t i = 0; i < num_elements; ++i) {
    tmp_shape.shape_size_[i] = output->element_shape_size_;
    tmp_shape.shape_[i] = output->element_shape_;
  }
  int ret = MallocTensorListData(output, reserve_param->element_dtype_, &tmp_shape);
  free(tmp_shape.shape_size_);
  free(tmp_shape.shape_);
  return ret;
}

REG_INFER(TensorListReserve, PrimType_TensorListReserve, TensorListReserveInferShape)
