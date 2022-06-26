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

#include "nnacl/infer/strided_slice_grad_infer.h"
#include "nnacl/infer/infer_register.h"

bool StridedSliceCheckInputs(const TensorC *const *inputs, size_t inputs_size) {
  for (size_t i = 1; i < inputs_size; ++i) {
    if (inputs[i]->data_ == NULL) {
      return false;
    }
  }
  if (GetElementNum(inputs[2]) > MAX_SHAPE_SIZE) {
    return false;
  }
  if (GetElementNum(inputs[2]) != GetElementNum(inputs[3]) && GetElementNum(inputs[2]) != GetElementNum(inputs[4])) {
    return false;
  }
  return true;  // note: the original code is ndim_ <= in_shape_size
}

void ApplyBeginEndEllipsisMask(size_t ndim, int *begins, const uint32_t *const begins_mask, int *ends,
                               const uint32_t *const ends_mask, const uint32_t *const ellipsis_mask,
                               const int *const in_shape) {
  for (size_t i = 0; i < ndim; i++) {
    if (begins_mask[i]) {
      begins[i] = 0;
    }
    if (ends_mask[i]) {
      ends[i] = in_shape[i];
    }
  }
  for (size_t i = 0; i < ndim; i++) {
    if (ellipsis_mask[i]) {
      begins[i] = 0;
      ends[i] = in_shape[i];
      break;
    }
  }
}

int StridedSliceGradInferShape(const TensorC *const *inputs, size_t inputs_size, TensorC **outputs, size_t outputs_size,
                               OpParameter *parameter) {
  int check_ret = CheckAugmentNullSize(inputs, inputs_size, outputs, outputs_size, parameter, 5, 1);
  if (check_ret != NNACL_OK) {
    return check_ret;
  }

  const TensorC *input = inputs[0];
  SetDataTypeFormat(outputs[0], input);
  bool inferflag = InferFlag(inputs, inputs_size);

  int in_shape_[MAX_SHAPE_SIZE] = {0};
  size_t in_shape_size = 0;
  if (inferflag) {
    ShapeSet(in_shape_, &in_shape_size, input->shape_, input->shape_size_);
  }
  int begins_[MAX_SHAPE_SIZE] = {0};
  size_t begins_size = 0;
  int ends_[MAX_SHAPE_SIZE] = {0};
  size_t ends_size = 0;
  int strides_[MAX_SHAPE_SIZE] = {0};
  size_t strides_size = 0;

  if (!StridedSliceCheckInputs(inputs, inputs_size)) {
    return NNACL_INFER_INVALID;
  }

  // input order: dy, shapex, begins, ends, strides.
  const TensorC *begin_tensor = inputs[2];
  int *begin_data = (int *)(begin_tensor->data_);
  int *end_data = (int *)(inputs[3]->data_);
  int *stride_data = (int *)(inputs[4]->data_);

  size_t ndim_ = (size_t)GetElementNum(begin_tensor);
  for (size_t i = 0; i < ndim_; ++i) {
    ShapePush(begins_, &begins_size, begin_data[i]);
    ShapePush(ends_, &ends_size, end_data[i]);
    ShapePush(strides_, &strides_size, stride_data[i]);
  }

  // set all mask to original input shape
  uint32_t begins_mask_[MAX_SHAPE_SIZE] = {0};
  uint32_t ends_mask_[MAX_SHAPE_SIZE] = {0};
  uint32_t ellipsis_mask_[MAX_SHAPE_SIZE] = {0};
  uint32_t new_axis_mask_[MAX_SHAPE_SIZE] = {0};

  StridedSliceParameter *param = (StridedSliceParameter *)parameter;
  for (size_t i = 0; i < ndim_; i++) {
    begins_mask_[i] = (unsigned)(param->begins_mask_) & (1 << i);
    ends_mask_[i] = (unsigned)(param->ends_mask_) & (1 << i);
    ellipsis_mask_[i] = (unsigned)(param->ellipsisMask_) & (1 << i);
    new_axis_mask_[i] = (unsigned)(param->newAxisMask_) & (1 << i);
  }
  param->num_axes_ = (int)(in_shape_size);
  param->in_shape_length_ = (int)(in_shape_size);
  for (size_t i = 0; i < ndim_; ++i) {
    param->begins_[i] = begins_[i];
    param->ends_[i] = ends_[i];
    param->strides_[i] = strides_[i];
  }
  ShapeSet(param->in_shape_, &in_shape_size, input->shape_, input->shape_size_);
  // ApplyNewAxisMask;
  for (size_t i = 0; i < ndim_; i++) {
    if (new_axis_mask_[i]) {
      ndim_ += 1;
      int ret = ShapeInsert(in_shape_, &in_shape_size, i, 1);
      if (ret != NNACL_OK) {
        return NNACL_ERR;
      }
      begins_[i] = 0;
      ends_[i] = 1;
      strides_[i] = 1;

      ShapePush(begins_, &begins_size, 0);
      ShapePush(ends_, &ends_size, in_shape_[ndim_ - 1]);
      ShapePush(strides_, &strides_size, 1);

      begins_mask_[i] = false;
      ends_mask_[i] = false;
      ellipsis_mask_[i] = false;
    }
  }
  ApplyBeginEndEllipsisMask(ndim_, begins_, begins_mask_, ends_, ends_mask_, ellipsis_mask_, in_shape_);
  if (!inferflag) {
    return NNACL_OK;
  }
  int output_size = inputs[1]->shape_[0];
  int output_shape[MAX_SHAPE_SIZE] = {0};
  size_t output_shape_size = 0;
  if (inputs[1]->data_ == NULL) {
    return NNACL_ERR;
  }

  if (output_size > MAX_SHAPE_SIZE) {
    return NNACL_ERR;
  }
  for (int i = 0; i < output_size; i++) {
    ShapePush(output_shape, &output_shape_size, ((int *)(inputs[1]->data_))[i]);
  }
  SetShapeArray(outputs[0], output_shape, output_shape_size);
  return NNACL_OK;
}

REG_INFER(StridedSliceGrad, PrimType_StridedSliceGrad, StridedSliceGradInferShape)
