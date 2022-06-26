
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

#include "src/runtime/kernel/arm/fp32_grad/strided_slice_grad.h"
#include <vector>
#include <algorithm>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32_grad/strided_slice_grad.h"
#include "src/ops/populate/strided_slice_populate.h"
#include "include/errorcode.h"

using luojianet_ms::kernel::KERNEL_ARCH;
using luojianet_ms::lite::KernelRegistrar;
using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_OK;
using luojianet_ms::schema::PrimitiveType_StridedSliceGrad;

namespace luojianet_ms::kernel {
int StridedSliceGradCPUKernel::Prepare() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  param_ = reinterpret_cast<StridedSliceParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param_);
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  auto input = in_tensors_.at(0);
  CHECK_NULL_RETURN(input);
  if (input->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Not supported data type: " << input->data_type();
    return RET_ERROR;
  }
  param_->data_type = kDataTypeFloat;
  return ReSize();
}

void StridedSliceGradCPUKernel::FillEmptyDims() {
  int32_t begins[DIMENSION_8D];
  int32_t ends[DIMENSION_8D];
  int32_t strides[DIMENSION_8D];
  int32_t input_shape[DIMENSION_8D];
  int32_t i;

  // invert the order of the dimension and fill defout outsize actual ranae
  for (i = 0; i < DIMENSION_8D; ++i) {
    begins[i] = param_->begins_[i];
    ends[i] = param_->ends_[i];
    strides[i] = param_->strides_[i];
    input_shape[i] = param_->in_shape_[i];
  }

  int32_t real_index = param_->in_shape_length_ - 1;
  for (i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param_->in_shape_[i] = input_shape[real_index--];
    } else {
      param_->in_shape_[i] = 1;
    }
  }
  int out_shape_length = in_tensors_.at(1)->shape().at(0);
  real_index = out_shape_length - 1;
  for (i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      param_->begins_[i] = begins[real_index];
      param_->ends_[i] = ends[real_index];
      param_->strides_[i] = strides[real_index--];
    } else {
      param_->begins_[i] = 0;
      param_->ends_[i] = 1;
      param_->strides_[i] = 1;
    }
  }
  param_->num_axes_ = DIMENSION_8D;
  param_->in_shape_length_ = DIMENSION_8D;

  for (i = 0; i < DIMENSION_8D; ++i) {
    if (param_->begins_[i] < 0) {
      param_->begins_[i] += param_->in_shape_[i];
    }
    if (param_->ends_[i] < 0) {
      param_->ends_[i] += param_->in_shape_[i];
    }
  }
}

void StridedSliceGradCPUKernel::FillOutputDim() {
  auto output = out_tensors_.at(0);
  size_t out_size = output->shape().size();
  for (size_t i = 0; i < DIMENSION_8D; i++) {
    if (i < out_size) {
      output_shape_.push_back(output->shape()[i]);
    } else {
      output_shape_.insert(output_shape_.begin(), 1);
    }
  }
}

int StridedSliceGradCPUKernel::ReSize() {
  FillEmptyDims();
  FillOutputDim();
  return RET_OK;
}

int StridedSliceGradImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto slice = reinterpret_cast<StridedSliceGradCPUKernel *>(cdata);
  auto error_code = slice->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "StridedSliceGrad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, StridedSliceGradImpl, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Strided slice error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int StridedSliceGradCPUKernel::DoExecute(int task_id) {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);

  int *po = output_shape_.data();
  auto dx = reinterpret_cast<float *>(output->MutableData());
  auto dy = reinterpret_cast<float *>(input->MutableData());
  CHECK_NULL_RETURN(po);
  CHECK_NULL_RETURN(dx);
  CHECK_NULL_RETURN(dy);
  std::fill(dx, dx + output->ElementsNum(), 0.f);
  auto ret = DoStridedSliceGrad(dy, dx, po, param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "StridedSliceGrad error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_StridedSliceGrad, LiteKernelCreator<StridedSliceGradCPUKernel>)
}  // namespace luojianet_ms::kernel
