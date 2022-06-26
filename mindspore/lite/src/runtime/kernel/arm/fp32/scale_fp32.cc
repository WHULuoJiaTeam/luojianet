/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/scale_fp32.h"
#include <cstring>
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::kernel {
namespace {
int CheckInputsOutputsDataType(const std::vector<lite::Tensor *> &in_tensors,
                               const std::vector<lite::Tensor *> &out_tensors) {
  if (std::any_of(in_tensors.begin(), in_tensors.end(), [](const lite::Tensor *input) {
        return input->data_type() != kNumberTypeFloat && input->data_type() != kNumberTypeFloat32 &&
               input->data_type() != kNumberTypeFloat16;
      })) {
    MS_LOG(ERROR) << "scale op input data type should float32";
    return RET_ERROR;
  }
  if (std::any_of(out_tensors.begin(), out_tensors.end(), [](const lite::Tensor *output) {
        return output->data_type() != kNumberTypeFloat && output->data_type() != kNumberTypeFloat32 &&
               output->data_type() != kNumberTypeFloat16;
      })) {
    MS_LOG(ERROR) << "scale op output data type should float32";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace
ScaleCPUKernel::~ScaleCPUKernel() {
  if (scale_param_->const_scale_) {
    if (scale_ != nullptr) {
      free(scale_);
      scale_ = nullptr;
    }
  }
  if (scale_param_->const_offset_) {
    if (offset_ != nullptr) {
      free(offset_);
      offset_ = nullptr;
    }
  }
}

int ScaleCPUKernel::InitScaleOffset() {
  auto scale_tensor = in_tensors_.at(1);
  if (reinterpret_cast<float *>(scale_tensor->data()) != nullptr) {
    scale_param_->const_scale_ = true;
    MS_CHECK_TRUE_RET(scale_tensor->ElementsNum() > 0, RET_ERROR);
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, scale_tensor->ElementsNum() * sizeof(float));
    scale_ = reinterpret_cast<float *>(malloc(scale_tensor->ElementsNum() * sizeof(float)));
    if (scale_ == nullptr) {
      MS_LOG(ERROR) << "Malloc buffer failed.";
      return RET_ERROR;
    }
    (void)memcpy(scale_, scale_tensor->data(), scale_tensor->ElementsNum() * sizeof(float));
  } else {
    scale_param_->const_scale_ = false;
    scale_ = nullptr;
  }

  if (in_tensors_.size() == 2) {
    scale_param_->const_offset_ = true;
    offset_ = reinterpret_cast<float *>(malloc(scale_tensor->ElementsNum() * sizeof(float)));
    if (offset_ == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
    memset(offset_, 0, scale_tensor->ElementsNum() * sizeof(float));
  } else if (in_tensors_.size() == C3NUM && reinterpret_cast<float *>(in_tensors_.at(THIRD_INPUT)->data()) != nullptr) {
    scale_param_->const_offset_ = true;
    auto offset_tensor = in_tensors_.at(2);
    MS_CHECK_TRUE_RET(scale_tensor->ElementsNum() == offset_tensor->ElementsNum(), RET_ERROR);
    offset_ = reinterpret_cast<float *>(malloc(offset_tensor->ElementsNum() * sizeof(float)));
    if (offset_ == nullptr) {
      MS_LOG(ERROR) << "Malloc data failed";
      return RET_ERROR;
    }
    (void)memcpy(offset_, offset_tensor->data(), offset_tensor->ElementsNum() * sizeof(float));
  } else {
    scale_param_->const_offset_ = false;
    offset_ = nullptr;
  }

  return RET_OK;
}

int ScaleCPUKernel::CalculateParameter() {
  auto in_tensor = in_tensors_.at(0);
  auto in_shape = in_tensor->shape();
  auto scale_tensor = in_tensors_.at(1);
  auto scale_shape = scale_tensor->shape();

  if (scale_param_->axis_ < 0) {
    scale_param_->axis_ = scale_param_->axis_ + in_shape.size();
  }
  if (scale_param_->axis_ < 0 || scale_shape.size() + scale_param_->axis_ > in_shape.size()) {
    MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
    return RET_ERROR;
  }
  scale_param_->outer_size_ = 1;
  scale_param_->axis_size_ = 1;
  scale_param_->inner_size_ = 1;
  for (int i = 0; i < scale_param_->axis_; i++) {
    scale_param_->outer_size_ *= in_shape.at(i);
  }
  for (size_t i = 0; i < scale_shape.size(); i++) {
    if (in_shape.at(i + scale_param_->axis_) != scale_shape.at(i)) {
      MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
      return RET_ERROR;
    }
    scale_param_->axis_size_ *= in_shape.at(i + scale_param_->axis_);
  }
  for (size_t i = scale_param_->axis_ + scale_shape.size(); i < in_shape.size(); i++) {
    scale_param_->inner_size_ *= in_shape.at(i);
  }
  scale_param_->op_parameter_.thread_num_ = MSMIN(scale_param_->op_parameter_.thread_num_, scale_param_->outer_size_);
  return RET_OK;
}

int ScaleCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto ret = CheckInputsOutputsDataType(in_tensors_, out_tensors_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale inputs or outputs data type is invalid.";
    return RET_ERROR;
  }
  ret = InitScaleOffset();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp32 InitScaleOffset failed.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  ret = ReSize();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp32 Resize failed";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleCPUKernel::ReSize() {
  auto ret = CalculateParameter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp32 CalculateParameter failed.";
    return RET_ERROR;
  }

  return RET_OK;
}

int ScaleCPUKernel::Scale(int task_id) {
  switch (scale_param_->activation_type_) {
    case schema::ActivationType_RELU6:
      DoScaleRelu6(input_ptr_, output_ptr_, scale_, offset_, task_id, scale_param_);
      break;
    case schema::ActivationType_RELU:
      DoScaleRelu(input_ptr_, output_ptr_, scale_, offset_, task_id, scale_param_);
      break;
    case schema::ActivationType_NO_ACTIVATION:
      DoScale(input_ptr_, output_ptr_, scale_, offset_, task_id, scale_param_);
      break;
    default:
      MS_LOG(ERROR) << "Scale does not support activation type " << scale_param_->activation_type_;
      return RET_ERROR;
  }
  return RET_OK;
}

int ScaleRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto scale = reinterpret_cast<ScaleCPUKernel *>(cdata);
  auto ret = scale->Scale(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScaleRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleCPUKernel::Run() {
  auto in_tensor = in_tensors_.front();
  input_ptr_ = reinterpret_cast<float *>(in_tensor->data());
  CHECK_NULL_RETURN(input_ptr_);
  if (!scale_param_->const_scale_) {
    auto scale_tensor = in_tensors_.at(1);
    scale_ = reinterpret_cast<float *>(scale_tensor->data());
    CHECK_NULL_RETURN(scale_);
  }
  if (!scale_param_->const_offset_) {
    auto offset_tensor = in_tensors_.at(2);
    offset_ = reinterpret_cast<float *>(offset_tensor->data());
    CHECK_NULL_RETURN(offset_);
  }
  auto out_tensor = out_tensors_.front();
  output_ptr_ = reinterpret_cast<float *>(out_tensor->MutableData());

  auto ret = ParallelLaunch(this->ms_context_, ScaleRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ScaleFusion, LiteKernelCreator<ScaleCPUKernel>)
}  // namespace mindspore::kernel
