/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/int8/scale_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::kernel {
namespace {
constexpr size_t kScaleInputsSize = 2;
constexpr size_t kScaleBiasInputsSize = 3;
constexpr int kOffsetIndex = 2;
}  // namespace
ScaleInt8CPUKernel::~ScaleInt8CPUKernel() {
  if (tile_para != nullptr) {
    free(tile_para);
    tile_para = nullptr;
  }
  if (input1_data_ != nullptr && malloced_scale_) {
    free(input1_data_);
    input1_data_ = nullptr;
  }
  if (input2_data_ != nullptr && malloced_offset_) {
    free(input2_data_);
    input2_data_ = nullptr;
  }
}

int ScaleInt8CPUKernel::InitScaleOffset() {
  CalcMultiplesAndStrides(tile_para);
  scale_param_->const_scale_ = false;
  auto *scale_ptr = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data());
  // scale may be const value ,can be processed in prepare stage
  if (scale_ptr != nullptr) {
    scale_param_->const_scale_ = true;
    input1_data_ = scale_ptr;
    // need broadcasting
    if (in_tensors_.at(0)->ElementsNum() != in_tensors_.at(1)->ElementsNum()) {
      input1_data_ = reinterpret_cast<int8_t *>(malloc(out_tensors_.at(0)->Size()));
      if (input1_data_ == nullptr) {
        MS_LOG(ERROR) << "malloc input1_data_  failed.";
        return RET_ERROR;
      }
      malloced_scale_ = true;
      TileOneDimensionInt8(reinterpret_cast<int8_t *>(in_tensors_.at(1)->data()),
                           reinterpret_cast<int8_t *>(input1_data_), 0, tile_para->ndim_, tile_para->in_shape1_,
                           tile_para->in_strides1_, tile_para->out_strides_, tile_para->multiples1_);
    }
  }

  scale_param_->const_offset_ = false;
  if (in_tensors_.size() == kScaleBiasInputsSize) {
    has_bias_ = true;
    auto offset_tensor = in_tensors_.at(kOffsetIndex);
    auto *offset_ptr = reinterpret_cast<int8_t *>(offset_tensor->data());
    // offset may be const value ,can be processed in prepare stage
    if (offset_ptr != nullptr) {
      scale_param_->const_offset_ = true;
      input2_data_ = offset_ptr;
      // need broadcasting
      if (in_tensors_.at(0)->ElementsNum() != in_tensors_.at(kOffsetIndex)->ElementsNum()) {
        input2_data_ = reinterpret_cast<int8_t *>(malloc(out_tensors_.at(0)->Size()));
        if (input2_data_ == nullptr) {
          MS_LOG(ERROR) << "malloc input2_data_  failed.";
          if (malloced_scale_) {
            free(input1_data_);
            input1_data_ = nullptr;
          }
          return RET_ERROR;
        }
        malloced_offset_ = true;
        TileOneDimensionInt8(reinterpret_cast<int8_t *>(in_tensors_.at(kOffsetIndex)->data()),
                             reinterpret_cast<int8_t *>(input2_data_), 0, tile_para->ndim_, tile_para->in_shape1_,
                             tile_para->in_strides1_, tile_para->out_strides_, tile_para->multiples1_);
      }
    }
  }

  return RET_OK;
}

int ScaleInt8CPUKernel::InitParameter() {
  auto input0_shape = in_tensors_[FIRST_INPUT]->shape();
  auto input1_shape = in_tensors_[SECOND_INPUT]->shape();
  if (scale_param_->axis_ < 0) {
    scale_param_->axis_ += input0_shape.size();
  }
  if (input1_shape.size() + scale_param_->axis_ > input0_shape.size()) {
    MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < input1_shape.size(); i++) {
    if (input0_shape[i + scale_param_->axis_] != input1_shape[i]) {
      MS_LOG(ERROR) << "Scale tensor shape is incorrect.";
      return RET_ERROR;
    }
  }

  if (tile_para == nullptr) {
    tile_para = reinterpret_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  }
  MS_CHECK_TRUE_MSG(tile_para != nullptr, RET_ERROR, "scale's arithmetic-param is a nullptr.");
  auto out_shape = out_tensors_.front()->shape();
  tile_para->ndim_ = out_shape.size();
  int i = 0;
  for (; i < scale_param_->axis_; ++i) {
    tile_para->in_shape0_[i] = input0_shape[i];
    tile_para->in_shape1_[i] = 1;
    tile_para->out_shape_[i] = out_shape[i];
  }
  for (; i < static_cast<int>(input1_shape.size()) + scale_param_->axis_; ++i) {
    tile_para->in_shape0_[i] = input0_shape[i];
    tile_para->in_shape1_[i] = input0_shape[i];
    tile_para->out_shape_[i] = out_shape[i];
  }
  for (; i < static_cast<int>(tile_para->ndim_); ++i) {
    tile_para->in_shape0_[i] = input0_shape[i];
    tile_para->in_shape1_[i] = 1;
    tile_para->out_shape_[i] = out_shape[i];
  }
  return RET_OK;
}

int ScaleInt8CPUKernel::InitQuantArgs() {
  auto input = in_tensors_.at(0);
  auto scale = in_tensors_.at(1);
  auto output = out_tensors_.at(0);
  CHECK_LESS_RETURN(input->quant_params().size(), 1);
  CHECK_LESS_RETURN(scale->quant_params().size(), 1);
  CHECK_LESS_RETURN(output->quant_params().size(), 1);
  auto input_scale = input->quant_params().front().scale;
  auto scale_scale = scale->quant_params().front().scale;
  auto output_scale = output->quant_params().front().scale;
  scale_param_->input_zp_ = input->quant_params().front().zeroPoint;
  scale_param_->scale_zp_ = scale->quant_params().front().zeroPoint;
  scale_param_->output_zp_ = output->quant_params().front().zeroPoint;

  // (in * scale + offset) / output
  const double input_output_multiplier = input_scale * scale_scale / output_scale;
  int shift;
  QuantizeMultiplier(input_output_multiplier, &scale_param_->scale_mul_arg_.multiplier_, &shift);
  scale_param_->scale_mul_arg_.left_shift_ = shift > 0 ? shift : 0;
  scale_param_->scale_mul_arg_.right_shift_ = shift < 0 ? -shift : 0;

  if (in_tensors_.size() == kScaleBiasInputsSize) {
    auto offset = in_tensors_.at(kOffsetIndex);
    CHECK_LESS_RETURN(offset->quant_params().size(), 1);
    auto offset_scale = offset->quant_params().front().scale;
    scale_param_->offset_zp_ = offset->quant_params().front().zeroPoint;

    const double offset_multiplier = offset_scale / output_scale;
    QuantizeMultiplier(offset_multiplier, &scale_param_->offset_mul_arg_.multiplier_, &shift);
    scale_param_->offset_mul_arg_.left_shift_ = shift > 0 ? shift : 0;
    scale_param_->offset_mul_arg_.right_shift_ = shift < 0 ? -shift : 0;
  }

  switch (scale_param_->activation_type_) {
    case schema::ActivationType_RELU:
      scale_param_->output_activation_min_ = 0;
      scale_param_->output_activation_max_ = INT8_MAX;
      break;
    case schema::ActivationType_RELU6:
      scale_param_->output_activation_min_ = 0;
      scale_param_->output_activation_max_ = 6;
      break;
    case schema::ActivationType_NO_ACTIVATION:
      scale_param_->output_activation_min_ = INT8_MIN;
      scale_param_->output_activation_max_ = INT8_MAX;
      break;
    default:
      MS_LOG(ERROR) << "Scale does not support activation type " << scale_param_->activation_type_;
      return RET_ERROR;
  }
  return RET_OK;
}

int ScaleInt8CPUKernel::Prepare() {
  if (in_tensors_.size() < kScaleInputsSize || in_tensors_.size() > kScaleBiasInputsSize) {
    MS_LOG(ERROR) << "inputs to Scale operator should be 2 or 3, but " << in_tensors_.size() << " is given.";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }

  if (ReSize() != RET_OK) {
    MS_LOG(ERROR) << "Scale prepare resize failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleInt8CPUKernel::ReSize() {
  auto ret = InitParameter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale int8 InitParameter failed.";
    return RET_ERROR;
  }

  ret = InitScaleOffset();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale int8 InitScaleOffset failed.";
    return RET_ERROR;
  }

  ret = InitQuantArgs();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale fp32 InitQuantArgs failed.";
    return ret;
  }
  return RET_OK;
}

int ScaleInt8CPUKernel::Scale(int task_id) const {
  MS_CHECK_INT_MUL_NOT_OVERFLOW(task_id, count_unit_, RET_ERROR);
  int real_dst_count = MSMIN(elements_num_ - task_id * count_unit_, count_unit_);
  if (real_dst_count <= 0) {
    return lite::RET_OK;
  }
  int8_t *cur_input0_data = input0_data_ + task_id * count_unit_;
  CHECK_NULL_RETURN(cur_input0_data);
  int8_t *cur_input1_data = input1_data_ + task_id * count_unit_;
  CHECK_NULL_RETURN(cur_input1_data);
  int8_t *cur_output_data = output_data_ + task_id * count_unit_;
  CHECK_NULL_RETURN(cur_output_data);

  if (has_bias_) {
    int8_t *cur_input2_data = input2_data_ + task_id * count_unit_;
    CHECK_NULL_RETURN(cur_input2_data);
    DoScaleWithBiasInt8(cur_input0_data, cur_output_data, cur_input1_data, cur_input2_data, scale_param_,
                        real_dst_count);
  } else {
    DoScaleInt8(cur_input0_data, cur_output_data, cur_input1_data, scale_param_, real_dst_count);
  }
  return RET_OK;
}

int ScaleRunInt8(void *cdata, int task_id, float, float) {
  auto scale = reinterpret_cast<ScaleInt8CPUKernel *>(cdata);
  auto ret = scale->Scale(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ScaleRunInt8 error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleInt8CPUKernel::Run() {
  elements_num_ = out_tensors_.at(0)->ElementsNum();
  count_unit_ = thread_count_ > 1 ? UP_DIV(elements_num_, thread_count_) : elements_num_;
  input0_data_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->data());
  output_data_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->data());
  int ret = RET_ERROR;
  // need broadcasting
  if (in_tensors_.at(0)->ElementsNum() != in_tensors_.at(1)->ElementsNum()) {
    // scale is passed by previous node, need do broadcasting online
    if (!scale_param_->const_scale_) {
      input1_data_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(out_tensors_.at(0)->Size()));
      if (input1_data_ == nullptr) {
        MS_LOG(ERROR) << "malloc input1_data_  failed.";
        return RET_ERROR;
      }
      TileOneDimensionInt8(reinterpret_cast<int8_t *>(in_tensors_.at(1)->data()),
                           reinterpret_cast<int8_t *>(input1_data_), 0, tile_para->ndim_, tile_para->in_shape1_,
                           tile_para->in_strides1_, tile_para->out_strides_, tile_para->multiples1_);
    }

    // If has bias, bias is passed by previous node case, need do broadcasting online
    if (has_bias_ && !scale_param_->const_offset_) {
      input2_data_ = reinterpret_cast<int8_t *>(ctx_->allocator->Malloc(out_tensors_.at(0)->Size()));
      if (input2_data_ == nullptr) {
        MS_LOG(ERROR) << "malloc input2_data_  failed.";
        ctx_->allocator->Free(input1_data_);
        input1_data_ = nullptr;
        return RET_ERROR;
      }
      TileOneDimensionInt8(reinterpret_cast<int8_t *>(in_tensors_.at(kOffsetIndex)->data()),
                           reinterpret_cast<int8_t *>(input2_data_), 0, tile_para->ndim_, tile_para->in_shape1_,
                           tile_para->in_strides1_, tile_para->out_strides_, tile_para->multiples1_);
    }

    ret = ParallelLaunch(this->ms_context_, ScaleRunInt8, this, op_parameter_->thread_num_);
    // free memory malloced from memory pool
    if (!scale_param_->const_scale_) {
      ctx_->allocator->Free(input1_data_);
      input1_data_ = nullptr;
    }
    if (has_bias_ && !scale_param_->const_offset_) {
      ctx_->allocator->Free(input2_data_);
      input2_data_ = nullptr;
    }
    return ret;
  }

  // input1 has the same shape with input0 situation
  if (input1_data_ == nullptr) {
    input1_data_ = reinterpret_cast<int8_t *>(in_tensors_.at(1)->data());
  }
  if (has_bias_ && !scale_param_->const_offset_) {
    input2_data_ = reinterpret_cast<int8_t *>(in_tensors_.at(kOffsetIndex)->data());
  }
  ret = ParallelLaunch(this->ms_context_, ScaleRunInt8, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ScaleFusion, LiteKernelCreator<ScaleInt8CPUKernel>)
}  // namespace mindspore::kernel
