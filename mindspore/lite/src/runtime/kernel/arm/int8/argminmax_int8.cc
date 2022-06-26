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
#include "src/runtime/kernel/arm/int8/argminmax_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_ArgMaxFusion;
using mindspore::schema::PrimitiveType_ArgMinFusion;

namespace mindspore::kernel {
ArgMinMaxInt8CPUKernel::~ArgMinMaxInt8CPUKernel() {
  if (in_quant_arg_ != nullptr) {
    free(in_quant_arg_);
    in_quant_arg_ = nullptr;
  }
  if (out_quant_arg_ != nullptr) {
    free(out_quant_arg_);
    out_quant_arg_ = nullptr;
  }
}

int ArgMinMaxInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  param->data_type_ = kNumberTypeInt8;
  in_quant_arg_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (in_quant_arg_ == nullptr) {
    MS_LOG(ERROR) << "Malloc QuantArg for argmin or argmax int8 op failed!";
    return RET_ERROR;
  }
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  in_quant_arg_->scale_ = in_quant_args.front().scale;
  in_quant_arg_->zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  out_quant_arg_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  out_quant_arg_->scale_ = out_quant_args.front().scale;
  out_quant_arg_->zp_ = out_quant_args.front().zeroPoint;
  if (out_quant_arg_ == nullptr) {
    MS_LOG(ERROR) << "Malloc QuantArg for argmin or argmax int8 op failed!";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArgMinMaxInt8CPUKernel::ReSize() {
  auto in_shape = in_tensors_.at(0)->shape();
  auto dims_size = in_shape.size();
  CHECK_LESS_RETURN(in_shape.size(), 1);
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  int axis = param->axis_ < 0 ? param->axis_ + dims_size : param->axis_;
  param->axis_ = axis;
  param->dims_size_ = dims_size;
  if (param->topk_ <= 0) {
    MS_LOG(ERROR) << "Invalid topk " << param->topk_;
    return RET_ERROR;
  }
  param->topk_ = MSMIN(param->topk_, in_shape.at(axis));
  CHECK_NULL_RETURN(in_shape.data());
  ComputeStrides(in_shape.data(), param->in_strides_, in_shape.size());
  auto out_shape = out_tensors_.at(0)->shape();
  CHECK_NULL_RETURN(out_shape.data());
  ComputeStrides(out_shape.data(), param->out_strides_, out_shape.size());
  return RET_OK;
}

int ArgMinMaxInt8CPUKernel::Run() {
  auto input = in_tensors_.at(0);

  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_.at(0)->MutableData());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  int8_t *output_value = nullptr;
  if (out_tensors_.size() == C2NUM) {
    output_value = reinterpret_cast<int8_t *>(out_tensors_.at(C1NUM)->MallocData());
  }
  CHECK_NULL_RETURN(input_data);
  CHECK_NULL_RETURN(output_data);
  auto in_shape = input->shape();
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  CHECK_NULL_RETURN(in_shape.data());
  CHECK_NULL_RETURN(param);
  if (param->topk_ == 1) {
    Int8ArgMinMaxQuant(input_data, output_data, output_value, in_shape.data(), param, in_quant_arg_, out_quant_arg_);
    return RET_OK;
  }
  CHECK_NULL_RETURN(in_quant_arg_);
  CHECK_NULL_RETURN(out_quant_arg_);
  switch (param->axis_) {
    case 0:
      Int8ArgMinMaxDim0(input_data, output_data, output_value, in_shape.data(), param, in_quant_arg_, out_quant_arg_);
      break;
    case 1:
      Int8ArgMinMaxDim1(input_data, output_data, output_value, in_shape.data(), param, in_quant_arg_, out_quant_arg_);
      break;
    case 2:
      Int8ArgMinMaxDim2(input_data, output_data, output_value, in_shape.data(), param, in_quant_arg_, out_quant_arg_);
      break;
    case 3:
      Int8ArgMinMaxDim3(input_data, output_data, output_value, in_shape.data(), param, in_quant_arg_, out_quant_arg_);
      break;
    default:
      MS_LOG(ERROR) << "axis is invalid";
      return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ArgMaxFusion, LiteKernelCreator<ArgMinMaxInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ArgMinFusion, LiteKernelCreator<ArgMinMaxInt8CPUKernel>)
}  // namespace mindspore::kernel
