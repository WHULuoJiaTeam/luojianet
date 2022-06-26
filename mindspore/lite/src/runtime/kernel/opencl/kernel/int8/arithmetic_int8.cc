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

#include "src/runtime/kernel/opencl/kernel/int8/arithmetic_int8.h"
#include <vector>
#include "nnacl/fp32/common_func_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/int8/arithmetic.cl.inc"
#include "nnacl/arithmetic.h"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::lite::opencl::MemType;
using mindspore::schema::ActivationType_NO_ACTIVATION;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::EltwiseMode_MAXIMUM;
using mindspore::schema::EltwiseMode_PROD;
using mindspore::schema::EltwiseMode_SUM;
using mindspore::schema::PrimitiveType_AddFusion;
using mindspore::schema::PrimitiveType_BiasAdd;
using mindspore::schema::PrimitiveType_DivFusion;
using mindspore::schema::PrimitiveType_Eltwise;
using mindspore::schema::PrimitiveType_MulFusion;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
int ArithmeticInt8OpenCLKernel::CheckSpecs() {
  for (auto &tensor : in_tensors_) {
    if (tensor->data_type() != kNumberTypeInt8) {
      MS_LOG(WARNING) << "ArithmeticInt8OpenCLKernel only support int8 input";
      return RET_ERROR;
    }
  }
  for (auto &tensor : out_tensors_) {
    if (tensor->data_type() != kNumberTypeInt8) {
      MS_LOG(WARNING) << "ArithmeticInt8OpenCLKernel only support int8 output";
      return RET_ERROR;
    }
  }

  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<const ArithmeticParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  if (!IsArithmetic(type())) {
    MS_LOG(WARNING) << "UnSupported Operator: " << schema::EnumNamePrimitiveType(type());
    return RET_ERROR;
  }
  if (type() == schema::PrimitiveType_Eltwise) {
    auto mode = param->eltwise_mode_;
    if (mode != EltwiseMode_PROD && mode != EltwiseMode_SUM && mode != EltwiseMode_MAXIMUM) {
      MS_LOG(WARNING) << "Eltwise mode not support, mode:" << mode;
      return RET_ERROR;
    }
  }
  if (!(param->activation_type_ == ActivationType_NO_ACTIVATION || param->activation_type_ == ActivationType_RELU ||
        param->activation_type_ == ActivationType_RELU6)) {
    MS_LOG(WARNING) << "Unsupported activation type " << param->activation_type_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticInt8OpenCLKernel::SetGlobalLocal() {
  if (element_flag_) {
    global_size_ = {out_shape_.width, out_shape_.height};
  } else {
    global_size_ = {out_shape_.Slice, out_shape_.W, out_shape_.H * out_shape_.N};
  }
  AlignGlobalLocal(global_size_, {});

  return RET_OK;
}

int ArithmeticInt8OpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto fp16_enable = ocl_runtime_->GetFp16Enable();
  for (int i = 0; i < 2; ++i) {
    const auto &in_tensor = in_tensors_.at(i);
    GpuTensorInfo in_shape = GpuTensorInfo(in_tensor);
    if (in_tensor->IsConst()) {
      std::vector<char> weight(in_shape.Image2DSize, 0);
      bool src_is_fp16 = in_tensor->data_type() == kNumberTypeFloat16;
      PackNHWCToNHWC4(in_tensor->data(), weight.data(), src_is_fp16, fp16_enable, in_shape);
      size_t dtype = fp16_enable ? CL_HALF_FLOAT : CL_FLOAT;
      ImageSize img_size{in_shape.width, in_shape.height, dtype};
      auto weight_ptr_ = allocator->Malloc(img_size, weight.data());
      if (weight_ptr_ == nullptr) {
        MS_LOG(ERROR) << "Malloc failed.";
        return RET_ERROR;
      }
      weight_ptrs_.push_back(weight_ptr_);
    } else {
      weight_ptrs_.push_back(nullptr);
    }
  }
  return RET_OK;
}

int ArithmeticInt8OpenCLKernel::SetConstArgs() {
  int arg_idx = 3;
  if (!element_flag_) {
    cl_int4 in0_shape = {static_cast<int>(in0_shape_.N), static_cast<int>(in0_shape_.H), static_cast<int>(in0_shape_.W),
                         static_cast<int>(in0_shape_.Slice)};
    cl_int4 in1_shape = {static_cast<int>(in1_shape_.N), static_cast<int>(in1_shape_.H), static_cast<int>(in1_shape_.W),
                         static_cast<int>(in1_shape_.Slice)};
    cl_int4 out_shape = {static_cast<int>(out_shape_.N), static_cast<int>(out_shape_.H), static_cast<int>(out_shape_.W),
                         static_cast<int>(out_shape_.Slice)};
    int broadcastC_flag = 0;  // do not need broadcast in C4
    if (in0_shape_.C == 1 && in1_shape_.C != 1) {
      broadcastC_flag = 1;  // BroadCast C4 in input0
    } else if (in0_shape_.C != 1 && in1_shape_.C == 1) {
      broadcastC_flag = 2;  // BroadCast C4 in input1
    }
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in0_shape) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, in1_shape) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_shape) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, broadcastC_flag) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  } else {
    cl_int2 output_shape{static_cast<int>(global_range_[0]), static_cast<int>(global_range_[1])};
    if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, output_shape) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, activation_min_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, activation_max_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }

  // set quantization parameter.
  auto input0_quant_param = in_tensors_[0]->quant_params().front();
  auto input1_quant_param = in_tensors_[1]->quant_params().front();
  auto output_quant_param = out_tensors_[0]->quant_params().front();
  cl_float4 scale = {static_cast<float>(input0_quant_param.scale), static_cast<float>(input1_quant_param.scale),
                     static_cast<float>(output_quant_param.scale), 0.0};
  cl_char4 zero_point = {static_cast<int8_t>(input0_quant_param.zeroPoint),
                         static_cast<int8_t>(input1_quant_param.zeroPoint),
                         static_cast<int8_t>(output_quant_param.zeroPoint), 0};
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, scale) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // scale
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, zero_point) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }  // zero_point
  return RET_OK;
}

int ArithmeticInt8OpenCLKernel::Prepare() {
  in0_shape_ = GpuTensorInfo(in_tensors_[0]);
  in1_shape_ = GpuTensorInfo(in_tensors_[1]);
  out_shape_ = GpuTensorInfo(out_tensors_[0]);

  auto *param = reinterpret_cast<const ArithmeticParameter *>(op_parameter_);
  CHECK_NULL_RETURN(param);
  if (type() == PrimitiveType_BiasAdd) {
    const_cast<ArithmeticParameter *>(param)->broadcasting_ = true;
  }
  element_flag_ = !param->broadcasting_;
  kernel_name_ = param->broadcasting_ ? "BroadcastNHWC4" : "Element";
  switch (type()) {
    case PrimitiveType_MulFusion:
      kernel_name_ += "MulInt8";
      break;
    case PrimitiveType_AddFusion:
      kernel_name_ += "AddInt8";
      break;
    case PrimitiveType_SubFusion:
      kernel_name_ += "SubInt8";
      break;
    case PrimitiveType_DivFusion:
      kernel_name_ += "DivInt8";
      break;
    case PrimitiveType_Eltwise: {
      auto mode = param->eltwise_mode_;
      if (mode == EltwiseMode_PROD) {
        kernel_name_ += "MulInt8";
      } else if (mode == EltwiseMode_SUM) {
        kernel_name_ += "AddInt8";
      } else if (mode == EltwiseMode_MAXIMUM) {
        kernel_name_ += "MaximumInt8";
      }
      break;
    }
    default:
      kernel_name_ += schema::EnumNamePrimitiveType(type());
  }

  if (param->activation_type_ == ActivationType_RELU) {
    activation_min_ = 0.f;
  } else if (param->activation_type_ == ActivationType_RELU6) {
    activation_min_ = 0.f;
    activation_max_ = 6.f;
  }

  const std::string program_name = "Arithmetic";
  std::string source = arithmetic_source;
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  int error_code = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name_);
  if (error_code != RET_OK) {
    return error_code;
  }

  (void)SetGlobalLocal();
  // BiasAdd InitWeight will be called in opencl_subgraph prepare
  if (type() != PrimitiveType_BiasAdd) {
    InitWeights();
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << kernel_name_ << " Init Done!";
  return RET_OK;
}

int ArithmeticInt8OpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  auto input_0_ptr = weight_ptrs_[0] == nullptr ? in_tensors_[0]->data() : weight_ptrs_[0];
  auto input_1_ptr = weight_ptrs_[1] == nullptr ? in_tensors_[1]->data() : weight_ptrs_[1];
  int arg_idx = 0;

  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, input_0_ptr) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, input_1_ptr) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeInt8, PrimitiveType_AddFusion, OpenCLKernelCreator<ArithmeticInt8OpenCLKernel>)
}  // namespace mindspore::kernel
