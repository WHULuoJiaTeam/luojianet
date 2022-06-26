/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/opencl/kernel/arithmetic.h"
#include <vector>
#include <string>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/runtime/kernel/opencl/cl/arithmetic.cl.inc"
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
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_FloorDiv;
using mindspore::schema::PrimitiveType_FloorMod;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_LogicalAnd;
using mindspore::schema::PrimitiveType_LogicalOr;
using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_Minimum;
using mindspore::schema::PrimitiveType_MulFusion;
using mindspore::schema::PrimitiveType_NotEqual;
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
int ArithmeticOpenCLKernel::CheckSpecsWithoutShape() {
  for (auto &tensor : in_tensors_) {
    if (tensor->data_type() != kNumberTypeFloat32 && tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(WARNING) << "ArithmeticOpenCLKernel only support fp32/fp16 input";
      return RET_ERROR;
    }
  }
  for (auto &tensor : out_tensors_) {
    if (tensor->data_type() != kNumberTypeFloat32 && tensor->data_type() != kNumberTypeFloat16) {
      MS_LOG(WARNING) << "ArithmeticOpenCLKernel only support fp32/fp16 output";
      return RET_ERROR;
    }
  }

  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2 || out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  auto *param = reinterpret_cast<const ArithmeticParameter *>(op_parameter_);
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

int ArithmeticOpenCLKernel::CheckSpecs() { return RET_OK; }

int ArithmeticOpenCLKernel::SetGlobalLocal() {
  if (element_flag_) {
    global_size_ = {out_shape_->width, out_shape_->height};
  } else {
    global_size_ = {out_shape_->Slice, out_shape_->W, out_shape_->H * out_shape_->D * out_shape_->N};
  }
  AlignGlobalLocal(global_size_, {});
  return RET_OK;
}

void SwitchGpuTensorInfoNWDim(GpuTensorInfo *gpuTensorInfo) {
  auto tmp = gpuTensorInfo->N;
  gpuTensorInfo->N = gpuTensorInfo->W;
  gpuTensorInfo->W = tmp;

  gpuTensorInfo->width = gpuTensorInfo->W * gpuTensorInfo->Slice;
  gpuTensorInfo->height = gpuTensorInfo->H * gpuTensorInfo->D * gpuTensorInfo->N;
}

int ArithmeticOpenCLKernel::InitWeights() {
  auto allocator = ocl_runtime_->GetAllocator();
  auto fp16_enable = ocl_runtime_->GetFp16Enable();
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    const auto &in_tensor = in_tensors_.at(i);
    auto in_shape = GpuTensorInfo::CreateGpuTensorInfo(in_tensor);
    if (in1_shape_switch_flag_ && i == 1) {
      SwitchGpuTensorInfoNWDim(in_shape.get());
    }
    if (in_tensor->IsConst()) {
      std::vector<char> weight(in_shape->Image2DSize, 0);
      bool src_is_fp16 = in_tensor->data_type() == kNumberTypeFloat16;
      PackNHWCToNHWC4(in_tensor->data(), weight.data(), src_is_fp16, fp16_enable, *in_shape);
      size_t dtype = fp16_enable ? CL_HALF_FLOAT : CL_FLOAT;
      ImageSize img_size{in_shape->width, in_shape->height, dtype};
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

int ArithmeticOpenCLKernel::SetConstArgs() {
  int arg_idx = CLARGSINDEX3;
  if (!element_flag_) {
    cl_int4 in0_shape = {static_cast<int>(in0_shape_->N), static_cast<int>(in0_shape_->D * in0_shape_->H),
                         static_cast<int>(in0_shape_->W), static_cast<int>(in0_shape_->Slice)};
    cl_int4 in1_shape = {static_cast<int>(in1_shape_->N), static_cast<int>(in1_shape_->D * in1_shape_->H),
                         static_cast<int>(in1_shape_->W), static_cast<int>(in1_shape_->Slice)};
    cl_int4 out_shape = {static_cast<int>(out_shape_->N), static_cast<int>(out_shape_->D * out_shape_->H),
                         static_cast<int>(out_shape_->W), static_cast<int>(out_shape_->Slice)};
    int broadcastC_flag = 0;  // do not need broadcast in C4
    if (in0_shape_->C == 1 && in1_shape_->C != 1) {
      broadcastC_flag = 1;  // BroadCast C4 in input0
    } else if (in0_shape_->C != 1 && in1_shape_->C == 1) {
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
  return RET_OK;
}

int ArithmeticOpenCLKernel::InitGpuTensorInfoShape() {
  auto shape0 = in_tensors_.at(0)->shape();
  auto shape1 = in_tensors_.at(1)->shape();

  in0_shape_ = GpuTensorInfo::CreateGpuTensorInfo(in_tensors_.at(0));
  if (in0_shape_ == nullptr) {
    MS_LOG(ERROR) << "Create gpu tensor info failed.";
    return RET_ERROR;
  }
  in1_shape_ = GpuTensorInfo::CreateGpuTensorInfo(in_tensors_.at(1));
  if (in1_shape_ == nullptr) {
    MS_LOG(ERROR) << "Create gpu tensor info failed.";
    return RET_ERROR;
  }
  out_shape_ = GpuTensorInfo::CreateGpuTensorInfo(out_tensors_.at(0));
  if (out_shape_ == nullptr) {
    MS_LOG(ERROR) << "Create gpu tensor info failed.";
    return RET_ERROR;
  }
  if (shape0.size() == DIMENSION_4D && shape1.size() == DIMENSION_2D) {
    if (shape0.at(kNHWC_W) == shape1.at(kNHWC_N) && shape0.at(kNHWC_C) == shape1.at(kNHWC_H)) {
      SwitchGpuTensorInfoNWDim(in1_shape_.get());
      in1_shape_switch_flag_ = true;
    }
  }
  if (shape0.size() == DIMENSION_5D && shape1.size() == DIMENSION_2D) {
    if (shape0.at(kNDHWC_W) == shape1.at(kNHWC_N) && shape0.at(kNDHWC_C) == shape1.at(kNHWC_H)) {
      SwitchGpuTensorInfoNWDim(in1_shape_.get());
      in1_shape_switch_flag_ = true;
    }
  }
  return RET_OK;
}

int ArithmeticOpenCLKernel::Prepare() {
  auto ret = InitGpuTensorInfoShape();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init gpu tensor info shape failed.";
    return ret;
  }

  auto *param = reinterpret_cast<const ArithmeticParameter *>(op_parameter_);
  if (type() == PrimitiveType_BiasAdd) {
    const_cast<ArithmeticParameter *>(param)->broadcasting_ = true;
  }
  element_flag_ = !param->broadcasting_;
  kernel_name_ = param->broadcasting_ ? "BroadcastNHWC4" : "Element";
  switch (type()) {
    case PrimitiveType_MulFusion:
      kernel_name_ += "Mul";
      break;
    case PrimitiveType_AddFusion:
      kernel_name_ += "Add";
      break;
    case PrimitiveType_SubFusion:
      kernel_name_ += "Sub";
      break;
    case PrimitiveType_DivFusion:
      kernel_name_ += "Div";
      break;
    case PrimitiveType_Eltwise: {
      auto mode = param->eltwise_mode_;
      if (mode == EltwiseMode_PROD) {
        kernel_name_ += "Mul";
      } else if (mode == EltwiseMode_SUM) {
        kernel_name_ += "Add";
      } else if (mode == EltwiseMode_MAXIMUM) {
        kernel_name_ += "Maximum";
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
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  int error_code = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name_, build_options_ext);
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

int ArithmeticOpenCLKernel::Run() {
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

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_MulFusion, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_AddFusion, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SubFusion, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_DivFusion, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Maximum, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Minimum, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_FloorMod, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_SquaredDifference, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Equal, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_NotEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Less, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_LessEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Greater, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Eltwise, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_BiasAdd, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_MulFusion, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_AddFusion, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SubFusion, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_DivFusion, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LogicalAnd, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LogicalOr, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Maximum, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Minimum, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FloorDiv, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_FloorMod, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_SquaredDifference, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Equal, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_NotEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Less, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_LessEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Greater, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_GreaterEqual, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Eltwise, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_BiasAdd, OpenCLKernelCreator<ArithmeticOpenCLKernel>)
}  // namespace mindspore::kernel
