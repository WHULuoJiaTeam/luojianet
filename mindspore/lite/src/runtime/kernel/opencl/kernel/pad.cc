/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <string>
#include <algorithm>
#include "src/common/utils.h"
#include "src/runtime/kernel/opencl/kernel/pad.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/opencl/cl/pad.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PaddingMode_CONSTANT;
using mindspore::schema::PrimitiveType_PadFusion;

namespace mindspore::kernel {
int PadOpenCLKernel::CheckSpecs() {
  auto param = reinterpret_cast<PadParameter *>(op_parameter_);
  MS_ASSERT(param);
  if (in_tensors_.size() != INPUT_TENSOR_SIZE_2) {
    MS_LOG(WARNING) << "Pad only support 1 input Tensor.";
    return RET_ERROR;
  }
  if (out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "Pad only support 1 output Tensor.";
    return RET_ERROR;
  }
  auto in_ndim = in_tensors_.front()->shape().size();
  if (in_ndim < DIMENSION_1D || in_ndim > DIMENSION_4D) {
    MS_LOG(WARNING) << "Pad only supports 1D-4D input Tensor but get " << in_ndim << "D.";
    return RET_ERROR;
  }
  auto out_ndim = in_tensors_.front()->shape().size();
  if (out_ndim < DIMENSION_1D || out_ndim > DIMENSION_4D) {
    MS_LOG(WARNING) << "Pad only supports 1D-4D output Tensor but get " << out_ndim << "D.";
    return RET_ERROR;
  }
  if (in_ndim != out_ndim) {
    MS_LOG(WARNING) << "Pad: input ndim != output ndim.";
    return RET_ERROR;
  }
  if (param->pad_mode_ != PaddingMode_CONSTANT) {
    MS_LOG(WARNING) << "Pad only support CONSTANT MODE.";
    return RET_ERROR;
  }
  // Compatibility code
  if (param->padding_length == static_cast<int>(DIMENSION_2D * in_ndim)) {
    return RET_OK;
  }
  auto pad_shape = in_tensors_.at(1)->shape();
  if (pad_shape.size() != DIMENSION_2D || pad_shape[0] != static_cast<int>(in_ndim) || pad_shape[1] != DIMENSION_2D) {
    MS_LOG(WARNING) << "pad tensor shape invalid.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadOpenCLKernel::Prepare() {
  const std::string source = pad_source;
  const std::string program_name = "Pad";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, "Pad", build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadOpenCLKernel::SetConstArgs() {
  auto input = GpuTensorInfo(in_tensors_.front());
  auto output = GpuTensorInfo(out_tensors_.front());
  cl_int4 input_shape = {static_cast<cl_int>(input.N), static_cast<cl_int>(input.H), static_cast<cl_int>(input.W),
                         static_cast<cl_int>(input.C)};
  cl_int4 output_shape = {static_cast<cl_int>(output.N), static_cast<cl_int>(output.H), static_cast<cl_int>(output.W),
                          static_cast<cl_int>(output.C)};
  cl_int2 io_slices = {static_cast<cl_int>(input.Slice), static_cast<cl_int>(output.Slice)};

  int ndim = in_tensors_.front()->shape().size();
  std::vector<int> pad_before_ori;
  pad_before_ori.reserve(ndim);
  auto paddings = reinterpret_cast<int32_t *>(in_tensors_.at(1)->data());
  for (auto i = 0; i < ndim; i++) {
    pad_before_ori.push_back(paddings[2 * i]);
  }
  cl_int4 pad_before;
  Broadcast2GpuShape(pad_before_ori.data(), ndim, pad_before.s, DIMENSION_4D, 0);

  int arg_cn = 2;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, input_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, output_shape) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, io_slices) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, pad_before) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn, param_->constant_value_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  local_size_ = {8, 4, 1};
  global_size_ = {output.N * output.H, output.W, output.Slice};
  AlignGlobalLocal(global_size_, local_size_);
  return RET_OK;
}

int PadOpenCLKernel::Run() {
  if (ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_.front()->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_PadFusion, OpenCLKernelCreator<PadOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_PadFusion, OpenCLKernelCreator<PadOpenCLKernel>)
}  // namespace mindspore::kernel
