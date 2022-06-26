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

#include <set>
#include <string>
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/kernel/reshape.h"
#include "src/runtime/kernel/opencl/cl/reshape.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpandDims;
using mindspore::schema::PrimitiveType_Reshape;
using mindspore::schema::PrimitiveType_Squeeze;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::kernel {
int ReshapeOpenCLKernel::CheckSpecs() {
  if ((in_tensors_.size() != INPUT_TENSOR_SIZE_1 && in_tensors_.size() != INPUT_TENSOR_SIZE_2) ||
      out_tensors_.size() != OUTPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "Reshape input output size unsupported.";
    return RET_ERROR;
  }
  if (in_tensors_[0]->data_type() != kNumberTypeFloat32 && in_tensors_[0]->data_type() != kNumberTypeFloat16 &&
      in_tensors_[0]->data_type() != kNumberTypeInt32) {
    MS_LOG(WARNING) << "Unsupported data type " << in_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (in_tensors_[0]->shape().size() > DIMENSION_5D) {
    MS_LOG(WARNING) << "Reshape input size should in 0-5, actual: " << in_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  if (out_tensors_[0]->shape().size() > DIMENSION_5D) {
    MS_LOG(WARNING) << "Reshape output size should in 0-5, actual: " << out_tensors_[0]->shape().size();
    return RET_ERROR;
  }
  return RET_OK;
}

int ReshapeOpenCLKernel::SetConstArgs() {
  auto in = GpuTensorInfo(in_tensors_.front());
  auto out = GpuTensorInfo(out_tensors_.front());
  cl_int4 src_size = {cl_int(in.C), cl_int(in.W), cl_int(in.D * in.H), cl_int(in.N)};
  cl_int4 dst_size = {cl_int(out.width), cl_int(out.height), cl_int(out.C), cl_int(out.C * out.W)};

  int arg_idx = 2;
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, src_size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_idx++, dst_size) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReshapeOpenCLKernel::SetGlobalLocal() {
  auto out = GpuTensorInfo(out_tensors_.front());
  local_size_ = {};
  global_size_ = {out.width, out.height};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);

  return RET_OK;
}

int ReshapeOpenCLKernel::Prepare() {
  const std::string kernel_name = "reshape_NHWC4";
  std::string source = reshape_source;
  const std::string program_name = "reshape";
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }

  (void)SetGlobalLocal();
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  return RET_OK;
}

int ReshapeOpenCLKernel::Run() {
  MS_LOG(DEBUG) << this->name() << " Running!";
  if (ocl_runtime_->SetKernelArg(kernel_, 0, in_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->SetKernelArg(kernel_, 1, out_tensors_[0]->data()) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReshapeOpenCLKernel::PreProcess() {
  if (type() == PrimitiveType_Reshape && !InferShapeDone()) {
    auto shape_tensor = in_tensors_[1];
    if (!shape_tensor->IsConst()) {
      if (!ocl_runtime_->SyncCommandQueue()) {
        MS_LOG(ERROR) << "SyncCommandQueue failed.";
        return RET_ERROR;
      }
      if (shape_tensor->MutableData() == nullptr) {
        MS_LOG(ERROR) << "MutableData failed.";
        return RET_ERROR;
      }
    }
  }
  return OpenCLKernel::PreProcess();
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Reshape, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Reshape, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeInt32, PrimitiveType_Reshape, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Squeeze, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Squeeze, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeInt32, PrimitiveType_Squeeze, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Unsqueeze, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Unsqueeze, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeInt32, PrimitiveType_Unsqueeze, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_ExpandDims, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_ExpandDims, OpenCLKernelCreator<ReshapeOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeInt32, PrimitiveType_ExpandDims, OpenCLKernelCreator<ReshapeOpenCLKernel>)
}  // namespace mindspore::kernel
