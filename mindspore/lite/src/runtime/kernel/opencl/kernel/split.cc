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

#include "src/runtime/kernel/opencl/kernel/split.h"
#include <cstring>
#include <string>
#include <algorithm>
#include "src/kernel_registry.h"
#include "src/runtime/kernel/opencl/cl/split.cl.inc"

using mindspore::kernel::KERNEL_ARCH::kGPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::opencl::ImageSize;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {
int SplitOpenCLKernel::RunAxis0() {
  auto allocator_ = ocl_runtime_->GetAllocator();
  auto src_data = in_tensors_[0]->data();
  CHECK_NULL_RETURN(src_data);
  cl::Image2D *in_image = allocator_->GetImage(src_data);
  if (in_image == nullptr) {
    MS_LOG(ERROR) << "RunAxis0 in_image can not be nullptr";
    return RET_ERROR;
  }
  auto src_area = cl::array<cl::size_type, 3U>{0, 0, 0};
  for (size_t i = 0; i < out_tensors_.size(); i++) {
    auto dst_data = out_tensors_[i]->data();
    CHECK_NULL_RETURN(dst_data);
    ImageSize img_size;
    if (allocator_->GetImageSize(dst_data, &img_size) != RET_OK) {
      MS_LOG(ERROR) << "GetImageSize failed.";
      return RET_ERROR;
    }
    auto dst_area = cl::array<cl::size_type, 3U>{0, 0, 0};
    auto region = cl::array<cl::size_type, 3U>{img_size.width, img_size.height, 1};
    cl::Image2D *out_image = allocator_->GetImage(dst_data);
    if (out_image == nullptr) {
      MS_LOG(ERROR) << "RunAxis0 out_image can not be nullptr";
      return RET_ERROR;
    }
    if (ocl_runtime_->GetDefaultCommandQueue()->enqueueCopyImage(*in_image, *out_image, src_area, dst_area, region) !=
        CL_SUCCESS) {
      MS_LOG(WARNING) << "enqueueCopyImage failed.";
    }
    src_area[1] += region[1];
  }
  return RET_OK;
}

int SplitOpenCLKernel::CheckSpecs() {
  auto param = reinterpret_cast<SplitParameter *>(this->op_parameter_);
  CHECK_NULL_RETURN(param);
  if ((out_tensors_.size() != OUTPUT_TENSOR_SIZE_2 ||
       (out_tensors_.size() != OUTPUT_TENSOR_SIZE_3 && param->split_dim_ == 0)) &&
      in_tensors_.size() != INPUT_TENSOR_SIZE_1) {
    MS_LOG(WARNING) << "in size: " << in_tensors_.size() << ", out size: " << out_tensors_.size();
    return RET_ERROR;
  }
  if (in_tensors_.at(0)->IsConst()) {
    MS_LOG(WARNING) << "in_tensors_ must be tensor";
    return RET_ERROR;
  }
  for (auto &out_tensor : out_tensors_) {
    if (out_tensor->IsConst()) {
      MS_LOG(WARNING) << "out_tensor must be tensor";
      return RET_ERROR;
    }
  }

  if (!(param->num_split_ == 2 || param->split_dim_ == 0)) {
    MS_LOG(WARNING) << "num_split_ only supported = 2 or split_dim_ = 0 yet";
    return RET_ERROR;
  }
  if (param->split_dim_ < 0 || param->split_dim_ > 3) {
    MS_LOG(WARNING) << "split_dim_ must between 0~3";
    return RET_ERROR;
  }
  if (param->split_sizes_ == nullptr) {
    MS_LOG(WARNING) << "split_sizes_ can not nullptr";
    return RET_ERROR;
  }
  if (param->num_split_ == 1 && param->split_sizes_[0] == 0) {
    MS_LOG(WARNING) << "param->split_sizes_[0] is zero.";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitOpenCLKernel::AlignSplitSizes(SplitParameter *param, const std::vector<int> &in_shape) {
  auto allocator = ocl_runtime_->GetAllocator();
  CHECK_LESS_RETURN(static_cast<int>(in_shape.size()), param->split_dim_ + 1);
  int shape_dim = in_shape.at(param->split_dim_);
  if (num_split_ == 1) {
    CHECK_LESS_RETURN(param->split_sizes_[0], 1);
    size_t num_split = UP_DIV(shape_dim, param->split_sizes_[0]);
    split_sizes_ = reinterpret_cast<int *>(allocator->Malloc(num_split * sizeof(int), lite::opencl::MemType::BUF));
    if (split_sizes_ == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    for (size_t i = 0; i < num_split - 1; ++i) {
      split_sizes_[i] = (i + 1) * param->split_sizes_[0];
    }
  } else {
    int sum = 0;
    split_sizes_ = reinterpret_cast<int *>(allocator->Malloc(num_split_ * sizeof(int), lite::opencl::MemType::BUF));
    if (split_sizes_ == nullptr) {
      MS_LOG(ERROR) << "Malloc failed.";
      return RET_ERROR;
    }
    for (size_t i = 0; i < num_split_ - 1; ++i) {
      sum += param->split_sizes_[i];
      split_sizes_[i] = sum;
    }
  }
  return RET_OK;
}

int SplitOpenCLKernel::Prepare() {
  auto param = reinterpret_cast<SplitParameter *>(this->op_parameter_);
  CHECK_NULL_RETURN(param);
  auto in_shape = in_tensors_.at(0)->shape();
  int increment_dim = C4NUM - in_shape.size();
  split_dim_ = param->split_dim_ == 0 ? param->split_dim_ : param->split_dim_ + increment_dim;
  num_split_ = param->num_split_;
  if (split_dim_ == 0) {
    return RET_OK;
  }
  for (size_t i = 0; i < out_tensors_.size(); ++i) {
    int length = out_tensors_[0]->shape().size();
    if (split_dim_ == 3) {
      if (out_tensors_[i]->shape()[length - 1] % C4NUM != 0) {
        Align_ = false;
      }
    }
  }
  if (AlignSplitSizes(param, in_shape) != RET_OK) {
    MS_LOG(ERROR) << "AlignSplitSizes failed.";
    return RET_ERROR;
  }
  std::string kernel_name = "split_out";
  kernel_name += std::to_string(num_split_);
  kernel_name += "_axis" + std::to_string(split_dim_);
  if (!Align_) {
    kernel_name += "_unalign";
  }
  MS_LOG(DEBUG) << "kernel_name=: " << kernel_name;
  std::string source = split_source;
  const std::string program_name = "split";
  if (!ocl_runtime_->LoadSource(program_name, source)) {
    MS_LOG(ERROR) << "Load source failed.";
    return RET_ERROR;
  }
  auto build_options_ext = CreateBuildOptionsExtByDType(this->registry_data_type_);

  auto ret = ocl_runtime_->BuildKernel(kernel_, program_name, kernel_name, build_options_ext);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build kernel failed.";
    return ret;
  }
  MS_LOG(DEBUG) << kernel_name << " Init Done!";
  if (SetConstArgs() != RET_OK) {
    MS_LOG(ERROR) << "SeConstArgs failed.";
    return RET_ERROR;
  }
  (void)SetGlobalLocal();
  return RET_OK;
}

int SplitOpenCLKernel::SetConstArgs() {
  int arg_cn = out_tensors_.size() + 2;
  cl_int4 shape = {};
  for (size_t i = 0; i < in_tensors_[0]->shape().size(); ++i) {
    shape.s[i] = in_tensors_[0]->shape()[i];
  }
  Broadcast2GpuShape(shape.s, out_tensors_[0]->shape().size(), in_shape_.s, DIMENSION_4D, 1);
  if (Align_) {
    in_shape_.s[3] = UP_DIV(in_shape_.s[3], C4NUM);
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_shape_) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < out_tensors_.size(); ++i) {
    cl_int4 temp = {};
    for (size_t j = 0; j < out_tensors_[i]->shape().size(); ++j) {
      temp.s[j] = out_tensors_[i]->shape()[j];
    }
    Broadcast2GpuShape(temp.s, out_tensors_[i]->shape().size(), out_shape_.s, DIMENSION_4D, 1);
    if (Align_) {
      out_shape_.s[3] = UP_DIV(out_shape_.s[3], C4NUM);
    }
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_shape_) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (!Align_) {
    GpuTensorInfo img_info(in_tensors_.at(0));
    size_t dtype = enable_fp16_ ? sizeof(cl_half) : sizeof(cl_float);
    stride_w = img_info.RowPitch() / dtype;
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, stride_w) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int SplitOpenCLKernel::SetGlobalLocal() {
  OH = in_shape_.s[0] * in_shape_.s[1];
  OW = in_shape_.s[2];
  if (Align_) {
    OC = in_shape_.s[3];
  }
  global_size_ = {OH, OW, OC};
  local_size_ = {1, 1, 1};
  OpenCLKernel::AlignGlobalLocal(global_size_, local_size_);
  return RET_OK;
}

int SplitOpenCLKernel::Run() {
  if (split_dim_ == 0) {
    int ret = RunAxis0();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "RunAxis0 failed.";
      return ret;
    }
    return RET_OK;
  }
  int arg_cn = 0;
  if (Align_) {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  } else {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, in_tensors_.at(0)->data(), true) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  for (size_t i = 0; i < out_tensors_.size(); ++i) {
    if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, out_tensors_.at(i)->data()) != CL_SUCCESS) {
      MS_LOG(ERROR) << "SetKernelArg failed.";
      return RET_ERROR;
    }
  }
  if (ocl_runtime_->SetKernelArg(kernel_, arg_cn++, split_sizes_, true) != CL_SUCCESS) {
    MS_LOG(ERROR) << "SetKernelArg failed.";
    return RET_ERROR;
  }
  if (ocl_runtime_->RunKernel(kernel_, global_range_, local_range_, nullptr, &event_) != RET_OK) {
    MS_LOG(ERROR) << "RunKernel failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kGPU, kNumberTypeFloat32, PrimitiveType_Split, OpenCLKernelCreator<SplitOpenCLKernel>)
REG_KERNEL(kGPU, kNumberTypeFloat16, PrimitiveType_Split, OpenCLKernelCreator<SplitOpenCLKernel>)
}  // namespace mindspore::kernel
