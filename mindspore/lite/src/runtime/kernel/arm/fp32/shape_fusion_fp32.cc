/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/shape_fusion_fp32.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/runtime/infer_manager.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ShapeFusionCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), kInputSize1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  return RET_OK;
}

int ShapeFusionCPUKernel::ReSize() { return KernelInferShape(in_tensors_, out_tensors_, op_parameter_); }

int ShapeFusionCPUKernel::Run() {
#ifndef DELEGATE_CLIP
  return KernelInferShape(in_tensors_, out_tensors_, op_parameter_);
#else
  return RET_OK;
#endif
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimType_Inner_ShapeFusion, LiteKernelCreator<ShapeFusionCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimType_Inner_ShapeFusion, LiteKernelCreator<ShapeFusionCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_ShapeFusion, LiteKernelCreator<ShapeFusionCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimType_Inner_ShapeFusion, LiteKernelCreator<ShapeFusionCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimType_Inner_ShapeFusion, LiteKernelCreator<ShapeFusionCPUKernel>)
}  // namespace mindspore::kernel
