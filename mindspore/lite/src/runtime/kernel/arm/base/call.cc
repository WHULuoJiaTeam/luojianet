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

#include "src/runtime/kernel/arm/base/call.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#ifndef CONTROLFLOW_TENSORLIST_CLIP
#include "src/tensorlist.h"
#endif
#include "src/common/utils.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Call;

// this file is useless when move create actor before schedule.
namespace mindspore::kernel {
int CallCPUKernel::Prepare() { return RET_OK; }
int CallCPUKernel::ReSize() { return RET_OK; }
int CallCPUKernel::Run() { return RET_OK; }

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Call, LiteKernelCreator<CallCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Call, LiteKernelCreator<CallCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_Call, LiteKernelCreator<CallCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Call, LiteKernelCreator<CallCPUKernel>)
}  // namespace mindspore::kernel
