/**
* Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
* Copyright 2021, 2022 Huawei Technologies Co., Ltd
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
#include "src/runtime/kernel/arm/fp16/ragged_range_fp16.h"
#include <vector>
#include "nnacl/fp16/ragged_range_fp16.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using luojianet_ms::kernel::KERNEL_ARCH;
using luojianet_ms::lite::KernelRegistrar;
using luojianet_ms::lite::RET_OK;
using luojianet_ms::schema::PrimitiveType_RaggedRange;

namespace luojianet_ms::kernel {
int RaggedRangeFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 3);
  CHECK_LESS_RETURN(out_tensors_.size(), 2);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int RaggedRangeFp16CPUKernel::ReSize() { return RET_OK; }

int RaggedRangeFp16CPUKernel::Run() {
  RaggedRangeFp16(
    static_cast<float16_t *>(in_tensors_.at(0)->data()), static_cast<float16_t *>(in_tensors_.at(1)->data()),
    static_cast<float16_t *>(in_tensors_.at(2)->data()), static_cast<int *>(out_tensors_.at(0)->data()),
    static_cast<float16_t *>(out_tensors_.at(1)->data()), reinterpret_cast<RaggedRangeParameter *>(op_parameter_));
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_RaggedRange, LiteKernelCreator<RaggedRangeFp16CPUKernel>)
}  // namespace luojianet_ms::kernel
