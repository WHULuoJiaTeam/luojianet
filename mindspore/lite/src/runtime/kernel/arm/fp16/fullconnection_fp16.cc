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

#include "src/runtime/kernel/arm/fp16/fullconnection_fp16.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
int FullconnectionFP16CPUKernel::InitAShape() {
  auto a_shape = in_tensors_.at(0)->shape();
  MS_CHECK_TRUE_MSG(a_shape.size(), C2NUM, "fully-connection A-metrics' shape is invalid.");
  params_->row_ = a_shape[0];
  params_->deep_ = a_shape[1];
  return RET_OK;
}

int FullconnectionFP16CPUKernel::InitBShape() {
  auto b_shape = in_tensors_.at(1)->shape();
  MS_CHECK_TRUE_MSG(b_shape.size(), C2NUM, "fully-connection B-metrics' shape is invalid.");
  params_->col_ = b_shape[0];
  params_->deep_ = b_shape[1];
  return RET_OK;
}

int FullconnectionFP16CPUKernel::ReSize() {
  auto ret = InitAShape();
  MS_CHECK_TRUE_RET(ret == RET_OK, RET_ERROR);
  ret = InitBShape();
  MS_CHECK_TRUE_RET(ret == RET_OK, RET_ERROR);
  return MatmulBaseFP16CPUKernel::ReSize();
}

int FullconnectionFP16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
#ifdef ENABLE_ARM64
  row_tile_ = C16NUM;
#else
  row_tile_ = C12NUM;
#endif
  params_->batch = 1;
  a_batch_ = 1;
  b_batch_ = 1;
  a_offset_.resize(params_->batch, 0);
  b_offset_.resize(params_->batch, 0);
  params_->a_transpose_ = false;
  params_->b_transpose_ = true;
  auto ret = MatmulBaseFP16CPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Do fully-connection prepare failed.";
  }
  return ret;
}

int FullconnectionFP16CPUKernel::Run() {
  auto ret = MatmulBaseFP16CPUKernel::Run();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FullconnectionFP16CPUKernel run failed";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FullConnection, LiteKernelCreator<FullconnectionFP16CPUKernel>)
}  // namespace mindspore::kernel
