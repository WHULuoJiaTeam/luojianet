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

#include "src/runtime/kernel/arm/fp16/power_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PowFusion;

namespace mindspore::kernel {
int PowerFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  exp_tensor_ = in_tensors_[1];
  MSLITE_CHECK_PTR(exp_tensor_);
  if (exp_tensor_->IsConst()) {
    auto ret = GetExpData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "GetExpData is error in Init()!";
      return ret;
    }
  }
  return RET_OK;
}

int PowerFp16CPUKernel::ReSize() { return RET_OK; }

int PowerFp16CPUKernel::GetExpData() {
  exp_data_type_ = exp_tensor_->data_type();
  if (exp_data_type_ == kNumberTypeFloat || exp_data_type_ == kNumberTypeFloat32) {
    exp_data_ = reinterpret_cast<float16_t *>(malloc(exp_tensor_->ElementsNum() * sizeof(float16_t)));
    if (exp_data_ == nullptr) {
      MS_LOG(ERROR) << "exp_data_ is nullptr";
      return RET_NULL_PTR;
    }
    auto exp = reinterpret_cast<float *>(exp_tensor_->data());
    if (exp == nullptr) {
      MS_LOG(ERROR) << "exp is nullptr!";
      return RET_NULL_PTR;
    }
    for (int i = 0; i < exp_tensor_->ElementsNum(); ++i) {
      exp_data_[i] = (float16_t)(exp[i]);
    }
  } else {
    exp_data_ = reinterpret_cast<float16_t *>(exp_tensor_->data());
    if (exp_data_ == nullptr) {
      MS_LOG(ERROR) << "exp_data_ is nullptr";
      return RET_NULL_PTR;
    }
  }
  return RET_OK;
}

int PowerImplFp16(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<PowerFp16CPUKernel *>(cdata);
  CHECK_NULL_RETURN(kernel);
  auto ret = kernel->RunImpl(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerFp16Impl error: " << ret;
    return ret;
  }
  return RET_OK;
}

int PowerFp16CPUKernel::Run() {
  if (exp_data_ == nullptr) {
    auto ret = GetExpData();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "GetExpData is error in run!";
      return ret;
    }
  }
  auto ret = ParallelLaunch(this->ms_context_, PowerImplFp16, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PowerFp16CPUKernel error: " << ret;
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerFp16CPUKernel::RunImpl(int task_id) {
  auto x_addr = reinterpret_cast<float16_t *>(in_tensors_.at(0)->data());
  MSLITE_CHECK_PTR(x_addr);
  auto output_addr = reinterpret_cast<float16_t *>(out_tensors_.at(0)->data());
  MSLITE_CHECK_PTR(output_addr);
  auto size = in_tensors_.at(0)->ElementsNum();
  CHECK_NULL_RETURN(x_addr);
  CHECK_NULL_RETURN(output_addr);

  int stride = UP_DIV(size, thread_count_);
  int len = MSMIN(stride, size - stride * task_id);
  if (len <= 0) {
    return RET_OK;
  }
  bool broadcast = in_tensors_[0]->shape() == in_tensors_[1]->shape() ? false : true;
  float16_t *cur_exp = nullptr;
  if (broadcast) {
    cur_exp = exp_data_;
  } else {
    cur_exp = exp_data_ + stride * task_id;
  }
  CHECK_NULL_RETURN(cur_exp);
  auto error_code =
    PowerFp16(x_addr + stride * task_id, cur_exp, output_addr + stride * task_id, len, scale_, shift_, broadcast);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Power Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

PowerFp16CPUKernel::~PowerFp16CPUKernel() {
  if ((exp_data_type_ == kNumberTypeFloat || exp_data_type_ == kNumberTypeFloat32) && exp_data_ != nullptr) {
    free(exp_data_);
    exp_data_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_PowFusion, LiteKernelCreator<PowerFp16CPUKernel>)
}  // namespace mindspore::kernel
