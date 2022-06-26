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

#include "src/runtime/kernel/arm/fp32_grad/adam.h"
#include <cmath>
#include <string>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"

using luojianet_ms::kernel::KERNEL_ARCH;
using luojianet_ms::lite::KernelRegistrar;
using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_OK;
using luojianet_ms::schema::PrimitiveType_Adam;

namespace luojianet_ms::kernel {
constexpr static int kWeightIdx = 0;
constexpr static int kMomentVector1stIdx = 1;
constexpr static int kMomentVector2stIdx = 2;
constexpr static int kBeta1PowerIdx = 3;
constexpr static int kBeta2PowerIdx = 4;
constexpr static int kBeta1Idx = 6;
constexpr static int kBeta2Idx = 7;
constexpr static int kEpsilonIdx = 8;
constexpr static int kGradientIdx = 9;

int AdamCPUKernel::ReSize() { return RET_OK; }

static int DoAdam(float *m, float *v, const float *gradient, float *weight, float beta1, float beta2, float beta1_power,
                  float beta2_power, float eps, float learning_rate, bool nesterov, int start, int end) {
  if ((1.f - beta1_power) <= 0.0f) {
    MS_LOG(ERROR) << "divisor cannot be 0 or below";
    return RET_ERROR;
  }
  if ((1.f - beta2_power) < 0.0f) {
    MS_LOG(ERROR) << "sqrt cannot be negative";
    return RET_ERROR;
  }

  auto update_lr = learning_rate * std::sqrt(1.f - beta2_power) / (1.f - beta1_power);
  const float one_minus_beta1 = 1.f - beta1;
  const float one_minus_beta2 = 1.f - beta2;
  if (nesterov) {  // Nadam
    for (int i = start; i < end; ++i) {
      m[i] += (gradient[i] - m[i]) * one_minus_beta1;
      v[i] += (gradient[i] * gradient[i] - v[i]) * one_minus_beta2;
      weight[i] -= update_lr * (m[i] * beta1 + one_minus_beta1 * gradient[i]) / (std::sqrt(v[i]) + eps);
    }
  } else {
    for (int i = start; i < end; ++i) {
      m[i] += (gradient[i] - m[i]) * one_minus_beta1;
      v[i] += (gradient[i] * gradient[i] - v[i]) * one_minus_beta2;
      weight[i] -= update_lr * m[i] / (std::sqrt(v[i]) + eps);
    }
  }
  return RET_OK;
}

int AdamCPUKernel::DoExecute(int task_id) {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_10D);
  auto weight = reinterpret_cast<float *>(in_tensors_.at(kWeightIdx)->MutableData());
  auto m = reinterpret_cast<float *>(in_tensors_.at(kMomentVector1stIdx)->MutableData());
  auto v = reinterpret_cast<float *>(in_tensors_.at(kMomentVector2stIdx)->MutableData());
  auto beta1_power = reinterpret_cast<float *>(in_tensors_.at(kBeta1PowerIdx)->MutableData())[0];
  auto beta2_power = reinterpret_cast<float *>(in_tensors_.at(kBeta2PowerIdx)->MutableData())[0];
  auto learning_rate = lr_;
  auto beta1 = reinterpret_cast<float *>(in_tensors_.at(kBeta1Idx)->MutableData())[0];
  auto beta2 = reinterpret_cast<float *>(in_tensors_.at(kBeta2Idx)->MutableData())[0];
  auto eps = reinterpret_cast<float *>(in_tensors_.at(kEpsilonIdx)->MutableData())[0];
  auto gradient = reinterpret_cast<float *>(in_tensors_.at(kGradientIdx)->MutableData());
  int length = in_tensors_.at(kWeightIdx)->ElementsNum();
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(m);
  CHECK_NULL_RETURN(v);
  CHECK_NULL_RETURN(gradient);

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  int start = stride * task_id;
  int end = start + count;

  return DoAdam(m, v, gradient, weight, beta1, beta2, beta1_power, beta2_power, eps, learning_rate,
                adam_param_->use_nesterov_, start, end);
}

int AdamRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto adam_kernel = reinterpret_cast<AdamCPUKernel *>(cdata);
  CHECK_NULL_RETURN(adam_kernel);
  auto error_code = RET_OK;
  if (adam_kernel->get_optimizer_mode() == WeightUpdateMode::VIRTUAL_BATCH) {
    error_code = adam_kernel->ExecuteVirtualBatch(task_id);
  } else if (adam_kernel->get_optimizer_mode() == WeightUpdateMode::ACCUMULATE_GRADS) {
    error_code = adam_kernel->ExecuteVirtualBatch(task_id);
  } else {
    error_code = adam_kernel->DoExecute(task_id);
  }

  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Adam run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AdamCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, AdamRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Adam function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AdamCPUKernel::Prepare() {
  CHECK_NULL_RETURN(adam_param_);
  auto ret = OptimizerKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Failed to initialize Adam Kernel";
    return RET_ERROR;
  }
  return RET_OK;
}

std::vector<int> AdamCPUKernel::GetOptimizerParamsIdxs() const {
  std::vector<int> indices = {6, 7, 3, 4, 8};
  return indices;
}

int AdamCPUKernel::OptimizerStep() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_10D - 1);
  auto weight = reinterpret_cast<float *>(in_tensors_.at(kWeightIdx)->MutableData());
  auto m = reinterpret_cast<float *>(in_tensors_.at(kMomentVector1stIdx)->MutableData());
  auto v = reinterpret_cast<float *>(in_tensors_.at(kMomentVector2stIdx)->MutableData());
  auto beta1_power = reinterpret_cast<float *>(in_tensors_.at(kBeta1PowerIdx)->MutableData())[0];
  auto beta2_power = reinterpret_cast<float *>(in_tensors_.at(kBeta2PowerIdx)->MutableData())[0];
  auto learning_rate = lr_;
  auto beta1 = reinterpret_cast<float *>(in_tensors_.at(kBeta1Idx)->MutableData())[0];
  auto beta2 = reinterpret_cast<float *>(in_tensors_.at(kBeta2Idx)->MutableData())[0];
  auto eps = reinterpret_cast<float *>(in_tensors_.at(kEpsilonIdx)->MutableData())[0];
  size_t length = in_tensors_.at(kWeightIdx)->ElementsNum();
  CHECK_NULL_RETURN(weight);
  CHECK_NULL_RETURN(m);
  CHECK_NULL_RETURN(v);

  int ret = RET_OK;
  if (grad_sum_ != nullptr && valid_grad_sum_) {
    size_t start = 0;
    size_t end = length;
    ret = DoAdam(m, v, grad_sum_, weight, beta1, beta2, beta1_power, beta2_power, eps, learning_rate,
                 adam_param_->use_nesterov_, start, end);
    std::fill(grad_sum_, grad_sum_ + length, 0);
    OptimizerKernel::OptimizerStep();
  }
  return ret;
}

kernel::InnerKernel *CpuAdamFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                              const lite::Context *ctx, const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_MSG(opParameter != nullptr, nullptr, "Op parameter is nullptr.");
  MS_ASSERT(desc.type == schema::PrimitiveType_Adam);
  auto *kernel =
    new (std::nothrow) AdamCPUKernel(opParameter, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new AdamCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Adam, CpuAdamFp32KernelCreator)
}  // namespace luojianet_ms::kernel
