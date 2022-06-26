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

#include "plugin/device/cpu/kernel/layer_norm_cpu_kernel.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kLayerNormInputsNum = 3;
constexpr size_t kLayerNormOutputsNum = 3;
}  // namespace

void LayerNormCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  std::vector<size_t> x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  auto begin_norm_axis = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "begin_norm_axis");
  auto begin_params_axis = common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, "begin_params_axis");
  if (begin_norm_axis < 0) {
    begin_norm_axis += SizeToLong(x_shape.size());
  }
  if (begin_params_axis < 0) {
    begin_params_axis += SizeToLong(x_shape.size());
  }
  for (size_t i = 0; i < LongToSize(begin_norm_axis); i++) {
    block_num_ *= x_shape[i];
  }
  for (size_t i = LongToSize(begin_norm_axis); i < x_shape.size(); i++) {
    block_size_ *= x_shape[i];
  }
  for (size_t i = LongToSize(begin_params_axis); i < x_shape.size(); i++) {
    param_num_ *= x_shape[i];
  }
  if (block_num_ == 0 || block_size_ == 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of 'input_x' should be at least 1, but got "
                      << Vector2Str(x_shape);
  }
}

bool LayerNormCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kLayerNormInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kLayerNormOutputsNum, kernel_name_);
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'input_x' should be float16, float32, or float64, but got " << dtype_;
  }
  return true;
}

template <typename T>
void LayerNormCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &outputs) {
  size_t f_size = sizeof(T);
  if (inputs[1]->size != f_size * param_num_ || inputs[2]->size != f_size * param_num_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the product of gamma and beta's shape must be " << param_num_;
  }
  if (outputs[1]->size != f_size * block_num_ || outputs[2]->size != f_size * block_num_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the product of mean and var's shape must be " << block_num_;
  }
  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  auto gamma = reinterpret_cast<T *>(inputs[1]->addr);
  auto beta = reinterpret_cast<T *>(inputs[2]->addr);
  auto y = reinterpret_cast<T *>(outputs[0]->addr);
  auto mean = reinterpret_cast<T *>(outputs[1]->addr);
  auto var = reinterpret_cast<T *>(outputs[2]->addr);
  size_t thread_num = common::ThreadPool::GetInstance().GetSyncRunThreadNum();
  if (block_num_ < thread_num) {
    thread_num = block_num_;
  }
  std::vector<common::Task> tasks;
  tasks.reserve(thread_num);
  auto task = [this, &x, &gamma, &beta, &y, &mean, &var, thread_num](size_t start) {
    for (size_t c = 0; c < ceil(static_cast<double>(block_num_) / thread_num); ++c) {
      if (c * thread_num + start >= block_num_) {
        continue;
      }
      size_t i = c * thread_num + start;
      T sum = (T)0.0;
      T square_sum = (T)0.0;
      for (size_t j = i * block_size_; j < (i + 1) * block_size_; ++j) {
        sum += x[j];
        square_sum += x[j] * x[j];
      }
      T block_mean = sum / static_cast<T>(block_size_);
      T block_var = square_sum / static_cast<T>(block_size_) - block_mean * block_mean;
      for (size_t j = i * block_size_; j < (i + 1) * block_size_; ++j) {
        auto param_shift = j % param_num_;
        y[j] = (x[j] - block_mean) / (T)std::sqrt(static_cast<double>(block_var) + eps_) * gamma[param_shift] +
               beta[param_shift];
      }
      mean[i] = block_mean;
      var[i] = block_var;
    }
  };
  for (size_t i = 0; i < thread_num; ++i) {
    auto block = [&, i]() {
      task(i);
      return common::SUCCESS;
    };
    (void)tasks.emplace_back(block);
  }
  ParallelLaunch(tasks);
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, LayerNorm, LayerNormCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
