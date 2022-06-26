/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/adam_delta_cpu_kernel.h"

#include <vector>
#include <string>
#include <memory>

#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/adam_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kSizeFloat32 = sizeof(float);
constexpr size_t kAdamDeltaInputsNum = 9;
constexpr size_t kAdamDeltaOutputsNum = 1;
constexpr size_t kMIndex = 0;
constexpr size_t kVIndex = 1;
constexpr size_t kBeta1PowIndex = 2;
constexpr size_t kBeta2PowIndex = 3;
constexpr size_t kLRIndex = 4;
constexpr size_t kBeta1Index = 5;
constexpr size_t kBeta2Index = 6;
constexpr size_t kEpsIndex = 7;
constexpr size_t kGradIndex = 8;
}  // namespace

template <typename T>
void AdamDeltaCpuKernelMod::LaunchAdamDelta(T *delta, T *m, T *v, float lr, float beta1, float beta2, float epsilon,
                                            const T *gradient, size_t size) {
  std::function<void(size_t, size_t)> task;
  if (dtype_ == kNumberTypeFloat32) {
    task = [this, delta, m, v, lr, beta1, beta2, epsilon, gradient](size_t start, size_t end) {
      (void)AdamDeltaFp32(delta, m, v, lr, beta1, beta2, epsilon, gradient, start, end, use_nesterov_);
    };
  } else {
    task = [this, delta, m, v, lr, beta1, beta2, epsilon, gradient](size_t start, size_t end) {
      for (size_t c1 = start; c1 < end; ++c1) {
        m[c1] *= beta1;
        m[c1] += (1 - beta1) * gradient[c1];
        v[c1] *= beta2;
        v[c1] += (1 - beta2) * gradient[c1] * gradient[c1];
        if (use_nesterov_) {
          delta[c1] = -lr * (m[c1] * beta1 + (1 - beta1) * gradient[c1]) / (std::sqrt(v[c1]) + epsilon);
        } else {
          delta[c1] = -lr * m[c1] / (std::sqrt(v[c1]) + epsilon);
        }
      }
    };
  }
  ParallelLaunchAutoSearch(task, size, this, &parallel_search_info_);
}

void AdamDeltaCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<size_t> delta_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
  std::vector<size_t> m_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kMIndex);
  std::vector<size_t> v_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kVIndex);
  std::vector<size_t> grad_shape = AnfAlgo::GetInputDeviceShape(kernel_node, kGradIndex);
  dtype_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (!IsSameShape(delta_shape, m_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'delta' should be same with the shape of 'm', but got the shape of 'delta': "
                      << Vector2Str(delta_shape) << " and 'm': " << Vector2Str(m_shape);
  }
  if (!IsSameShape(delta_shape, v_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'delta' should be same with the shape of 'v', but got the shape of 'delta': "
                      << Vector2Str(delta_shape) << " and 'v': " << Vector2Str(v_shape);
  }
  if (!IsSameShape(delta_shape, grad_shape)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the shape of 'delta' should be same with the shape of 'grad', "
                         "but got the shape of 'delta': "
                      << Vector2Str(delta_shape) << " and 'grad': " << Vector2Str(grad_shape);
  }
  if (delta_shape.empty()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'delta' should be at least 1-D, but got empty shape!";
  }
  elem_num_ = 1;
  for (size_t i = 0; i < delta_shape.size(); ++i) {
    elem_num_ *= delta_shape[i];
  }
  if (elem_num_ < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'delta' should be at least 1-D, but got empty shape!";
  }
  if (common::AnfAlgo::HasNodeAttr(USE_NESTEROV, kernel_node)) {
    use_nesterov_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "use_nesterov");
  }
}

void AdamDeltaCpuKernelMod::CheckParams(const std::vector<kernel::AddressPtr> &inputs,
                                        const std::vector<kernel::AddressPtr> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kAdamDeltaInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kAdamDeltaOutputsNum, kernel_name_);

  size_t elem_size = elem_num_ * kSizeFloat32;
  std::vector<size_t> expect_sizes = {elem_size,    elem_size,    kSizeFloat32, kSizeFloat32, kSizeFloat32,
                                      kSizeFloat32, kSizeFloat32, kSizeFloat32, elem_size};
  std::vector<std::string> input_names = {"m",     "v",     "beta1_power", "beta2_power", "lr",
                                          "beta1", "beta2", "epsilon",     "grad"};
  for (size_t i = 0; i < kAdamDeltaInputsNum; ++i) {
    if (inputs[i]->size != expect_sizes[i]) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of input '" << input_names[i]
                        << "' should be equal to " << expect_sizes[i] << ", but got address size: " << inputs[i]->size;
    }
  }
  if (outputs.size() < 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the size of outputs should be at least 1, but got "
                      << outputs.size();
  }
  if (outputs[0]->size != elem_size) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the address size of 'outputs[0]' should be equal to "
                      << elem_size << ", but got " << outputs[0]->size;
  }
}

bool AdamDeltaCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                   const std::vector<kernel::AddressPtr> &,
                                   const std::vector<kernel::AddressPtr> &outputs) {
  CheckParams(inputs, outputs);
  auto m = reinterpret_cast<float *>(inputs[kMIndex]->addr);
  auto v = reinterpret_cast<float *>(inputs[kVIndex]->addr);
  auto beta1_power = reinterpret_cast<float *>(inputs[kBeta1PowIndex]->addr)[0];
  if (beta1_power == 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'beta1_power' should not be 1.";
  }
  auto beta2_power = reinterpret_cast<float *>(inputs[kBeta2PowIndex]->addr)[0];
  auto lr = reinterpret_cast<float *>(inputs[kLRIndex]->addr)[0];
  auto beta1 = reinterpret_cast<float *>(inputs[kBeta1Index]->addr)[0];
  auto beta2 = reinterpret_cast<float *>(inputs[kBeta2Index]->addr)[0];
  auto epsilon = reinterpret_cast<float *>(inputs[kEpsIndex]->addr)[0];
  auto grad = reinterpret_cast<float *>(inputs[kGradIndex]->addr);
  auto delta = reinterpret_cast<float *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(m);
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(grad);
  MS_EXCEPTION_IF_NULL(delta);

  lr = lr * std::sqrt(1 - beta2_power) / (1 - beta1_power);
  // multithreading
  size_t lens = inputs[0]->size > 0 ? static_cast<size_t>(inputs[0]->size / sizeof(float)) : 1;
  LaunchAdamDelta<float>(delta, m, v, lr, beta1, beta2, epsilon, grad, lens);
  return true;
}

std::vector<KernelAttr> AdamDeltaCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, AdamNoUpdateParam, AdamDeltaCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
