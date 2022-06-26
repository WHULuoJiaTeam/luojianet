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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRIORITY_REPLAY_BUFFER_CPU_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRIORITY_REPLAY_BUFFER_CPU_KERNEL_H_
#include <stdlib.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/rl/priority_replay_buffer.h"

namespace luojianet_ms {
namespace kernel {
class PriorityReplayBufferCreateCpuKernel : public NativeCpuKernelMod {
 public:
  PriorityReplayBufferCreateCpuKernel() = default;
  ~PriorityReplayBufferCreateCpuKernel() override = default;

  // Collect and prepare kernel algorithm parameter.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {KernelAttr().AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferPushCpuKernel : public NativeCpuKernelMod {
 public:
  PriorityReplayBufferPushCpuKernel() = default;
  ~PriorityReplayBufferPushCpuKernel() override = default;

  // Init kernel from CNode.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;

 private:
  int64_t handle_{-1};
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferSampleCpuKernel : public NativeCpuKernelMod {
 public:
  PriorityReplayBufferSampleCpuKernel() = default;
  ~PriorityReplayBufferSampleCpuKernel() override = default;

  // Init kernel from CNode.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;

 private:
  int64_t handle_{-1};
  size_t batch_size_{0};
  std::vector<size_t> schema_;
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer_{nullptr};
};

class PriorityReplayBufferUpdateCpuKernel : public NativeCpuKernelMod {
 public:
  PriorityReplayBufferUpdateCpuKernel() = default;
  ~PriorityReplayBufferUpdateCpuKernel() override = default;

  // Init kernel from CNode.
  void InitKernel(const CNodePtr &kernel_node);

  // Execute kernel.
  bool Launch(const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> support_list = {
      KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  int64_t handle_{-1};
  std::vector<size_t> indices_shape_;
  std::vector<size_t> priorities_shape_;
  std::shared_ptr<PriorityReplayBuffer> prioriory_replay_buffer_{nullptr};
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRIORITY_REPLAY_BUFFER_CPU_KERNEL_H_
