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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ENV_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ENV_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/gpu/kernel/rl/environment_factory.h"

namespace luojianet_ms {
namespace kernel {
// Class for reinforcement learning environment creation.
// It create environment instance base on name and parameters in operator attribution,
// and result environment instance handle. The environment instance and handle will cache in
// EnvironmentFactory. It is notice that repeate calls launch() will returns the same
// handle created before.
class EnvCreateKernelMod : public NativeGpuKernelMod {
 public:
  EnvCreateKernelMod() = default;
  ~EnvCreateKernelMod() = default;

  bool Init(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  void InitSizeLists() override;

 private:
  int64_t handle_ = kInvalidHandle;
  std::shared_ptr<Environment> env_ = nullptr;
};

// Class for reinforcement environment reset.
// It reset environment state (for example agent state, timestep etc.) and result initial observations.
// The environment instance should already created with `EnvCreateKernelMod`.
class EnvResetKernelMod : public NativeGpuKernelMod {
 public:
  EnvResetKernelMod() = default;
  ~EnvResetKernelMod() = default;

  bool Init(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  void InitSizeLists() override;

 private:
  int64_t handle_ = kInvalidHandle;
  std::shared_ptr<Environment> env_ = nullptr;
};

// Class for environment step.
// It execute one time step and result observation, reward and done flag.
// The environment instance should already created with `EnvCreateKernelMod`.
class EnvStepKernelMod : public NativeGpuKernelMod {
 public:
  EnvStepKernelMod() = default;
  ~EnvStepKernelMod() = default;

  bool Init(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  void InitSizeLists() override;

 private:
  int64_t handle_ = kInvalidHandle;
  std::shared_ptr<Environment> env_ = nullptr;
};

MS_REG_GPU_KERNEL(EnvCreate, EnvCreateKernelMod)
MS_REG_GPU_KERNEL(EnvReset, EnvResetKernelMod)
MS_REG_GPU_KERNEL(EnvStep, EnvStepKernelMod)
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ENV_KERNEL_H_
