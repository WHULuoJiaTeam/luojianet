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

#ifndef MINDSPORE_CCSRC_FL_SERVER_KERNEL_SGD_KERNEL_H_
#define MINDSPORE_CCSRC_FL_SERVER_KERNEL_SGD_KERNEL_H_

#include <vector>
#include <memory>
#include <utility>
#include "plugin/device/cpu/kernel/sgd_cpu_kernel.h"
#include "fl/server/kernel/optimizer_kernel.h"
#include "fl/server/kernel/optimizer_kernel_factory.h"

namespace mindspore {
namespace fl {
namespace server {
namespace kernel {
using mindspore::kernel::SGDCpuKernelMod;
template <typename T>
class SGDKernelMod : public SGDCpuKernelMod, public OptimizerKernelMod {
 public:
  SGDKernelMod() = default;
  ~SGDKernelMod() override = default;

  void InitKernel(const CNodePtr &cnode) override {
    SGDCpuKernelMod::InitKernel(cnode);
    InitServerKernelInputOutputSize(cnode);
    GenerateReuseKernelNodeInfo();
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return SGDCpuKernelMod::Launch(inputs, workspace, outputs);
  }

  void GenerateReuseKernelNodeInfo() override {
    constexpr int kWeightIndex = 0;
    constexpr int kLearningRateIndex = 2;
    constexpr int kAccumulationIndex = 3;
    constexpr int kMomentumIndex = 4;
    constexpr int kStatIndex = 5;
    MS_LOG(INFO) << "SGD reuse 'weight', 'learning rate', 'accumulation', 'momentum' and 'stat' of the kernel node.";
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kWeight, kWeightIndex));
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kLearningRate, kLearningRateIndex));
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kAccumulation, kAccumulationIndex));
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kMomentum, kMomentumIndex));
    reuse_kernel_node_inputs_info_.insert(std::make_pair(kStat, kStatIndex));
    return;
  }
};
}  // namespace kernel
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_KERNEL_SGD_KERNEL_H_
