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

#ifndef LUOJIANET_MS_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PERLAYER_GRAD_GPUKERNEL_H_
#define LUOJIANET_MS_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PERLAYER_GRAD_GPUKERNEL_H_

#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace luojianet_ms {
namespace kernel {
class FakeLearnedScaleQuantPerLayerGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  FakeLearnedScaleQuantPerLayerGradGpuKernelMod();
  ~FakeLearnedScaleQuantPerLayerGradGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const CNodePtr &kernel_node) override;

 protected:
  void InitSizeLists() override;

 private:
  size_t input_size_;
  size_t workspace_size_;

  int quant_num_;
  int quant_delay_;
  int global_step_;
  bool neg_trunc_;
};
}  // namespace kernel
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_KERNEL_GPU_FAKE_LEARNED_SCALE_QUANT_PERLAYER_GRAD_GPUKERNEL_H_
