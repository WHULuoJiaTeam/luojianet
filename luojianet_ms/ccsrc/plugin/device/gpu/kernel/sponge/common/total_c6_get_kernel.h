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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_TOTAL_C6_GET_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_TOTAL_C6_GET_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/common/total_c6_get_impl.cuh"

namespace luojianet_ms {
namespace kernel {
template <typename T, typename T1>
class TotalC6GetGpuKernelMod : public NativeGpuKernelMod {
 public:
  TotalC6GetGpuKernelMod() : ele_atom_lj_type(1) {}
  ~TotalC6GetGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));

    auto shape_atom_lj_type = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_lj_b = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

    for (size_t i = 0; i < shape_atom_lj_type.size(); i++) ele_atom_lj_type *= shape_atom_lj_type[i];
    for (size_t i = 0; i < shape_lj_b.size(); i++) ele_lj_b *= shape_lj_b[i];

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto atom_lj_type = GetDeviceAddress<T1>(inputs, 0);
    auto lj_b = GetDeviceAddress<T>(inputs, 1);

    auto factor = GetDeviceAddress<T>(outputs, 0);

    total_c6_get(atom_numbers, atom_lj_type, lj_b, factor, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_atom_lj_type * sizeof(T1));
    input_size_list_.push_back(ele_lj_b * sizeof(T));

    output_size_list_.push_back(sizeof(T));
  }

 private:
  size_t ele_atom_lj_type = 1;
  size_t ele_lj_b = 1;

  int atom_numbers;
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_GETCENTER_KERNEL_H_
