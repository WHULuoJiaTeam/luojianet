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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_MDTEMPERATURE_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_MDTEMPERATURE_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/common/mdtemperature_impl.cuh"

namespace luojianet_ms {
namespace kernel {
template <typename T, typename T1>
class MDTemperatureGpuKernelMod : public NativeGpuKernelMod {
 public:
  MDTemperatureGpuKernelMod() : ele_start(1) {}
  ~MDTemperatureGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    residue_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "residue_numbers"));

    auto shape_start = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_end = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_atom_vel = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_atom_mass = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);

    for (size_t i = 0; i < shape_start.size(); i++) ele_start *= shape_start[i];
    for (size_t i = 0; i < shape_end.size(); i++) ele_end *= shape_end[i];
    for (size_t i = 0; i < shape_atom_vel.size(); i++) ele_atom_vel *= shape_atom_vel[i];
    for (size_t i = 0; i < shape_atom_mass.size(); i++) ele_atom_mass *= shape_atom_mass[i];

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto start = GetDeviceAddress<const T1>(inputs, 0);
    auto end = GetDeviceAddress<const T1>(inputs, 1);
    auto atom_vel_f = GetDeviceAddress<const T>(inputs, 2);
    auto atom_mass = GetDeviceAddress<const T>(inputs, 3);

    auto ek = GetDeviceAddress<T>(outputs, 0);

    MDTemperature(residue_numbers, start, end, atom_vel_f, atom_mass, ek, reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_start * sizeof(T1));
    input_size_list_.push_back(ele_end * sizeof(T1));
    input_size_list_.push_back(ele_atom_vel * sizeof(T));
    input_size_list_.push_back(ele_atom_mass * sizeof(T));

    output_size_list_.push_back(residue_numbers * sizeof(T));
  }

 private:
  size_t ele_start = 1;
  size_t ele_end = 1;
  size_t ele_atom_vel = 1;
  size_t ele_atom_mass = 1;

  int residue_numbers;
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_MDTEMPERATURE_KERNEL_H_
