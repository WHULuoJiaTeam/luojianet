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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_GET_CENTER_OF_MASS_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_GET_CENTER_OF_MASS_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/common/get_center_of_mass_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class GetCenterOfMassGpuKernelMod : public NativeGpuKernelMod {
 public:
  GetCenterOfMassGpuKernelMod() : ele_start(1) {}
  ~GetCenterOfMassGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    residue_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "residue_numbers"));

    auto shape_start = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_end = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_crd = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_atom_mass = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_residue_mass_inverse = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);

    for (size_t i = 0; i < shape_start.size(); i++) ele_start *= shape_start[i];
    for (size_t i = 0; i < shape_end.size(); i++) ele_end *= shape_end[i];
    for (size_t i = 0; i < shape_crd.size(); i++) ele_crd *= shape_crd[i];
    for (size_t i = 0; i < shape_atom_mass.size(); i++) ele_atom_mass *= shape_atom_mass[i];
    for (size_t i = 0; i < shape_residue_mass_inverse.size(); i++)
      ele_residue_mass_inverse *= shape_residue_mass_inverse[i];

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto start = GetDeviceAddress<T1>(inputs, 0);
    auto end = GetDeviceAddress<T1>(inputs, 1);
    auto crd = GetDeviceAddress<T>(inputs, 2);
    auto atom_mass = GetDeviceAddress<T>(inputs, 3);
    auto residue_mass_inverse = GetDeviceAddress<T>(inputs, 4);

    auto center_of_mass = GetDeviceAddress<T>(outputs, 0);

    GetCenterOfMass(residue_numbers, start, end, crd, atom_mass, residue_mass_inverse, center_of_mass,
                    reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_start * sizeof(T1));
    input_size_list_.push_back(ele_end * sizeof(T1));
    input_size_list_.push_back(ele_crd * sizeof(T));
    input_size_list_.push_back(ele_atom_mass * sizeof(T));
    input_size_list_.push_back(ele_residue_mass_inverse * sizeof(T));
    output_size_list_.push_back(3 * sizeof(T) * residue_numbers);
  }

 private:
  size_t ele_start = 1;
  size_t ele_end = 1;
  size_t ele_crd = 1;
  size_t ele_atom_mass = 1;
  size_t ele_residue_mass_inverse = 1;

  int residue_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_GET_CENTER_OF_MASS_KERNEL_H_
