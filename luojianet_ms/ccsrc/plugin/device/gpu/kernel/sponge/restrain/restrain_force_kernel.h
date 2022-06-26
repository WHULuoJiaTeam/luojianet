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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_RESTRAIN_RESTRAIN_FORCE_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_RESTRAIN_RESTRAIN_FORCE_KERNEL_H_

#include "plugin/device/gpu/kernel/cuda_impl/sponge/restrain/restrain_force_impl.cuh"

#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

namespace luojianet_ms {
namespace kernel {
template <typename T, typename T1>
class RestrainForceGpuKernelMod : public NativeGpuKernelMod {
 public:
  RestrainForceGpuKernelMod() : ele_uint_crd(1) {}
  ~RestrainForceGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    restrain_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "restrain_numbers"));
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    factor = static_cast<int>(GetAttr<float>(kernel_node, "factor"));
    auto shape_restrain_list = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_uint_crd = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_uint_crd_ref = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_scaler = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);

    for (size_t i = 0; i < shape_uint_crd.size(); i++) ele_uint_crd *= shape_uint_crd[i];
    for (size_t i = 0; i < shape_scaler.size(); i++) ele_scaler *= shape_scaler[i];
    for (size_t i = 0; i < shape_restrain_list.size(); i++) ele_restrain_list *= shape_restrain_list[i];
    for (size_t i = 0; i < shape_uint_crd_ref.size(); i++) ele_uint_crd_ref *= shape_uint_crd_ref[i];

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto restrain_list = GetDeviceAddress<const T1>(inputs, 0);
    auto uint_crd_f = GetDeviceAddress<const T1>(inputs, 1);
    auto uint_crd_ref = GetDeviceAddress<const T1>(inputs, 2);
    auto scaler_f = GetDeviceAddress<T>(inputs, 3);

    auto frc_f = GetDeviceAddress<T>(outputs, 0);

    restrainforce(restrain_numbers, atom_numbers, restrain_list, uint_crd_f, uint_crd_ref, factor, scaler_f, frc_f,
                  reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_restrain_list * sizeof(T1));
    input_size_list_.push_back(ele_uint_crd * sizeof(T1));
    input_size_list_.push_back(ele_uint_crd_ref * sizeof(T1));
    input_size_list_.push_back(ele_scaler * sizeof(T));

    output_size_list_.push_back(atom_numbers * 3 * sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;
  size_t ele_scaler = 1;
  size_t ele_restrain_list = 1;
  size_t ele_uint_crd_ref = 1;

  int restrain_numbers;
  int atom_numbers;
  float factor;
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_RESTRAIN_RESTRAIN_FORCE_KERNEL_H_
