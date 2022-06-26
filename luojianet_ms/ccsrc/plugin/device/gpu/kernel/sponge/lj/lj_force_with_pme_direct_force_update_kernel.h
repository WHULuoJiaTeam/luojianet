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
/**
 * Note:
 *  LJForceWithPMEDirectForceUpdate. This is an experimental interface that is subject to change and/or deletion.
 */
#ifndef LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_LJ_LJ_FORCE_WITH_PME_DIRECT_FORCE_KERNEL_H_
#define LUOJIANET_MS_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_LJ_LJ_FORCE_WITH_PME_DIRECT_FORCE_KERNEL_H_
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/lj/lj_force_with_pme_direct_force_impl.cuh"
namespace luojianet_ms {
namespace kernel {
template <typename T, typename T1>
class LJForceWithPMEDirectForceUpdateGpuKernelMod : public NativeGpuKernelMod {
 public:
  LJForceWithPMEDirectForceUpdateGpuKernelMod() : ele_uint_crd(1) {}
  ~LJForceWithPMEDirectForceUpdateGpuKernelMod() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    cutoff = static_cast<float>(GetAttr<float_t>(kernel_node, "cutoff"));
    pme_beta = static_cast<float>(GetAttr<float_t>(kernel_node, "pme_beta"));
    need_update = static_cast<int>(GetAttr<int64_t>(kernel_node, "need_update"));

    auto shape_uint_crd = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_LJtype = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_charge = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_scaler = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_nl_numbers = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    auto shape_nl_serial = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);
    auto shape_d_LJ_a = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 6);
    auto shape_d_LJ_b = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 7);

    for (size_t i = 0; i < shape_uint_crd.size(); i++) ele_uint_crd *= shape_uint_crd[i];
    for (size_t i = 0; i < shape_LJtype.size(); i++) ele_LJtype *= shape_LJtype[i];
    for (size_t i = 0; i < shape_charge.size(); i++) ele_charge *= shape_charge[i];
    for (size_t i = 0; i < shape_scaler.size(); i++) ele_scaler *= shape_scaler[i];
    for (size_t i = 0; i < shape_d_LJ_a.size(); i++) ele_d_LJ_a *= shape_d_LJ_a[i];
    for (size_t i = 0; i < shape_d_LJ_b.size(); i++) ele_d_LJ_b *= shape_d_LJ_b[i];

    InitSizeLists();
    return true;
  }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto uint_crd = GetDeviceAddress<T1>(inputs, 0);
    auto LJtype = GetDeviceAddress<T1>(inputs, 1);
    auto charge = GetDeviceAddress<T>(inputs, 2);
    auto scaler = GetDeviceAddress<T>(inputs, 3);
    auto nl_numbers = GetDeviceAddress<T1>(inputs, 4);
    auto nl_serial = GetDeviceAddress<T1>(inputs, 5);
    auto d_LJ_a = GetDeviceAddress<T>(inputs, 6);
    auto d_LJ_b = GetDeviceAddress<T>(inputs, 7);
    auto d_beta = GetDeviceAddress<T>(inputs, 8);

    if (need_update) {
      cudaMemcpyAsync(&pme_beta, d_beta, sizeof(float), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    auto uint_crd_with_LJ = GetDeviceAddress<T>(workspace, 0);
    auto nl = GetDeviceAddress<T1>(workspace, 1);

    auto frc = GetDeviceAddress<T>(outputs, 0);
    LJForceWithPMEDirectForce(atom_numbers, cutoff, pme_beta, uint_crd, LJtype, charge, scaler, uint_crd_with_LJ,
                              nl_numbers, nl_serial, nl, d_LJ_a, d_LJ_b, frc,
                              reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_uint_crd * sizeof(T1));
    input_size_list_.push_back(ele_LJtype * sizeof(T1));
    input_size_list_.push_back(ele_charge * sizeof(T));
    input_size_list_.push_back(ele_scaler * sizeof(T));
    input_size_list_.push_back(atom_numbers * sizeof(T1));
    input_size_list_.push_back(max_nl_numbers * sizeof(T1));
    input_size_list_.push_back(ele_d_LJ_a * sizeof(T));
    input_size_list_.push_back(ele_d_LJ_b * sizeof(T));
    input_size_list_.push_back(sizeof(T));

    workspace_size_list_.push_back(atom_numbers * max_nl_numbers * sizeof(T1));
    workspace_size_list_.push_back(atom_numbers * sizeof(UINT_VECTOR_LJ_TYPE));

    output_size_list_.push_back(atom_numbers * 3 * sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;
  size_t ele_LJtype = 1;
  size_t ele_charge = 1;
  size_t ele_scaler = 1;
  size_t ele_nl = 1;
  size_t ele_d_LJ_a = 1;
  size_t ele_d_LJ_b = 1;

  int atom_numbers;
  float pme_beta;
  float cutoff;
  int need_update;
  int max_nl_numbers = 800;
  struct UINT_VECTOR_LJ_TYPE {
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
    int LJ_type;
    float charge;
  };
  struct NEIGHBOR_LIST {
    int atom_numbers;
    int *atom_serial;
  };
  struct VECTOR {
    float x;
    float y;
    float z;
  };
};
}  // namespace kernel
}  // namespace luojianet_ms
#endif
