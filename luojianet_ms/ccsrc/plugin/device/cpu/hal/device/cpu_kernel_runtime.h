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
#ifndef LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_RUNTIME_H_
#define LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_RUNTIME_H_

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <set>
#include "runtime/device/kernel_runtime.h"
#include "backend/common/session/kernel_graph.h"
#include "backend/common/session/session_basic.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "utils/any.h"
#include "profiler/device/cpu/cpu_profiling.h"
namespace luojianet_ms {
namespace device {
namespace cpu {
class CPUKernelRuntime : public KernelRuntime {
 public:
  CPUKernelRuntime() = default;
  ~CPUKernelRuntime() override = default;

  bool Init();
  bool Run(const session::KernelGraph &graph, bool is_task_sink) override;
  void AssignKernelGraphAddress(session::KernelGraph *kernel_graph);
  void CreateOutputTensors(session::KernelGraph *kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                           VectorRef *outputs, std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node);
  void BindInputOutput(session::KernelGraph *kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                       VectorRef *outputs);
  void IncreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs);
  void DecreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs);
  bool GenDynamicKernel(const session::KernelGraph &graph) override { return true; }
  bool RunDynamicKernelAsync(const session::KernelGraph &graph) override { return true; }
  DeviceAddressType GetTargetDeviceAddressType() const override { return DeviceAddressType::kCPU; };

 protected:
  bool SyncStream() override { return true; };
  bool MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) override { return true; };
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                       TypeId type_id) const override;
  DeviceAddressPtr CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format, TypeId type_id,
                                       const KernelWithIndex &node_index) const override;

 private:
  tensor::TensorPtr CreateTensorForOutput(session::KernelGraph *kernel_graph, const CNodePtr &node, size_t index,
                                          std::set<DeviceAddressPtr> *bound_addresses);
  BaseRef GetOrCreateTensorForOutput(session::KernelGraph *kernel_graph,
                                     const session::KernelWithIndex &kernel_with_index,
                                     std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
                                     std::map<AnfNodePtr, tensor::TensorPtr> *input_param_tensor_map,
                                     std::set<DeviceAddressPtr> *bound_addresses);
  void BindInputTensorAddressPtr(const session::KernelGraph &graph, const std::vector<tensor::TensorPtr> &inputs);
  void BindOutputTensorAddressPtr(const VectorRef *outputs);
  void AssignValueNodeAddress(session::KernelGraph *kernel_graph);
  void AssignInputNodeAddress(const session::KernelGraph *kernel_graph);
  void AssignKernelOutputAddress(const session::KernelGraph *kernel_graph);
  void AddRuntimeAddress(DeviceAddress *address, std::vector<kernel::AddressPtr> *input_list);

  bool initialized_{false};
};
}  // namespace cpu
}  // namespace device
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_RUNTIME_DEVICE_CPU_CPU_KERNEL_RUNTIME_H_
