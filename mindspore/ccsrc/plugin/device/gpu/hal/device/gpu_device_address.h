/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_ADDRESS_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_ADDRESS_H_

#include <string>
#include <vector>
#include "runtime/device/device_address.h"

using ShapeVecotr = std::vector<int>;

namespace mindspore {
#ifdef ENABLE_DEBUGGER
class Debugger;
#endif
namespace device {
namespace gpu {
class GPUDeviceAddress : public DeviceAddress {
 public:
  GPUDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}
  GPUDeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id)
      : DeviceAddress(ptr, size, format, type_id) {}
  GPUDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const KernelWithIndex &node_index)
      : DeviceAddress(ptr, size, format, type_id, node_index) {}
  GPUDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const std::string &device_name,
                   uint32_t device_id)
      : DeviceAddress(ptr, size, format, type_id, device_name, device_id) {}
  ~GPUDeviceAddress() override;

  bool SyncDeviceToHost(size_t size, void *host_ptr) const override;
  bool SyncHostToDevice(size_t size, const void *host_ptr) const override;
  bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const override;
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                        const std::string &format) const override;
  bool SyncDeviceToDevice(const DeviceSync *src_device_addr) const override;

  void ClearDeviceMemory() override;
  void set_status(DeviceAddressStatus status) { status_ = status; }
  DeviceAddressStatus status() const { return status_; }
  DeviceAddressType DeviceType() const override { return DeviceAddressType::kGPU; }

#ifdef ENABLE_DEBUGGER
  bool LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                     const ShapeVector &host_shape, TypeId host_type, size_t slot, bool keep_prev,
                     uint32_t root_graph_id, bool force_update) const override;
#endif

 private:
  DeviceAddressStatus status_{DeviceAddressStatus::kInDevice};
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_ADDRESS_H_
