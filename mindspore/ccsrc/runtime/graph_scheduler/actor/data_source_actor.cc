/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/data_source_actor.h"
#include "runtime/graph_scheduler/actor/kernel_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/recorder_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
void DataSourceActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() < device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  // Init output data.
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
    (void)output_data_.emplace_back(std::move(data));
  }
}

void DataSourceActor::FetchData(OpContext<DeviceTensor> *const context) {
  MS_LOG(INFO) << "Data source actor(" << GetAID().Name() << ") fetches data.";
  MS_EXCEPTION_IF_NULL(context);
  // Pop the data of last time.
  if (!buffers_.empty()) {
    buffers_.pop();
  }

  // Construct device tensors and fill to the buffers from member nodes.
  FillDataBuffer();
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Allocate memory for device tensors.
  SendMemoryAllocReq(context);
}

void DataSourceActor::UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                       const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(context);

  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }
  const auto &output_device_tensors = buffers_.front();

  auto position = FetchNodePosition(output_node);
  // Host data souruce actor uses the node position, device data source actor uses the output index.
  auto output_position = (position != 0) ? position : data_arrow->from_output_index_;
  if (output_position >= output_device_tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The output index is of range.");
  }
  output_data->data_ = output_device_tensors[output_position];
}

void DeviceQueueDataSourceActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  // Init output data.
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
    (void)output_data_.emplace_back(std::move(data));
  }

  // Init kernel launch info.
  MS_EXCEPTION_IF_NULL(kernel_info_);
  for (size_t i = 0; i < kernel_info_->output_address_list().size(); ++i) {
    (void)launch_info_.outputs_.emplace_back(std::make_shared<Address>());
  }
}

void DeviceQueueDataSourceActor::FillDataBuffer() {
  MS_EXCEPTION_IF_NULL(kernel_info_);
  // Construct device tensors.
  std::vector<DeviceTensor *> device_tensors;
  for (auto &device_tensor : kernel_info_->output_address_list()) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    (void)device_tensors.emplace_back(device_tensor.get());
  }

  buffers_.push(device_tensors);
}

void DeviceQueueDataSourceActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  auto &device_tensors = buffers_.back();
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &device_tensors, device_contexts_[0],
                        context, GetAID());
}

void DeviceQueueDataSourceActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  auto &device_tensors = buffers_.front();
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &device_tensors, device_contexts_[0],
                        context, GetAID());
}

void DeviceQueueDataSourceActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(data_kernel_);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Construct outputs of data kernel launching.
  auto &device_tensors = buffers_.back();
  if (launch_info_.outputs_.size() != device_tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The outputs number is not equal to the device tensors number.");
  }
  for (size_t i = 0; i < device_tensors.size(); ++i) {
    MS_EXCEPTION_IF_NULL(launch_info_.outputs_[i]);
    MS_EXCEPTION_IF_NULL(device_tensors[i]);
    launch_info_.outputs_[i]->addr = device_tensors[i]->GetMutablePtr();
    launch_info_.outputs_[i]->size = device_tensors[i]->GetSize();
  }

  // Copy data from device queue by data kernel launching.
  try {
    auto ret = device_contexts_[0]->LaunchKernel(data_kernel_, launch_info_.inputs_, launch_info_.workspaces_,
                                                 launch_info_.outputs_, common::AnfAlgo::IsDynamicShape(data_kernel_));
    if (!ret) {
      std::string error_info = "Launch kernel failed: " + data_kernel_->fullname_with_scope();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info = "Launch kernel exception: " + data_kernel_->fullname_with_scope();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
    return;
  }

  PostRun(context);
}

void DeviceQueueDataSourceActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  ActorDispatcher::Send(*debug_aid_, &DebugActor::Debug, data_kernel_, &launch_info_, device_contexts_[0], context,
                        &GetAID());
}

void DeviceQueueDataSourceActor::SendRecorderInfo(OpContext<DeviceTensor> *const context) const {
  if (recorder_aid_ != nullptr) {
    MS_EXCEPTION_IF_NULL(data_kernel_);
    ActorDispatcher::Send(*recorder_aid_, &RecorderActor::RecordInfo, data_kernel_->fullname_with_scope(),
                          &launch_info_, device_contexts_[0], context);
  }
}

void HostQueueDataSourceActor::FillDataBuffer() {
  // Construct device tensors.
  std::vector<DeviceTensor *> device_tensors;
  for (auto &data_node : data_nodes_) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(data_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);
    (void)device_tensors.emplace_back(device_address.get());
  }

  buffers_.push(device_tensors);
}

void HostQueueDataSourceActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  auto &device_tensors = buffers_.back();
  if (IsSameDeviceType()) {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &device_tensors,
                          device_contexts_[0], context, GetAID());
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateBatchMemory, &device_tensors,
                          &device_contexts_, context, GetAID());
  }
}

void HostQueueDataSourceActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  auto &device_tensors = buffers_.front();
  if (IsSameDeviceType()) {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &device_tensors, device_contexts_[0],
                          context, GetAID());
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeBatchMemory, &device_tensors, &device_contexts_,
                          context, GetAID());
  }
}

void HostQueueDataSourceActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Get host tensors from host queue and get device tensors from buffers.
  MS_EXCEPTION_IF_NULL(host_queue_);
  if (host_queue_->IsEmpty()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Host data queue is empty.");
  }
  auto &host_tensors = host_queue_->Pull();
  auto &device_tensors = buffers_.back();
  if (host_tensors.size() != device_tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context),
                                      "The length of host tensors is not equal to the length of device tensors.");
  }

  // Copy data from host tensor to device tensor.
  for (size_t i = 0; i < host_tensors.size(); ++i) {
    auto &host_tensor = host_tensors[i];
    auto &device_tensor = device_tensors[i];
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_EXCEPTION_IF_NULL(host_tensor);
    auto tensor_device_address = std::dynamic_pointer_cast<DeviceTensor>(host_tensor->device_address());
    // Sync data from host_tensor_device_address to device_tensor.
    if (tensor_device_address != nullptr) {
      if (tensor_device_address.get() == device_tensor) {
        continue;
      }
      if (!Copy(device_tensor, tensor_device_address.get())) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Copy data failed.");
      }
      continue;
    }

    // Sync data from host_tensor to device_tensor.
    if (!device_tensor->SyncHostToDevice(trans::GetRuntimePaddingShape(data_nodes_[i], 0),
                                         LongToSize(host_tensor->data().nbytes()), host_tensor->data_type(),
                                         host_tensor->data_c(), host_tensor->device_info().host_format_)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "SyncHostToDevice failed.");
    }
  }
  host_queue_->Pop();

  PostRun(context);
}

size_t HostQueueDataSourceActor::FetchNodePosition(const AnfNodePtr &data_node) const {
  MS_EXCEPTION_IF_NULL(data_node);
  const auto &iter = data_node_position_map_.find(data_node);
  if (iter == data_node_position_map_.end()) {
    MS_LOG(EXCEPTION) << "Data node: " << data_node->DebugString() << " is not exist.";
  }
  return iter->second;
}

AnfNodePtr HostQueueDataSourceActor::FetchNode(size_t node_position) const {
  if (node_position >= data_nodes_.size()) {
    MS_LOG(EXCEPTION) << "The position of node is out of range: " << node_position;
  }
  return data_nodes_[node_position];
}

bool HostQueueDataSourceActor::IsSameDeviceType() const {
  for (size_t i = 1; i < device_contexts_.size(); i++) {
    if (device_contexts_[i] != device_contexts_[0]) {
      return false;
    }
  }
  return true;
}
}  // namespace runtime
}  // namespace mindspore
