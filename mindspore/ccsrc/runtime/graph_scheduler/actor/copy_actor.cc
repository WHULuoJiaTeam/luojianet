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

#include "runtime/graph_scheduler/actor/copy_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "mindrt/include/async/async.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace runtime {
const size_t kInputDeviceContextIndex = 0;
const size_t kOutputDeviceContextIndex = 1;

void CopyActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumTwo) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  const size_t kDeviceTensorNum = 1;
  input_device_tensor_.resize(kDeviceTensorNum);
  output_device_tensor_.resize(kDeviceTensorNum);

  // Init output data.
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) != 0) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }
    auto data = std::make_unique<OpData<DeviceTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
    (void)output_data_.emplace_back(std::move(data));
  }
}

void CopyActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  FetchDeviceTensor(context);
  SendMemoryAllocReq(context);
}

void CopyActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &output_device_tensor_,
                        device_contexts_[kOutputDeviceContextIndex], context, GetAID());
}

void CopyActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_device_tensor_,
                        device_contexts_[kInputDeviceContextIndex], context, GetAID());
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &output_device_tensor_,
                        device_contexts_[kOutputDeviceContextIndex], context, GetAID());
}

void CopyActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(output_device_tensor_[0]);
  MS_EXCEPTION_IF_NULL(input_device_tensor_[0]);

  if (input_device_tensor_[0]->GetSize() != output_device_tensor_[0]->GetSize()) {
    MS_LOG(WARNING) << GetAID().Name() << " copy size is not equal, input size:" << input_device_tensor_[0]->GetSize()
                    << ", output size:" << output_device_tensor_[0]->GetSize();
  }

  if (!Copy(output_device_tensor_[0], input_device_tensor_[0])) {
    std::string error_info = "Copy device tensor failed: " + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  PostRun(context);
}

void CopyActor::FetchDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &input_device_context = device_contexts_[kInputDeviceContextIndex];
  const auto &output_device_context = device_contexts_[kOutputDeviceContextIndex];
  MS_EXCEPTION_IF_NULL(input_device_context);
  MS_EXCEPTION_IF_NULL(output_device_context);

  if (device_tensor_store_keys_.size() > 0) {
    const auto &device_tensor_store_node = device_tensor_store_keys_[0].second;
    MS_EXCEPTION_IF_NULL(device_tensor_store_node);
    input_device_tensor_[0] = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_node.get(),
                                                                     input_device_context->GetDeviceAddressType());
    if (input_device_tensor_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_node->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(input_device_context->GetDeviceAddressType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    output_device_tensor_[0] = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_node.get(),
                                                                      output_device_context->GetDeviceAddressType());
    if (output_device_tensor_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_node->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(output_device_context->GetDeviceAddressType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  } else {
    const auto &data_iter = input_op_datas_.find(context->sequential_num_);
    if (data_iter == input_op_datas_.end()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "No input data.");
    }
    const auto &input_data = data_iter->second[0];
    MS_EXCEPTION_IF_NULL(input_data);
    input_device_tensor_[0] = input_data->data_;

    MS_EXCEPTION_IF_NULL(output_);
    output_device_tensor_[0] = output_;
  }

  if (is_need_update_output_size_ && (input_device_tensor_[0]->GetSize() != output_device_tensor_[0]->GetSize())) {
    MS_LOG(INFO) << GetAID().Name() << " update output size from " << output_device_tensor_[0]->GetSize() << " to "
                 << input_device_tensor_[0]->GetSize();
    output_device_tensor_[0]->SetSize(input_device_tensor_[0]->GetSize());
  }
}

void CopyActor::UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &, const AnfNodePtr &,
                                 OpContext<DeviceTensor> *const) {
  MS_EXCEPTION_IF_NULL(output_data);
  output_data->data_ = output_device_tensor_[0];
}
}  // namespace runtime
}  // namespace mindspore
