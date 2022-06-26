/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/hccl/hcom_all_reduce.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"

namespace mindspore {
namespace kernel {
bool HcomAllReduceKernel::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                 const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  MS_LOG(DEBUG) << "HcclAllReduce launch";
  if (inputs.empty() || outputs.empty() || hccl_data_type_list_.empty()) {
    MS_LOG(ERROR) << "Invalid AllReduce input, output or data type size (" << inputs.size() << ", " << outputs.size()
                  << ", " << hccl_data_type_list_.size() << ").";
    return false;
  }
  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (stream_ == nullptr) {
    stream_ = stream_ptr;
  }
  auto hccl_result = hccl::HcclAdapter::GetInstance().HcclAllReduce(inputs[0]->addr, outputs[0]->addr, hccl_count_,
                                                                    hccl_data_type_list_[0], op_type_, stream_, group_);
  if (hccl_result != HCCL_SUCCESS) {
    MS_LOG(ERROR) << "HcclAllReduce faled, ret:" << hccl_result;
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
