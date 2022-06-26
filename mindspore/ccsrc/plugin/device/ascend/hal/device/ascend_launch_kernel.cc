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

#include "plugin/device/ascend/hal/device/ascend_launch_kernel.h"
#include "runtime/device/memory_manager.h"
#include "plugin/device/ascend/hal/device/ascend_memory_pool.h"
#include "plugin/device/ascend/hal/device/kernel_build_ascend.h"
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"

namespace mindspore::device::ascend {
void AscendLaunchKernel::FreeDeviceMem(void *addr) { AscendMemoryPool::GetInstance().FreeTensorMem(addr); }

size_t AscendLaunchKernel::AlignSizeForLaunchKernel(size_t size) { return MemoryManager::GetCommonAlignSize(size); }

uint8_t *AscendLaunchKernel::AllocDeviceMem(size_t size) {
  auto device_memory = AscendMemoryPool::GetInstance().AllocTensorMem(size);
  if (device_memory == nullptr) {
    MS_LOG(EXCEPTION) << "Fail to alloc memory, size: " << size;
  }
  return static_cast<uint8_t *>(device_memory);
}

void AscendLaunchKernel::KernelSelect(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto node_list = kernel_graph->execution_order();
  for (size_t i = 0; i < node_list.size(); ++i) {
    auto status = device::ascend::SelectKernelInfo(node_list[i]);
    if (status == ascend::kNoMatched) {
      MS_LOG(ERROR) << "Cnode name : " << node_list[i]->fullname_with_scope() << " kernel select failed";
    }
  }
}

void AscendLaunchKernel::KernelBuild(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto ret = device::ascend::KernelBuild(kernel_graph->execution_order());
  if (!ret) {
    MS_LOG(ERROR) << "kernel build failed";
  }
}
}  // namespace mindspore::device::ascend
