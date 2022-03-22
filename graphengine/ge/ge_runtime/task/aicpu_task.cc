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

#include "ge_runtime/task/aicpu_task.h"
#include <vector>
#include "ge_runtime/task/task_factory.h"
#include "aicpu/common/aicpu_task_struct.h"

namespace ge {
namespace model_runner {
AicpuTask::AicpuTask(const ModelContext &model_context, const std::shared_ptr<AicpuTaskInfo> &task_info)
    : TaskRepeater<AicpuTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      args_(nullptr),
      ext_info_(nullptr),
      input_output_addr_(nullptr) {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }

  auto stream_list = model_context.stream_list();
  if (stream_list.size() == 1) {
    stream_ = stream_list[0];
  } else if (stream_list.size() > task_info->stream_id()) {
    stream_ = stream_list[task_info->stream_id()];
  } else {
    GELOGW("index: %u >= stream_list.size(): %zu.", task_info->stream_id(), stream_list.size());
  }
}

AicpuTask::~AicpuTask() {
  ReleaseRtMem(&args_);
  ReleaseRtMem(&ext_info_);
}

bool AicpuTask::Distribute() {
  GELOGI("InitAicpuTask start.");
  vector<void *> io_addrs;
  io_addrs.insert(io_addrs.end(), task_info_->input_data_addrs().begin(), task_info_->input_data_addrs().end());
  io_addrs.insert(io_addrs.end(), task_info_->output_data_addrs().begin(), task_info_->output_data_addrs().end());
  auto io_addrs_num = static_cast<uint32_t>(io_addrs.size());
  auto io_addrs_size = static_cast<uint32_t>(io_addrs_num * sizeof(void *));
  constexpr uint32_t io_addr_offset = sizeof(aicpu::AicpuParamHead);
  uint32_t node_def_len_offset = io_addr_offset + io_addrs_size;
  uint32_t node_def_addr_offset = node_def_len_offset + sizeof(uint32_t);
  uint32_t args_size = sizeof(aicpu::AicpuParamHead) + io_addrs_size +
                       static_cast<uint32_t>(task_info_->node_def().size()) + sizeof(uint32_t);

  aicpu::AicpuParamHead aicpu_param_head;
  aicpu_param_head.length = args_size;
  aicpu_param_head.ioAddrNum = io_addrs_num;
  auto ext_info = task_info_->ext_info();
  uint32_t ext_size = ext_info.size();
  if (ext_info.empty()) {
    aicpu_param_head.extInfoLength = 0;
    aicpu_param_head.extInfoAddr = 0;
  } else {
    rtError_t flag = rtMalloc(&ext_info_, ext_size, RT_MEMORY_HBM);
    if (flag != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api(rtMalloc) failed, ret: 0x%X.", flag);
      return false;
    }

    flag = rtMemcpy(ext_info_, ext_size, const_cast<void *>(reinterpret_cast<const void *>(ext_info.data())), ext_size,
                    RT_MEMCPY_HOST_TO_DEVICE);
    if (flag != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api(rtMemCpy) failed, ret: 0x%X.", flag);
      return false;
    }

    GELOGI("ext info size: %u", ext_size);
    aicpu_param_head.extInfoLength = ext_size;
    aicpu_param_head.extInfoAddr = reinterpret_cast<uintptr_t>(ext_info_);
  }

  // Malloc device memory for args
  rtError_t rt_ret = rtMalloc(&args_, args_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api(rtMalloc) failed, ret: 0x%X.", rt_ret);
    return false;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "task args data.", args_size)
  // Memcpy AicpuParamHead
  rt_ret = rtMemcpy(args_, sizeof(aicpu::AicpuParamHead), reinterpret_cast<void *>(&aicpu_param_head),
                    sizeof(aicpu::AicpuParamHead), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api(rtMemcpy) failed, ret: 0x%X.", rt_ret);
    return false;
  }

  // Memcpy io addrs
  if (io_addrs_num != 0) {
    rt_ret = rtMemcpy(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(args_) + io_addr_offset), io_addrs_size,
                      reinterpret_cast<void *>(io_addrs.data()), io_addrs_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api(rtMemcpy) failed, ret: 0x%X.", rt_ret);
      return false;
    }
  }

  // Memcpy node def
  auto size = task_info_->node_def().size();
  rt_ret =
    rtMemcpy(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(args_) + node_def_len_offset), sizeof(uint32_t),
             reinterpret_cast<const void *>(&size), sizeof(uint32_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api(rtMemcpy) failed, ret: 0x%X.", rt_ret);
    return false;
  }

  // Memcpy node def
  rt_ret = rtMemcpy(reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(args_) + node_def_addr_offset),
                    task_info_->node_def().size(), reinterpret_cast<const void *>(task_info_->node_def().data()),
                    task_info_->node_def().size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api(rtMemcpy) failed, ret: 0x%X.", rt_ret);
    return false;
  }

  input_output_addr_ = reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(args_) + io_addr_offset);

  auto dump_flag = task_info_->dump_flag() ? RT_KERNEL_DUMPFLAG : RT_KERNEL_DEFAULT;
  GELOGI(
    "Distribute AicpuTask start, args_size = %u, io_addrs_num = %u, so_name = %s, kernel_name = %s, dump_flag = %d.",
    args_size, io_addrs_num, task_info_->so_name().data(), task_info_->kernel_name().data(), dump_flag);
  rt_ret = rtCpuKernelLaunchWithFlag(reinterpret_cast<const void *>(task_info_->so_name().data()),
                                     reinterpret_cast<const void *>(task_info_->kernel_name().data()), 1, args_,
                                     args_size, nullptr, stream_, dump_flag);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  GELOGI("Distribute AicpuTask end.");
  return true;
}

void AicpuTask::ReleaseRtMem(void **ptr) noexcept {
  if (ptr == nullptr || *ptr == nullptr) {
    return;
  }

  rtError_t rt_ret = rtFree(*ptr);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "ReleaseRtMem failed, ret: 0x%X", rt_ret);
    return;
  }
  *ptr = nullptr;
}

REGISTER_TASK(TaskInfoType::AICPU, AicpuTask, AicpuTaskInfo);
}  // namespace model_runner
}  // namespace ge
