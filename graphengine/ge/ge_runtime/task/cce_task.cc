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

#include "ge_runtime/task/cce_task.h"
#include "ge_runtime/task/task_factory.h"

namespace ge {
namespace model_runner {
CceTask::CceTask(const ModelContext &model_context, const std::shared_ptr<CceTaskInfo> &task_info)
    : TaskRepeater<CceTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      stub_func_(nullptr),
      args_(nullptr),
      sm_desc_(nullptr),
      flowtable_(nullptr),
      is_flowtable_(false) {
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

CceTask::~CceTask() {
  FreeRtMem(&args_);
  FreeRtMem(&flowtable_);
  rtError_t ret = (sm_desc_ != nullptr) ? rtMemFreeManaged(sm_desc_) : RT_ERROR_NONE;
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", ret);
  }
  sm_desc_ = nullptr;
}

void CceTask::FreeRtMem(void **ptr) noexcept {
  if (ptr == nullptr || *ptr == nullptr) {
    return;
  }
  rtError_t ret = rtFree(*ptr);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", ret);
  }

  *ptr = nullptr;
}

bool CceTask::Distribute() {
  GELOGI("Distribute CCETask start.");
  if (stream_ == nullptr) {
    GELOGE(PARAM_INVALID, "stream_ is null!");
    return false;
  }
  // Get stub_func
  if (task_info_->stub_func().empty()) {
    GELOGE(PARAM_INVALID, "kernel_info->stub_func is empty!");
    return false;
  }

  rtError_t rt_ret = rtGetFunctionByName(const_cast<char *>(task_info_->stub_func().c_str()), &stub_func_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtGetFunctionByName failed, ret: 0x%X", rt_ret);
    stub_func_ = nullptr;
    return false;
  }
  GELOGI("CCETask: stub_func = %s [%p].", task_info_->stub_func().c_str(), stub_func_);

  // Flowtable
  if (is_flowtable_) {
    rt_ret = rtMalloc(&flowtable_, task_info_->flow_table().size(), RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }
    GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "task information.", task_info_->flow_table().size())

    rt_ret = rtMemcpy(flowtable_, task_info_->flow_table().size(), task_info_->flow_table().data(),
                      task_info_->flow_table().size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }

    // Modify flowtable addr in args
    auto args = const_cast<uint8_t *>(task_info_->args().data());
    auto task_offset = reinterpret_cast<uint16_t *>(const_cast<uint8_t *>(task_info_->args_offset().data()));

    if (task_info_->args().size() < (task_offset[0] + sizeof(uint64_t))) {
      GELOGE(FAILED, "(context.args_offset().data()))[0]:%u + sizeof(uint64_t):%zu > kernelDef.args().size():%zu",
             static_cast<uint32_t>(task_offset[0]), sizeof(uint64_t), task_info_->args().size());
      return false;
    }

    *(reinterpret_cast<uintptr_t *>(args + task_offset[0])) = reinterpret_cast<uintptr_t>(flowtable_);
  }

  // Args
  rt_ret = rtMalloc(&args_, task_info_->args_size(), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "task information.", task_info_->args_size())

  rt_ret = rtMemcpy(args_, task_info_->args_size(), task_info_->args().data(), task_info_->args_size(),
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }

  // L2 sm_desc
  if (!task_info_->sm_desc().empty()) {
    rt_ret = rtMemAllocManaged(&sm_desc_, task_info_->sm_desc().size(), RT_MEMORY_SPM);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }

    rt_ret = rtMemcpy(sm_desc_, task_info_->sm_desc().size(), task_info_->sm_desc().data(),
                      task_info_->sm_desc().size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
      return false;
    }
  }

  // Kernel launch
  rt_ret = rtKernelLaunch(stub_func_, task_info_->block_dim(), args_, task_info_->args_size(),
                          static_cast<rtSmDesc_t *>(sm_desc_), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }
  return true;
}

REGISTER_TASK(TaskInfoType::CCE, CceTask, CceTaskInfo);
}  // namespace model_runner
}  // namespace ge
