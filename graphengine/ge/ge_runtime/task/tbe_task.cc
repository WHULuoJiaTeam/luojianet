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

#include "ge_runtime/task/tbe_task.h"
#include <vector>
#include "ge_runtime/task/task_factory.h"

namespace ge {
namespace model_runner {
TbeTask::TbeTask(const ModelContext &model_context, const std::shared_ptr<TbeTaskInfo> &task_info)
    : TaskRepeater<TbeTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      stub_func_(nullptr),
      args_(nullptr) {
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
    GELOGE(PARAM_INVALID, "Index: %u >= stream_list.size(): %zu.", task_info->stream_id(), stream_list.size());
    return;
  }
}

TbeTask::~TbeTask() {
  if (args_ != nullptr) {
    rtError_t rt_ret = rtFree(args_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGE(RT_FAILED, "rtFree fwkOpBuf failed! ret: 0x%X.", rt_ret);
    }
    args_ = nullptr;
  }
}

bool TbeTask::Distribute() {
  GELOGI("InitTbeTask start.");
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
    GELOGE(RT_FAILED, "rtGetFunctionByName failed, ret: %d", static_cast<int32_t>(rt_ret));
    stub_func_ = nullptr;
    return false;
  }
  GELOGI("TbeTask: stub_func = %s [%p].", task_info_->stub_func().c_str(), stub_func_);

  // Get args
  std::vector<void *> tensor_device_addrs;
  tensor_device_addrs.insert(tensor_device_addrs.end(), task_info_->input_data_addrs().begin(),
                             task_info_->input_data_addrs().end());
  tensor_device_addrs.insert(tensor_device_addrs.end(), task_info_->output_data_addrs().begin(),
                             task_info_->output_data_addrs().end());
  tensor_device_addrs.insert(tensor_device_addrs.end(), task_info_->workspace_addrs().begin(),
                             task_info_->workspace_addrs().end());
  auto args_size = static_cast<uint32_t>(tensor_device_addrs.size() * sizeof(void *));

  rt_ret = rtMalloc(&args_, args_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMalloc failed, ret: %d", static_cast<int32_t>(rt_ret));
    return false;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "task args data.", args_size)

  rt_ret = rtMemcpy(args_, args_size, reinterpret_cast<void *>(tensor_device_addrs.data()), args_size,
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtMemcpy fail, ret 0x%X.", rt_ret);
    return false;
  }

  GELOGI("DistributeTbeTask start.");
  auto dump_flag = task_info_->dump_flag() ? RT_KERNEL_DUMPFLAG : RT_KERNEL_DEFAULT;
  rt_ret = rtKernelLaunchWithFlag(stub_func_, task_info_->block_dim(), args_, args_size, nullptr, stream_, dump_flag);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api rtKernelLaunch failed, ret: 0x%X", rt_ret);
    return false;
  }
  GELOGI("[DataDump] task name:%s, dump_flag:%d", task_info_->op_name().c_str(), dump_flag);
  return true;
}

REGISTER_TASK(TaskInfoType::TBE, TbeTask, TbeTaskInfo);
}  // namespace model_runner
}  // namespace ge
