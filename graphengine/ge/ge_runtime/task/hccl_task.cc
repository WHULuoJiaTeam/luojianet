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

#include "ge_runtime/task/hccl_task.h"
#include <algorithm>
#include "framework/common/util.h"
#include "ge_runtime/task/task_factory.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/opskernel/ge_task_info.h"

namespace ge {
namespace model_runner {
std::map<rtModel_t, std::map<uint32_t, std::vector<std::weak_ptr<HcclTask::StreamGuard>>>>
  HcclTask::model_stream_mapping_;
std::mutex HcclTask::model_stream_mapping_mutex_;

HcclTask::HcclTask(const ModelContext &model_context, const std::shared_ptr<HcclTaskInfo> &task_info)
    : TaskRepeater<HcclTaskInfo>(model_context, task_info),
      task_info_(task_info),
      stream_(nullptr),
      workspace_mem_(nullptr),
      rt_model_handle_(nullptr),
      priority_(0),
      secondary_stream_list_() {
  if (task_info_ == nullptr) {
    GELOGW("task_info_ is null!");
    return;
  }

  priority_ = model_context.priority();
  rt_model_handle_ = model_context.rt_model_handle();
  auto stream_list = model_context.stream_list();

  if (stream_list.size() == 1) {
    stream_ = stream_list[0];
  } else if (stream_list.size() > task_info->stream_id()) {
    stream_ = stream_list[task_info->stream_id()];
  } else {
    GELOGW("Index: %u >= stream_list.size(): %zu.", task_info->stream_id(), stream_list.size());
  }
}

HcclTask::~HcclTask() {}

bool HcclTask::Distribute() {
  // Ops kernel info store
  // Get privateDef and opsKernelStorePtr
  GELOGI("Get custom info in modelTaskDef");
  void *ops_kernel_store = task_info_->ops_kernel_store();
  OpsKernelInfoStore *ops_kernel_info_store = reinterpret_cast<OpsKernelInfoStore *>(ops_kernel_store);
  if (ops_kernel_store == nullptr) {
    GELOGE(PARAM_INVALID, "No hcom distribute function ptr and no ops kernel store.");
    return false;
  }

  char *private_def = reinterpret_cast<char *>(const_cast<char unsigned *>(task_info_->private_def().data()));
  auto private_def_len = static_cast<uint32_t>(task_info_->private_def().size());
  GELOGI("The first address of the custom info, privateDef=%p", private_def);
  SetSecondaryStream();

  if (task_info_->workspace_size() > 0) {
    workspace_mem_ = task_info_->workspace_addr();
  }

  GELOGI("HcclTaskInfo Distribute Start. begin to call function LoadTask in hccl.");
  GETaskInfo ge_task;
  ge_task.id = 0;
  ge_task.type = static_cast<uint16_t>(RT_MODEL_TASK_HCCL);
  ge_task.stream = stream_;

  ge_task.kernelHcclInfo = std::vector<GETaskKernelHcclInfo>(1);
  ge_task.kernelHcclInfo[0].hccl_type = task_info_->hccl_type();
  ge_task.kernelHcclInfo[0].inputDataAddr = task_info_->input_data_addr();
  ge_task.kernelHcclInfo[0].outputDataAddr = task_info_->output_data_addr();
  ge_task.kernelHcclInfo[0].workSpaceAddr = workspace_mem_;
  ge_task.kernelHcclInfo[0].workSpaceMemSize = task_info_->workspace_size();
  ge_task.kernelHcclInfo[0].count = task_info_->count();
  ge_task.kernelHcclInfo[0].dataType = static_cast<int32_t>(task_info_->data_type());
  ge_task.kernelHcclInfo[0].opType = static_cast<int32_t>(task_info_->op_type());
  ge_task.kernelHcclInfo[0].rootId = task_info_->root_id();

  std::vector<rtStream_t> secondary_stream_list;
  std::transform(secondary_stream_list_.begin(), secondary_stream_list_.end(),
                 std::back_inserter(secondary_stream_list),
                 [](const std::shared_ptr<StreamGuard> &stream) -> rtStream_t { return stream->GetStream(); });
  ge_task.kernelHcclInfo[0].hcclStreamList = secondary_stream_list;

  ge_task.privateDef = private_def;
  ge_task.privateDefLen = private_def_len;
  ge_task.opsKernelStorePtr = ops_kernel_store;

  auto result = ops_kernel_info_store->LoadTask(ge_task);
  // tagHcclResult::HCCL_SUCCESS is 0
  if (result != 0) {
    GELOGE(INTERNAL_ERROR, "davinci_model : load task fail, return ret: %u", result);
    return false;
  }

  GELOGI("Call function LoadTask end.");
  return true;
}

bool HcclTask::SetSecondaryStream() {
  const uint32_t master_stream_id = task_info_->stream_id();
  const int64_t hccl_secondary_stream_num = task_info_->hccl_stream_num();
  Status ret;
  std::lock_guard<std::mutex> lock(model_stream_mapping_mutex_);
  if (model_stream_mapping_.find(rt_model_handle_) == model_stream_mapping_.end()) {
    GELOGI("Need to create map for rt_model_handle_:%p with new mainstream %u.", rt_model_handle_, master_stream_id);
    ret = CreateStream(hccl_secondary_stream_num, master_stream_id);
    if (!ret) {
      GELOGE(RT_FAILED, "Create hccl stream failed.");
      return false;
    }
    return true;
  }

  std::map<uint32_t, std::vector<std::weak_ptr<StreamGuard>>> &master_secondary_stream_map =
    model_stream_mapping_.at(rt_model_handle_);
  auto iter = master_secondary_stream_map.find(master_stream_id);
  if (iter != master_secondary_stream_map.end()) {
    std::vector<std::weak_ptr<StreamGuard>> &secondary_stream_vec = iter->second;
    auto lock_weak_ptr = [&secondary_stream_vec, this](int64_t index) -> bool {
      auto stream = secondary_stream_vec[index].lock();
      if (stream == nullptr) {
        rtStream_t new_stream = nullptr;
        bool ret = CreateStream(rt_model_handle_, &new_stream);
        if (!ret) {
          GELOGE(FAILED, "CreateStream failed.");
          return false;
        }
        stream = std::make_shared<HcclTask::StreamGuard>(rt_model_handle_, new_stream);
        GE_RT_FALSE_CHECK_NOTNULL(stream);
        secondary_stream_vec[index] = stream;
      }
      secondary_stream_list_.push_back(stream);
      return true;
    };

    if (static_cast<size_t>(hccl_secondary_stream_num) <= secondary_stream_vec.size()) {
      GELOGI("Number of secondary stream is enough to be reused.");
      for (int64_t i = 0; i < hccl_secondary_stream_num; ++i) {
        if (!lock_weak_ptr(i)) {
          GELOGE(FAILED, "Lock weak ptr failed.");
          return false;
        }
      }
    } else {
      GELOGI("Need to reuse secondary stream and create new secondary stream.");
      size_t created_stream_num = secondary_stream_vec.size();
      for (size_t i = 0; i < secondary_stream_vec.size(); ++i) {
        if (!lock_weak_ptr(i)) {
          GELOGE(FAILED, "Lock weak ptr failed.");
          return false;
        }
      }
      ret = CreateStream(hccl_secondary_stream_num - created_stream_num, master_stream_id);
      if (ret != SUCCESS) {
        GELOGE(RT_FAILED, "Create hccl stream failed.");
        return false;
      }
    }
    GELOGI("Initialize hccl secondary stream success, hccl_secondary_stream_num =%ld", hccl_secondary_stream_num);
  } else {
    GELOGI("Need to create secondary stream for %s with new mainstream %u.", task_info_->op_name().c_str(),
           master_stream_id);
    ret = CreateStream(hccl_secondary_stream_num, master_stream_id);
    if (!ret) {
      GELOGE(RT_FAILED, "Create hccl stream failed.");
      return false;
    }
  }
  return true;
}

bool HcclTask::CreateStream(int64_t stream_num, int64_t master_stream_id) {
  GELOGI("Start to create %ld hccl secondary stream.", stream_num);
  for (int64_t i = 0; i < stream_num; ++i) {
    rtStream_t stream = nullptr;
    bool ret = CreateStream(rt_model_handle_, &stream);
    if (!ret) {
      GELOGE(FAILED, "CreateStream failed.");
      return false;
    }

    GELOGD("hccl_stream addr is=%p", stream);
    auto shared_stream = std::make_shared<StreamGuard>(rt_model_handle_, stream);
    if (shared_stream == nullptr) {
      GELOGE(FAILED, "MakeShared failed.");
      return false;
    }
    SaveHcclSecondaryStream(master_stream_id, shared_stream);
    secondary_stream_list_.push_back(shared_stream);
  }
  GELOGI("CreateStream success.");
  return true;
}

bool HcclTask::CreateStream(rtModel_t model, rtStream_t *stream) const {
  if (stream == nullptr) {
    GELOGE(FAILED, "Output param stream is null.");
    return false;
  }

  rtError_t rt_ret = rtStreamCreateWithFlags(stream, priority_, RT_STREAM_PERSISTENT | RT_STREAM_FORCE_COPY);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }
  // Create secondary stream, inactive by default, activated by hccl
  rt_ret = rtModelBindStream(model, *stream, RT_MODEL_WAIT_ACTIVE_STREAM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Call rt api failed, ret: 0x%X", rt_ret);
    return false;
  }
  return true;
}

void HcclTask::SaveHcclSecondaryStream(int64_t master_stream_id, const std::shared_ptr<StreamGuard> &stream) {
  if (model_stream_mapping_.find(rt_model_handle_) == model_stream_mapping_.end()) {
    model_stream_mapping_.emplace(rt_model_handle_, std::map<uint32_t, std::vector<std::weak_ptr<StreamGuard>>>());
  }
  std::map<uint32_t, std::vector<std::weak_ptr<StreamGuard>>> &master_secondary_stream_map =
    model_stream_mapping_.at(rt_model_handle_);
  master_secondary_stream_map[master_stream_id].emplace_back(stream);
}

HcclTask::StreamGuard::~StreamGuard() {
  rtError_t rt_ret = rtModelUnbindStream(model_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Unbind stream from model failed!");
    return;
  }

  rt_ret = rtStreamDestroy(stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "Destroy stream failed!");
    return;
  }
}

REGISTER_TASK(TaskInfoType::HCCL, HcclTask, HcclTaskInfo);
}  // namespace model_runner
}  // namespace ge
