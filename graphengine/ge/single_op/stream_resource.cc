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

#include "single_op/stream_resource.h"

#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "runtime/rt.h"
#include "single_op/single_op_model.h"

namespace ge {
namespace {
// limit available device mem size  1M
const uint32_t kFuzzDeviceBufferSize = 1 * 1024 * 1024;
constexpr int kDefaultThreadNum = 4;
}

StreamResource::StreamResource(uintptr_t resource_id) : resource_id_(resource_id) {
}

StreamResource::~StreamResource() {
  for (auto mem : memory_list_) {
    if (mem != nullptr) {
      auto rt_ret = rtFree(mem);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[Free][Rt] failed."));
    }
  }

  for (auto weight : weight_list_) {
    if (weight != nullptr) {
      auto rt_ret = rtFree(weight);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[Free][Rt] failed."));
    }
  }

  if (device_buffer_ != nullptr) {
    auto rt_ret = rtFree(device_buffer_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[Free][Rt] failed."));
  }
}

Status StreamResource::Init() {
  auto rt_ret = rtMalloc(&device_buffer_, kFuzzDeviceBufferSize, RT_MEMORY_HBM);
  GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[Malloc][Rt] failed."));
  return SUCCESS;
}

SingleOp *StreamResource::GetOperator(const uint64_t key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = op_map_.find(key);
  if (it == op_map_.end()) {
    return nullptr;
  }

  return it->second.get();
}

DynamicSingleOp *StreamResource::GetDynamicOperator(const uint64_t key) {
  std::lock_guard<std::mutex> lk(mu_);
  auto it = dynamic_op_map_.find(key);
  if (it == dynamic_op_map_.end()) {
    return nullptr;
  }

  return it->second.get();
}

rtStream_t StreamResource::GetStream() const {
  return stream_;
}

void StreamResource::SetStream(rtStream_t stream) {
  stream_ = stream;
}

uint8_t *StreamResource::DoMallocMemory(const std::string &purpose,
                                        size_t size,
                                        size_t &max_allocated,
                                        std::vector<uint8_t *> &allocated) {
  if (size == 0) {
    GELOGD("Mem size == 0");
    return nullptr;
  }

  if (size <= max_allocated && !allocated.empty()) {
    GELOGD("reuse last memory");
    return allocated.back();
  }

  if (!allocated.empty()) {
    uint8_t *current_buffer = allocated.back();
    allocated.pop_back();
    if (rtStreamSynchronize(stream_) != RT_ERROR_NONE) {
      GELOGW("Failed to invoke rtStreamSynchronize");
    }
    (void) rtFree(current_buffer);
  }

  uint8_t *buffer = nullptr;
  auto ret = rtMalloc(reinterpret_cast<void **>(&buffer), size, RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[RtMalloc][Memory] failed, size = %zu, ret = %d", size, ret);
    REPORT_INNER_ERROR("E19999", "rtMalloc failed, size = %zu, ret = %d.", size, ret);
    return nullptr;
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, purpose.c_str(), size)

  ret = rtMemset(buffer, size, 0U, size);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[RtMemset][Memory] failed, ret = %d", ret);
    REPORT_INNER_ERROR("E19999", "rtMemset failed, ret = %d.", ret);
    auto rt_ret = rtFree(buffer);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, GELOGE(RT_FAILED, "[RtFree][Memory] failed"));
    return nullptr;
  }

  GELOGD("Malloc new memory succeeded. size = %zu", size);
  max_allocated = size;
  allocated.emplace_back(buffer);
  return buffer;
}

uint8_t *StreamResource::MallocMemory(const std::string &purpose, size_t size, bool holding_lock) {
  GELOGD("To Malloc memory, size = %zu", size);
  if (holding_lock) {
    return DoMallocMemory(purpose, size, max_memory_size_, memory_list_);
  } else {
    std::lock_guard<std::mutex> lk(stream_mu_);
    return DoMallocMemory(purpose, size, max_memory_size_, memory_list_);
  }
}

uint8_t *StreamResource::MallocWeight(const std::string &purpose, size_t size) {
  GELOGD("To Malloc weight, size = %zu", size);
  uint8_t *buffer = nullptr;
  auto ret = rtMalloc(reinterpret_cast<void **>(&buffer), size, RT_MEMORY_HBM);
  if (ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[RtMalloc][Memory] failed, size = %zu, ret = %d", size, ret);
    REPORT_INNER_ERROR("E19999", "rtMalloc failed, size = %zu, ret = %d.", size, ret);
    return nullptr;
  }

  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, purpose.c_str(), size)
  weight_list_.emplace_back(buffer);
  return buffer;
}

Status StreamResource::BuildDynamicOperator(const ModelData &model_data,
                                            DynamicSingleOp **single_op,
                                            const uint64_t model_id) {
  const string &model_name = std::to_string(model_id);
  std::lock_guard<std::mutex> lk(mu_);
  auto it = dynamic_op_map_.find(model_id);
  if (it != dynamic_op_map_.end()) {
    *single_op = it->second.get();
    return SUCCESS;
  }

  SingleOpModel model(model_name, model_data.model_data, model_data.model_len);
  auto ret = model.Init();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][SingleOpModel] failed. model = %s, ret = %u", model_name.c_str(), ret);
    REPORT_CALL_ERROR("E19999", "SingleOpModel init failed, model = %s, ret = %u", model_name.c_str(), ret);
    return ret;
  }

  auto new_op = std::unique_ptr<DynamicSingleOp>(new(std::nothrow) DynamicSingleOp(resource_id_, &stream_mu_, stream_));
  GE_CHECK_NOTNULL(new_op);

  GELOGI("To build operator: %s", model_name.c_str());
  GE_CHK_STATUS_RET(model.BuildDynamicOp(*this, *new_op),
                    "[Build][DynamicOp]failed. op = %s, ret = %u", model_name.c_str(), ret);
  *single_op = new_op.get();
  dynamic_op_map_[model_id] = std::move(new_op);
  return SUCCESS;
}

Status StreamResource::BuildOperator(const ModelData &model_data, SingleOp **single_op, const uint64_t model_id) {
  const string &model_name = std::to_string(model_id);
  std::lock_guard<std::mutex> lk(mu_);
  auto it = op_map_.find(model_id);
  if (it != op_map_.end()) {
    *single_op = it->second.get();
    return SUCCESS;
  }

  SingleOpModel model(model_name, model_data.model_data, model_data.model_len);
  auto ret = model.Init();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][SingleOpModel] failed. model = %s, ret = %u", model_name.c_str(), ret);
    REPORT_CALL_ERROR("E19999", "SingleOpModel init failed, model = %s, ret = %u", model_name.c_str(), ret);
    return ret;
  }

  auto new_op = std::unique_ptr<SingleOp>(new(std::nothrow) SingleOp(this, &stream_mu_, stream_));
  if (new_op == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[New][SingleOp] failed.");
    REPORT_CALL_ERROR("E19999", "new SingleOp failed.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  GELOGI("To build operator: %s", model_name.c_str());
  GE_CHK_STATUS_RET(model.BuildOp(*this, *new_op), "[Build][Op] failed. op = %s, ret = %u", model_name.c_str(), ret);

  *single_op = new_op.get();
  op_map_[model_id] = std::move(new_op);
  return SUCCESS;
}

Status StreamResource::GetThreadPool(ThreadPool **thread_pool) {
  GE_CHECK_NOTNULL(thread_pool);
  if (thread_pool_ == nullptr) {
    thread_pool_.reset(new (std::nothrow) ThreadPool(kDefaultThreadNum));
    GE_CHECK_NOTNULL(thread_pool_);
  }
  *thread_pool = thread_pool_.get();
  return SUCCESS;
}

const uint8_t *StreamResource::GetMemoryBase() const {
  if (memory_list_.empty()) {
    return nullptr;
  }

  return memory_list_.back();
}
}  // namespace ge
