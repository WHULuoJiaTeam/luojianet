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

#include "graph/manager/host_mem_manager.h"

#include <sstream>

#include "graph/ge_context.h"
#include "graph/utils/tensor_utils.h"
#include "runtime/mem.h"

namespace {
const uint32_t kMallocHostMemFlag = 0;
}  // namespace
namespace ge {
Status SharedMemAllocator::Allocate(SharedMemInfo &mem_info) {
  auto device_id = GetContext().DeviceId();
  GELOGD("SharedMemAllocator::Malloc host mem size= %zu for devid:[%u].", mem_info.mem_size, device_id);

  auto dev_id = static_cast<int32_t>(device_id);
  GE_CHK_RT_RET(rtSetDevice(dev_id));
  // DeviceReset before memory finished!
  GE_MAKE_GUARD(not_used_var, [&] { GE_CHK_RT(rtDeviceReset(dev_id)); });

  rtMallocHostSharedMemoryIn input_para = {mem_info.shm_name.c_str(), mem_info.mem_size, kMallocHostMemFlag};
  rtMallocHostSharedMemoryOut output_para;
  rtError_t rt_ret = rtMallocHostSharedMemory(&input_para, &output_para);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMallocHostSharedMemory fail, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMallocHostSharedMemory] failed, devid:[%u].", device_id);
    return GE_GRAPH_MEMORY_ALLOC_FAILED;
  }
  mem_info.fd = output_para.fd;
  mem_info.host_aligned_ptr = AlignedPtr::BuildFromAllocFunc(
    [&output_para](std::unique_ptr<uint8_t[], AlignedPtr::Deleter> &ptr) {
      ptr.reset(reinterpret_cast<uint8_t *>(output_para.ptr));
    },
    [](uint8_t *ptr) { ptr = nullptr; });
  mem_info.device_address = reinterpret_cast<uint8_t *>(output_para.devPtr);
  return SUCCESS;
}

Status SharedMemAllocator::DeAllocate(SharedMemInfo &mem_info) {
  GELOGD("SharedMemAllocator::DeAllocate");
  rtFreeHostSharedMemoryIn free_para = {mem_info.shm_name.c_str(), mem_info.mem_size, mem_info.fd,
                                        mem_info.host_aligned_ptr->MutableGet(), mem_info.device_address};
  rtError_t rt_ret = rtFreeHostSharedMemory(&free_para);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtFreeHostSharedMemory fail, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtFreeHostSharedMemory] failed, ret:0x%X.", rt_ret);
    return RT_FAILED;
  }
  return ge::SUCCESS;
}

HostMemManager &HostMemManager::Instance() {
  static HostMemManager mem_manager;
  return mem_manager;
}

Status HostMemManager::Initialize() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  allocator_ = std::unique_ptr<SharedMemAllocator>(new (std::nothrow) SharedMemAllocator());
  if (allocator_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "New SharedMemAllocator fail");
    GELOGE(GE_GRAPH_MALLOC_FAILED, "[New][SharedMemAllocator] failed!");
    return GE_GRAPH_MALLOC_FAILED;
  }
  return SUCCESS;
}

void HostMemManager::Finalize() noexcept {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &it : var_memory_base_map_) {
    if (allocator_->DeAllocate(it.second) != SUCCESS) {
      GELOGW("Host %s mem release failed!", it.first.c_str());
    }
  }
  var_memory_base_map_.clear();
}

Status HostMemManager::MallocSharedMemory(SharedMemInfo &mem_info) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto iter = var_memory_base_map_.find(mem_info.op_name);
  if (iter != var_memory_base_map_.end()) {
    REPORT_INNER_ERROR("E19999", "Host shared memory for op %s has been malloced", mem_info.op_name.c_str());
    GELOGE(FAILED, "[Check][Param] Host shared memory for op %s has been malloced", mem_info.op_name.c_str());
    return FAILED;
  }
  mem_info.shm_name = OpNameToShmName(mem_info.op_name);
  GE_CHECK_NOTNULL(allocator_);
  GE_CHK_STATUS_RET(allocator_->Allocate(mem_info));
  var_memory_base_map_[mem_info.op_name] = mem_info;
  return SUCCESS;
}

bool HostMemManager::QueryVarMemInfo(const string &op_name, SharedMemInfo &mem_info) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = var_memory_base_map_.find(op_name);
  if (it == var_memory_base_map_.end()) {
    GELOGW("Host memory for node [%s] not found.", op_name.c_str());
    return false;
  }
  mem_info = it->second;
  return true;
}

string HostMemManager::OpNameToShmName(const string &op_name) {
  string sh_name("Ascend_");
  std::hash<std::string> hash_str;
  sh_name.append(std::to_string(hash_str(op_name)));
  return sh_name;
}
}  // namespace ge
