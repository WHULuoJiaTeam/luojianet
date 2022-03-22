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
#include "graph/load/model_manager/tbe_handle_store.h"

#include <limits>
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "runtime/kernel.h"

namespace ge {
void TbeHandleInfo::used_inc(uint32_t num) {
  if (used_ > std::numeric_limits<uint32_t>::max() - num) {
    REPORT_INNER_ERROR("E19999", "Used:%u reach numeric max", used_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Used[%u] reach numeric max.", used_);
    return;
  }

  used_ += num;
}

void TbeHandleInfo::used_dec(uint32_t num) {
  if (used_ < std::numeric_limits<uint32_t>::min() + num) {
    REPORT_INNER_ERROR("E19999", "Used:%u reach numeric min", used_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Used[%u] reach numeric min.", used_);
    return;
  }

  used_ -= num;
}

uint32_t TbeHandleInfo::used_num() const {
  return used_;
}

void *TbeHandleInfo::handle() const {
  return handle_;
}


TBEHandleStore &TBEHandleStore::GetInstance() {
  static TBEHandleStore instance;

  return instance;
}

///
/// @ingroup ge
/// @brief Find Registered TBE handle by name.
/// @param [in] name: TBE handle name to find.
/// @param [out] handle: handle names record.
/// @return true: found / false: not found.
///
bool TBEHandleStore::FindTBEHandle(const std::string &name, void *&handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    return false;
  } else {
    TbeHandleInfo &info = it->second;
    handle = info.handle();
    return true;
  }
}

///
/// @ingroup ge
/// @brief Store registered TBE handle info.
/// @param [in] name: TBE handle name to store.
/// @param [in] handle: TBE handle addr to store.
/// @param [in] kernel: TBE kernel bin to store.
/// @return NA
///
void TBEHandleStore::StoreTBEHandle(const std::string &name, void *handle,
                                    std::shared_ptr<OpKernelBin> &kernel) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    TbeHandleInfo info(handle, kernel);
    info.used_inc();
    kernels_.emplace(name, info);
  } else {
    TbeHandleInfo &info = it->second;
    info.used_inc();
  }
}

///
/// @ingroup ge
/// @brief Increase reference of registered TBE handle info.
/// @param [in] name: handle name increase reference.
/// @return NA
///
void TBEHandleStore::ReferTBEHandle(const std::string &name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(name);
  if (it == kernels_.end()) {
    REPORT_INNER_ERROR("E19999", "Kernel:%s not found in stored check invalid", name.c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Kernel[%s] not found in stored.", name.c_str());
    return;
  }

  TbeHandleInfo &info = it->second;
  info.used_inc();
}

///
/// @ingroup ge
/// @brief Erase TBE registered handle record.
/// @param [in] names: handle names erase.
/// @return NA
///
void TBEHandleStore::EraseTBEHandle(const std::map<std::string, uint32_t> &names) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &item : names) {
    auto it = kernels_.find(item.first);
    if (it == kernels_.end()) {
      REPORT_INNER_ERROR("E19999", "Kernel:%s not found in stored check invalid", item.first.c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Kernel[%s] not found in stored.", item.first.c_str());
      continue;
    }

    TbeHandleInfo &info = it->second;
    if (info.used_num() > item.second) {
      info.used_dec(item.second);
    } else {
      rtError_t rt_ret = rtDevBinaryUnRegister(info.handle());
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_INNER_ERROR("E19999", "Call rtDevBinaryUnRegister failed for Kernel:%s fail, ret:0x%X",
                           item.first.c_str(), rt_ret);
        GELOGE(INTERNAL_ERROR, "[Call][RtDevBinaryUnRegister] Kernel[%s] UnRegister handle fail:%u.",
               item.first.c_str(), rt_ret);
      }
      kernels_.erase(it);
    }
  }
}
} // namespace ge
