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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TDT_TDT_HANDLE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TDT_TDT_HANDLE_H_

#include <iostream>
#include <map>
#include <thread>
#include <utility>
#include "utils/log_adapter.h"
#include "acl/acl_tdt.h"

namespace mindspore {
namespace dataset {
class TdtHandle {
 public:
  static inline void AddHandle(acltdtChannelHandle **handle, std::thread *use_thread);

  static inline bool DestroyHandle();

  static inline void DelHandle(acltdtChannelHandle **handle);

 private:
  TdtHandle() {}
  ~TdtHandle() = default;
};

inline void TdtHandle::AddHandle(acltdtChannelHandle **handle, std::thread *use_thread) {
  if (*handle != nullptr) {
    auto ret = acl_handle_map.emplace(reinterpret_cast<void **>(handle), use_thread);
    if (!std::get<1>(ret)) {
      MS_LOG(ERROR) << "Failed to add new handle to acl_handle_map." << std::endl;
    }
  }
}

inline void TdtHandle::DelHandle(acltdtChannelHandle **handle) {
  void **void_handle = reinterpret_cast<void **>(handle);
  acl_handle_map.erase(void_handle);
}

inline bool TdtHandle::DestroyHandle() {
  bool destroy_all = true;
  for (auto &item : acl_handle_map) {
    acltdtChannelHandle **handle = reinterpret_cast<acltdtChannelHandle **>(item.first);
    if (*handle != nullptr) {
      aclError stop_status = acltdtStopChannel(*handle);
      if (stop_status != ACL_SUCCESS) {
        MS_LOG(ERROR) << "Failed stop acl data channel and the stop status is " << stop_status << std::endl;
        return false;
      }
      if (item.second != nullptr && item.second->joinable()) {
        item.second->join();
      }
      if (acltdtDestroyChannel(*handle) != ACL_SUCCESS) {
        destroy_all = false;
      } else {
        *handle = nullptr;
      }
    }
  }

  // clear the map container when all the handle has been destroyed
  if (destroy_all) {
    acl_handle_map.clear();
  }
  return destroy_all;
}

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_TDT_TDT_HANDLE_H_
