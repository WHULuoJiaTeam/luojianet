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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TBE_HANDLE_STORE_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_TBE_HANDLE_STORE_H_

#include <cstdint>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "framework/common/fmk_types.h"
#include "graph/op_kernel_bin.h"

namespace ge {
class TbeHandleInfo {
 public:
  TbeHandleInfo(void *handle, std::shared_ptr<OpKernelBin> &kernel) : used_(0), handle_(handle), kernel_(kernel) {}

  ~TbeHandleInfo() { handle_ = nullptr; }

  void used_inc(uint32_t num = 1);
  void used_dec(uint32_t num = 1);
  uint32_t used_num() const;

  void *handle() const;

 private:
  uint32_t used_;

  void *handle_;
  std::shared_ptr<OpKernelBin> kernel_;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY TBEHandleStore {
 public:
  static TBEHandleStore &GetInstance();

  ///
  /// @ingroup ge
  /// @brief Find Registered TBE handle by name.
  /// @param [in] name: TBE handle name to find.
  /// @param [out] handle: TBE handle addr found.
  /// @return true: found / false: not found.
  ///
  bool FindTBEHandle(const std::string &name, void *&handle);

  ///
  /// @ingroup ge
  /// @brief Store registered TBE handle info.
  /// @param [in] name: TBE handle name to store.
  /// @param [in] handle: TBE handle addr to store.
  /// @param [in] kernel: TBE kernel bin to store.
  /// @return NA
  ///
  void StoreTBEHandle(const std::string &name, void *handle, std::shared_ptr<OpKernelBin> &kernel);

  ///
  /// @ingroup ge
  /// @brief Increase reference of registered TBE handle info.
  /// @param [in] name: handle name increase reference.
  /// @return NA
  ///
  void ReferTBEHandle(const std::string &name);

  ///
  /// @ingroup ge
  /// @brief Erase TBE registered handle record.
  /// @param [in] names: handle names erase.
  /// @return NA
  ///
  void EraseTBEHandle(const std::map<std::string, uint32_t> &names);

 private:
  TBEHandleStore() = default;
  ~TBEHandleStore() = default;

  std::mutex mutex_;
  std::unordered_map<std::string, TbeHandleInfo> kernels_;
};
}  // namespace ge

#endif  // NEW_GE_TBE_HANDLE_STORE_H
