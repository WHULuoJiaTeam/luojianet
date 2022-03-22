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

#ifndef GE_GRAPH_MANAGER_HOST_VAR_MANAGER_H_
#define GE_GRAPH_MANAGER_HOST_VAR_MANAGER_H_

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "framework/common/l2_cache_optimize.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "external/graph/tensor.h"
#include "runtime/mem.h"

namespace ge {
struct SharedMemInfo {
  string op_name;
  string shm_name;
  uint64_t mem_size = 0;
  int fd = 0;
  uint8_t *device_address = nullptr;
  std::shared_ptr<AlignedPtr> host_aligned_ptr = nullptr;
  SharedMemInfo() = default;
  SharedMemInfo(string name, uint64_t size) : op_name(std::move(name)), mem_size(size) {}
};
class SharedMemAllocator {
 public:
  SharedMemAllocator() = default;
  ~SharedMemAllocator() = default;

  Status Allocate(SharedMemInfo &mem_info);
  Status DeAllocate(SharedMemInfo &mem_info);
};

class HostMemManager {
 public:
  HostMemManager() = default;
  ~HostMemManager() { Finalize(); }
  HostMemManager(const HostMemManager &) = delete;
  HostMemManager &operator=(const HostMemManager &) = delete;

  static HostMemManager &Instance();
  Status Initialize();
  void Finalize() noexcept;
  Status MallocSharedMemory(SharedMemInfo &mem_nfo);
  bool QueryVarMemInfo(const string &op_name, SharedMemInfo &mem_info);

 private:
  static string OpNameToShmName(const string &op_name);
  std::unordered_map<std::string, SharedMemInfo> var_memory_base_map_;
  std::unique_ptr<SharedMemAllocator> allocator_;
  mutable std::recursive_mutex mutex_;
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_HOST_VAR_MANAGER_H_
