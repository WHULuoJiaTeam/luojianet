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

#ifndef GE_SINGLE_OP_STREAM_RESOURCE_H_
#define GE_SINGLE_OP_STREAM_RESOURCE_H_

#include <string>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "runtime/stream.h"
#include "single_op/single_op.h"

namespace ge {
class StreamResource {
 public:
  explicit StreamResource(uintptr_t resource_id);
  ~StreamResource();

  StreamResource(const StreamResource &) = delete;
  StreamResource(StreamResource &&) = delete;
  StreamResource &operator=(const StreamResource &) = delete;
  StreamResource &operator=(StreamResource &&) = delete;
  rtStream_t GetStream() const;
  void SetStream(rtStream_t stream);

  Status Init();
  SingleOp *GetOperator(const uint64_t key);
  DynamicSingleOp *GetDynamicOperator(const uint64_t key);

  Status BuildOperator(const ModelData &model_data, SingleOp **single_op, const uint64_t model_id);
  Status BuildDynamicOperator(const ModelData &model_data, DynamicSingleOp **single_op, const uint64_t model_id);

  uint8_t *MallocMemory(const std::string &purpose, size_t size, bool holding_lock = true);
  uint8_t *MallocWeight(const std::string &purpose, size_t size);
  const uint8_t *GetMemoryBase() const;
  void *GetDeviceBufferAddr() const {
    return device_buffer_;
  }

  Status GetThreadPool(ThreadPool **thread_pool);

 private:
  uint8_t *DoMallocMemory(const std::string &purpose,
                          size_t size,
                          size_t &max_allocated,
                          std::vector<uint8_t *> &allocated);

  uintptr_t resource_id_;
  size_t max_memory_size_ = 0;
  std::vector<uint8_t *> memory_list_;
  std::vector<uint8_t *> weight_list_;
  std::unordered_map<uint64_t, std::unique_ptr<SingleOp>> op_map_;
  std::unordered_map<uint64_t, std::unique_ptr<DynamicSingleOp>> dynamic_op_map_;
  std::unique_ptr<ThreadPool> thread_pool_;
  rtStream_t stream_ = nullptr;
  std::mutex mu_;
  std::mutex stream_mu_;
  void *device_buffer_ = nullptr;
};
}  // namespace ge

#endif  // GE_SINGLE_OP_STREAM_RESOURCE_H_
