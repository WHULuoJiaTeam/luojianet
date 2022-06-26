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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_STORAGE_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_STORAGE_H_

#include <map>
#include <string>
#include <vector>
#include <tuple>
#include <utility>

#include "distributed/persistent/storage/block.h"
#include "distributed/persistent/storage/constants.h"

namespace mindspore {
namespace distributed {
namespace storage {
// InputData consists of shape, const buffer pointer and size
using InputData = std::tuple<std::vector<int>, const void *, size_t>;
// OutputData consists of buffer pointer and size
using OutputData = std::pair<void *, size_t>;
// DirtyInfo is used to indicate the part of the Tensor that needs to be rewritten to storage,
using DirtyInfo = std::vector<int>;

// This Class provides upper-layer interfaces for persistent storage.
class StorageBase {
 public:
  StorageBase() = default;
  virtual ~StorageBase() = default;

  // Write input tensor to storage medium or memory buffer.
  // The parameter dirty_info indicates that the part of the Tensor that needs to be rewritten to storage,
  // for example, some rows of embedding table need to be rewritten to storage, the dirty_info should contain these row
  // numbers.
  virtual void Write(const InputData &input, const DirtyInfo &dirty_info) {}

  // Write input to storage medium or memory buffer, only support the input composed of multiple tensors with same shape
  // and data type and using same dirty info at present.
  // The parameter dirty_info indicates that the part of the Tensor that needs to be rewritten to storage.
  virtual void Write(const std::vector<InputData> &input, const DirtyInfo &dirty_info) {}

  // Read data from the storage medium or memory buffer and merge them into contiguous memory.
  virtual void Read(const OutputData &output) {}

  // Read data from the storage medium or memory buffer and merge them into contiguous memory for multiple tensors.
  virtual void Read(const std::vector<OutputData> &outputs) {}
};
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_STORAGE_H_
