/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PS_EMBEDDING_TABLE_SHARD_METADATA_H_
#define MINDSPORE_CCSRC_PS_EMBEDDING_TABLE_SHARD_METADATA_H_

#include <iostream>
#include "utils/log_adapter.h"

namespace mindspore {
namespace ps {
class EmbeddingTableShardMetadata {
 public:
  explicit EmbeddingTableShardMetadata(uint64_t begin, uint64_t end) : begin_(begin), end_(end) {}
  virtual ~EmbeddingTableShardMetadata() = default;

  uint64_t begin() const;
  uint64_t end() const;
  uint64_t size() const;

 private:
  uint64_t begin_;
  uint64_t end_;
};
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_EMBEDDING_TABLE_SHARD_METADATA_H_
