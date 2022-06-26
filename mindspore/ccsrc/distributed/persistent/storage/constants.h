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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_CONSTANTS_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_CONSTANTS_H_

namespace mindspore {
namespace distributed {
namespace storage {
// Block and BlockMeta related.
constexpr char kFieldsLength[] = "field_length";
constexpr char kOffset[] = "offset";
constexpr char kShardShape[] = "shard_shape";
constexpr char kShardRangeLowerBound[] = "shard_range_lower_bound";
constexpr char kShardRangeUpperBound[] = "shard_range_upper_bound";
constexpr char kHashSeq[] = "hash_seq";

constexpr char kBlockFilePrefix[] = "block_";
constexpr char kBlockMetaFilePrefix[] = "block_meta_";
constexpr char kJsonSuffix[] = ".json";
constexpr size_t JSON_SUFFIX_LENS = 5;

// Storage config related.
constexpr char kFileStoragePath[] = "file_storage_path";
constexpr char kMaxBlockLength[] = "max_block_length";
}  // namespace storage
}  // namespace distributed
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_PERSISTENT_STORAGE_CONSTANTS_H_
