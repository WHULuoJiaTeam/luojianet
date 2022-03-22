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
#include "hash_utils.h"
#include "compile_cache_hasher.h"
namespace ge {
CacheHashKey CompileCacheHasher::GetCacheDescHashWithoutShape(const CompileCacheDesc &cache_desc) {
  CacheHashKey hash_key = HashUtils::MultiHash(cache_desc.unique_id_);
  hash_key = HashUtils::MultiHash(hash_key, cache_desc.formats_, cache_desc.origin_formats_,
                                            cache_desc.data_types_, cache_desc.other_desc_);
  return hash_key;
}
CacheHashKey CompileCacheHasher::GetCacheDescShapeHash(const CompileCacheDesc &cache_desc) {
  CacheHashKey hash_key = HashUtils::MultiHash();
  hash_key = HashUtils::MultiHash(hash_key, cache_desc.shapes_, cache_desc.origin_shapes_);
  return hash_key;
}
}