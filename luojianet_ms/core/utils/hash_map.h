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

#ifndef LUOJIANET_MS_CORE_UTILS_HASH_MAP_H_
#define LUOJIANET_MS_CORE_UTILS_HASH_MAP_H_

#include <functional>
#if (ENABLE_FAST_HASH_TABLE) && __has_include("robin_hood/robin_hood.h")
#include "robin_hood/robin_hood.h"

namespace luojianet_ms {
template <typename K, typename V, typename Hash = robin_hood::hash<K>, typename KeyEqual = std::equal_to<K>>
using HashMap = robin_hood::unordered_map<K, V, Hash, KeyEqual>;

#else
#include <unordered_map>

namespace luojianet_ms {
template <typename K, typename V, typename Hash = std::hash<K>, typename KeyEqual = std::equal_to<K>>
using HashMap = std::unordered_map<K, V, Hash, KeyEqual>;

#endif
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_UTILS_HASH_MAP_H_
