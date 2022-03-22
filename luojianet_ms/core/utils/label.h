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

#ifndef LUOJIANET_MS_CORE_UTILS_LABEL_H_
#define LUOJIANET_MS_CORE_UTILS_LABEL_H_
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "utils/hash_map.h"
#include "ir/anf.h"

namespace luojianet_ms {
namespace label_manage {
enum class TraceLabelType { kShortSymbol, kFullName, kWithUniqueId };
TraceLabelType GetGlobalTraceLabelType();
void SetGlobalTraceLabelType(TraceLabelType label_type);
std::string Label(const DebugInfoPtr &debug_info, TraceLabelType trace_type = TraceLabelType::kShortSymbol);
}  // namespace label_manage
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_UTILS_LABEL_H_
