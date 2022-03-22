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

#ifndef INC_FRAMEWORK_COMMON_OP_ATTR_VALUE_UTIL_H_
#define INC_FRAMEWORK_COMMON_OP_ATTR_VALUE_UTIL_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <google/protobuf/map.h>
#include <unordered_map>
#include <string>
#include "external/graph/types.h"
#include "graph/debug/ge_attr_define.h"
#include "proto/om.pb.h"

namespace ge {
GE_FUNC_VISIBILITY void SetAttrDef(const std::string &value, domi::AttrDef *const out);
GE_FUNC_VISIBILITY void SetAttrDef(const char *value, domi::AttrDef *const out);
GE_FUNC_VISIBILITY void SetAttrDef(const uint32_t value, domi::AttrDef *const out);
GE_FUNC_VISIBILITY void SetAttrDef(const int32_t value, domi::AttrDef *const out);
GE_FUNC_VISIBILITY void SetAttrDef(const int64_t value, domi::AttrDef *const out);
GE_FUNC_VISIBILITY void SetAttrDef(const float32_t value, domi::AttrDef *const out);
GE_FUNC_VISIBILITY void SetAttrDef(const float64_t value, domi::AttrDef *const out);
GE_FUNC_VISIBILITY void SetAttrDef(const bool value, domi::AttrDef *const out);
}  // namespace ge
#endif  // INC_FRAMEWORK_COMMON_OP_ATTR_VALUE_UTIL_H_