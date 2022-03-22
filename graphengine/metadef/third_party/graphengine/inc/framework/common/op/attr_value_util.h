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

#include "graph/debug/ge_attr_define.h"
#include "proto/om.pb.h"

using domi::AttrDef;
using domi::AttrDef_ListValue;
using domi::ModelDef;
using domi::NamedAttrs;
using domi::OpDef;

namespace ge {
using AttrDefMap = ::google::protobuf::Map<::std::string, ::domi::AttrDef>;
using AttrDefPair = ::google::protobuf::MapPair<std::string, domi::AttrDef>;

GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, AttrDef &attr, OpDef *opdef);
// DEFINE_ADD_ATTR_VALUE
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const std::string &value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const char *value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttr(const char *key, const char *value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const uint32_t value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const int32_t value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const int64_t value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const float value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const double value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const bool value, AttrDefMap *attrs);

GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const AttrDef_ListValue &value, AttrDefMap *attrs);

// DEFINE_ADD_ATTR_VALUE
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const std::string &value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const char *value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttr(const char *key, const char *value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const uint32_t value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const int32_t value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const int64_t value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const float value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const double value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const bool value, OpDef *opdef);

GE_FUNC_VISIBILITY void AddOpAttr(const std::string &key, const AttrDef_ListValue &value, OpDef *opdef);

GE_FUNC_VISIBILITY void AddOpBytesAttr(const std::string &key, const void *value, size_t size, OpDef *opdef);

// DEFINE_ADD_ATTR_VALUE_LIST
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const double value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const float value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const uint32_t value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const int32_t value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const std::string value, AttrDefMap *attrs);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const double value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const float value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const uint32_t value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const int32_t value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const bool value, OpDef *opdef);
GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const int64_t value, OpDef *opdef);

GE_FUNC_VISIBILITY void AddOpAttrList(const std::string &key, const std::string &value, OpDef *opdef);

GE_FUNC_VISIBILITY bool GetOpAttr(const std::string &key, std::string *value, const OpDef *opdef);
GE_FUNC_VISIBILITY bool GetOpAttr(const std::string &key, int32_t *value, const OpDef *opdef);
GE_FUNC_VISIBILITY bool GetOpAttr(const std::string &key, int64_t *value, const OpDef *opdef);
GE_FUNC_VISIBILITY bool GetOpAttr(const std::string &key, uint32_t *value, const OpDef *opdef);
GE_FUNC_VISIBILITY bool GetOpAttr(const std::string &key, float *value, const OpDef *opdef);
GE_FUNC_VISIBILITY bool GetOpAttr(const std::string &key, double *value, const OpDef *opdef);
GE_FUNC_VISIBILITY bool GetOpAttr(const std::string &key, bool *value, const OpDef *opdef);
GE_FUNC_VISIBILITY bool GetOpAttr(const std::string &key, AttrDef_ListValue *value, const OpDef *opdef);

GE_FUNC_VISIBILITY uint32_t GetOpAttrListSize(const std::string &key, std::string value, const OpDef *opdef);
GE_FUNC_VISIBILITY uint32_t GetOpAttrListSize(const std::string &key, int32_t value, const OpDef *opdef);
GE_FUNC_VISIBILITY uint32_t GetOpAttrListSize(const std::string &key, int64_t value, const OpDef *opdef);
GE_FUNC_VISIBILITY uint32_t GetOpAttrListSize(const std::string &key, uint32_t value, const OpDef *opdef);
GE_FUNC_VISIBILITY uint32_t GetOpAttrListSize(const std::string &key, float value, const OpDef *opdef);
GE_FUNC_VISIBILITY uint32_t GetOpAttrListSize(const std::string &key, double value, const OpDef *opdef);
GE_FUNC_VISIBILITY uint32_t GetOpAttrListSize(const std::string &key, bool value, const OpDef *opdef);

GE_FUNC_VISIBILITY bool GetBytesAttr(const std::string &key, std::string *value, const OpDef *opdef);
GE_FUNC_VISIBILITY bool GetBytesAttr(const std::string &key, std::string *value, const ModelDef *model_def);

GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const std::string &value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const char *value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const char *key, const char *value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const uint32_t value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const int32_t value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const int64_t value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const float value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const double value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const bool value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const void *value, size_t size, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttr(const std::string &key, const AttrDef_ListValue &value, ModelDef *model_def);

GE_FUNC_VISIBILITY void AddModelAttrList(const std::string &key, const double value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttrList(const std::string &key, const float value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttrList(const std::string &key, const uint32_t value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttrList(const std::string &key, const int32_t value, ModelDef *model_def);
GE_FUNC_VISIBILITY void AddModelAttrList(const std::string &key, const std::string &value, ModelDef *model_def);

GE_FUNC_VISIBILITY bool GetModelAttr(const std::string &key, std::string *value, const ModelDef *model_def);
GE_FUNC_VISIBILITY bool GetModelAttr(const std::string &key, int32_t *value, const ModelDef *model_def);
GE_FUNC_VISIBILITY bool GetModelAttr(const std::string &key, int64_t *value, const ModelDef *model_def);
GE_FUNC_VISIBILITY bool GetModelAttr(const std::string &key, uint32_t *value, const ModelDef *model_def);
GE_FUNC_VISIBILITY bool GetModelAttr(const std::string &key, float *value, const ModelDef *model_def);
GE_FUNC_VISIBILITY bool GetModelAttr(const std::string &key, double *value, const ModelDef *model_def);
GE_FUNC_VISIBILITY bool GetModelAttr(const std::string &key, bool *value, const ModelDef *model_def);
GE_FUNC_VISIBILITY bool GetModelAttr(const std::string &key, AttrDef_ListValue *value, const ModelDef *model_def);

GE_FUNC_VISIBILITY bool HasOpAttr(const OpDef *opdef, const std::string &attr_name);

GE_FUNC_VISIBILITY void SetAttrDef(const std::string &value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrDef(const char *value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrDef(const uint32_t value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrDef(const int32_t value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrDef(const float value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrDef(const double value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrDef(const bool value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrList(const std::string &value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrList(const bool value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrList(const float value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrList(const double value, AttrDef *out);
GE_FUNC_VISIBILITY void SetAttrList(const uint32_t value, AttrDef *out);

GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, std::string *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, int32_t *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, int64_t *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, uint32_t *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, float *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, double *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, bool *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, AttrDef_ListValue *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, NamedAttrs *&value, AttrDefMap *attr);
GE_FUNC_VISIBILITY bool GetAttrDefValue(const std::string &key, const NamedAttrs *&value, const AttrDefMap &attr);

GE_FUNC_VISIBILITY bool GetAttrDefListValue(const std::string &key, int idx, int32_t *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefListValue(const std::string &key, int idx, uint32_t *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefListValue(const std::string &key, int idx, float *value, const AttrDefMap &attr);
GE_FUNC_VISIBILITY bool GetAttrDefListValue(const std::string &key, int idx, double *value, const AttrDefMap &attr);
}

#endif  // INC_FRAMEWORK_COMMON_OP_ATTR_VALUE_UTIL_H_
