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

#include "framework/common/op/attr_value_util.h"
#include "framework/common/debug/log.h"
#include "framework/common/util.h"
#include "external/register/register_types.h"

namespace ge {
#define DEFINE_SET_ATTR_VALUE_ONE(ARG_TYPE, FIELD)                        \
  FMK_FUNC_DEV_VISIBILITY void SetAttrDef(ARG_TYPE value, AttrDef *out) { \
    GE_CHECK_NOTNULL_JUST_RETURN(out);                                    \
    out->set_##FIELD(value);                                              \
  }

#define DEFINE_SET_ATTR_VALUE_LIST(ARG_TYPE, FIELD)                        \
  FMK_FUNC_DEV_VISIBILITY void SetAttrList(ARG_TYPE value, AttrDef *out) { \
    GE_CHECK_NOTNULL_JUST_RETURN(out);                                     \
    GE_CHECK_NOTNULL_JUST_RETURN(out->mutable_list());                     \
    out->mutable_list()->add_##FIELD(value);                               \
  }

DEFINE_SET_ATTR_VALUE_ONE(const std::string &, s);
DEFINE_SET_ATTR_VALUE_ONE(const char *, s);
DEFINE_SET_ATTR_VALUE_ONE(const uint32_t, u);
DEFINE_SET_ATTR_VALUE_ONE(const int32_t, i);
DEFINE_SET_ATTR_VALUE_ONE(const int64_t, i);
DEFINE_SET_ATTR_VALUE_ONE(const float, f);
DEFINE_SET_ATTR_VALUE_ONE(const double, f);
DEFINE_SET_ATTR_VALUE_ONE(const bool, b);
DEFINE_SET_ATTR_VALUE_LIST(float, f);
DEFINE_SET_ATTR_VALUE_LIST(double, f);
DEFINE_SET_ATTR_VALUE_LIST(uint32_t, u);
DEFINE_SET_ATTR_VALUE_LIST(int32_t, i);
DEFINE_SET_ATTR_VALUE_LIST(bool, b);
DEFINE_SET_ATTR_VALUE_LIST(int64_t, i);
DEFINE_SET_ATTR_VALUE_LIST(const std::string &, s);

#define ADD_TO_ATTR_MAP(KEY, VALUE, ATTR_MAP)  \
  do {                                         \
    GE_CHECK_NOTNULL_JUST_RETURN(ATTR_MAP);    \
    AttrDef out;                               \
    auto it = ATTR_MAP->find(KEY);             \
    if (it != ATTR_MAP->end()) {               \
      auto &attr_value = it->second;           \
      SetAttrDef(VALUE, &attr_value);          \
    } else {                                   \
      SetAttrDef(VALUE, &out);                 \
      ATTR_MAP->insert(AttrDefPair(KEY, out)); \
    }                                          \
  } while (0);

#define ADD_TO_ATTR_MAP_LIST(KEY, VALUE, ATTR_MAP) \
  do {                                             \
    GE_CHECK_NOTNULL_JUST_RETURN(ATTR_MAP);        \
    AttrDef out;                                   \
    auto it = ATTR_MAP->find(KEY);                 \
    if (it != ATTR_MAP->end()) {                   \
      auto &attr_value = it->second;               \
      SetAttrList(VALUE, &attr_value);             \
    } else {                                       \
      SetAttrList(VALUE, &out);                    \
      ATTR_MAP->insert(AttrDefPair(KEY, out));     \
    }                                              \
  } while (0);

#define DEFINE_ADD_ATTR_VALUE(KEY_TYPE, VALUE_TYPE)                            \
  void AddOpAttr(KEY_TYPE map_key, VALUE_TYPE value, OpDef *op_def) {          \
    GE_CHECK_NOTNULL_JUST_RETURN(op_def);                                      \
    auto attr = op_def->mutable_attr();                                        \
    ADD_TO_ATTR_MAP(map_key, value, attr)                                      \
  }                                                                            \
  void AddOpAttr(KEY_TYPE map_key, VALUE_TYPE value, AttrDefMap *attr_map) {   \
    ADD_TO_ATTR_MAP(map_key, value, attr_map)                                  \
  }                                                                            \
  void AddModelAttr(KEY_TYPE map_key, VALUE_TYPE value, ModelDef *model_def) { \
    GE_CHECK_NOTNULL_JUST_RETURN(model_def);                                   \
    auto attr = model_def->mutable_attr();                                     \
    ADD_TO_ATTR_MAP(map_key, value, attr)                                      \
  }

#define DEFINE_ADD_ATTR_VALUE_LIST(KEY_TYPE, VALUE_TYPE)                           \
  void AddOpAttrList(KEY_TYPE map_key, VALUE_TYPE value, OpDef *op_def) {          \
    GE_CHECK_NOTNULL_JUST_RETURN(op_def);                                          \
    auto attr = op_def->mutable_attr();                                            \
    ADD_TO_ATTR_MAP_LIST(map_key, value, attr)                                     \
  }                                                                                \
  void AddOpAttrList(KEY_TYPE map_key, VALUE_TYPE value, AttrDefMap *attr_map) {   \
      ADD_TO_ATTR_MAP_LIST(map_key, value, attr_map)} FMK_FUNC_DEV_VISIBILITY void \
  AddModelAttrList(KEY_TYPE map_key, VALUE_TYPE value, ModelDef *model_def) {      \
    GE_CHECK_NOTNULL_JUST_RETURN(model_def);                                       \
    auto attr = model_def->mutable_attr();                                         \
    ADD_TO_ATTR_MAP_LIST(map_key, value, attr)                                     \
  }

DEFINE_ADD_ATTR_VALUE(const std::string &, const std::string &);
DEFINE_ADD_ATTR_VALUE(const char *, const char *);
DEFINE_ADD_ATTR_VALUE(const std::string &, const char *);
DEFINE_ADD_ATTR_VALUE(const std::string &, const uint32_t);
DEFINE_ADD_ATTR_VALUE(const std::string &, const int32_t);
DEFINE_ADD_ATTR_VALUE(const std::string &, const int64_t);
DEFINE_ADD_ATTR_VALUE(const std::string &, const float);
DEFINE_ADD_ATTR_VALUE(const std::string &, const double);
DEFINE_ADD_ATTR_VALUE(const std::string &, const bool);
DEFINE_ADD_ATTR_VALUE_LIST(const std::string &, const uint32_t);
DEFINE_ADD_ATTR_VALUE_LIST(const std::string &, const float);
DEFINE_ADD_ATTR_VALUE_LIST(const std::string &, const double);
DEFINE_ADD_ATTR_VALUE_LIST(const std::string &, const int32_t);
DEFINE_ADD_ATTR_VALUE_LIST(const std::string &, const bool);
DEFINE_ADD_ATTR_VALUE_LIST(const std::string &, const int64_t);
DEFINE_ADD_ATTR_VALUE_LIST(const std::string &, const std::string &);

void AddOpAttr(const std::string &map_key, AttrDef &attr, OpDef *op_def) {
  GE_CHECK_NOTNULL_JUST_RETURN(op_def);
  GE_CHECK_NOTNULL_JUST_RETURN(op_def->mutable_attr());
  (void)op_def->mutable_attr()->insert(AttrDefPair(map_key, attr));
}

#define DEFINE_GET_ATTR_VALUE(ARG_TYPE_KEY, ARG_TYPE_VALUE, FIELD)                           \
  bool GetAttrDefValue(ARG_TYPE_KEY map_key, ARG_TYPE_VALUE value, const AttrDefMap &attr) { \
    auto it = attr.find(map_key);                                                            \
    if (it != attr.end()) {                                                                  \
      *value = it->second.FIELD();                                                           \
      return true;                                                                           \
    }                                                                                        \
    return false;                                                                            \
  }

#define DEFINE_GET_ATTR_POINT_REF(ARG_TYPE_KEY, ARG_TYPE_VALUE, FIELD)                   \
  bool GetAttrDefValue(ARG_TYPE_KEY map_key, ARG_TYPE_VALUE *&value, AttrDefMap *attr) { \
    GE_RT_FALSE_CHECK_NOTNULL(attr);                                                     \
    auto it = attr->find(map_key);                                                       \
    if (it != attr->end()) {                                                             \
      value = it->second.mutable_##FIELD();                                              \
      return true;                                                                       \
    }                                                                                    \
    return false;                                                                        \
  }

#define DEFINE_GET_ATTR_CONST_POINT_REF(ARG_TYPE_KEY, ARG_TYPE_VALUE, FIELD)                         \
  bool GetAttrDefValue(ARG_TYPE_KEY map_key, const ARG_TYPE_VALUE *&value, const AttrDefMap &attr) { \
    auto it = attr.find(map_key);                                                                    \
    if (it == attr.end()) {                                                                          \
      return false;                                                                                  \
    }                                                                                                \
                                                                                                     \
    value = &(it->second.FIELD());                                                                   \
    return true;                                                                                     \
  }

#define DEFINE_GET_BYTES_ATTR_VALUE(ARG_TYPE_KEY, ARG_TYPE_VALUE)                      \
  bool GetBytesValue(ARG_TYPE_KEY key, ARG_TYPE_VALUE value, const AttrDefMap &attr) { \
    GE_RT_FALSE_CHECK_NOTNULL(value);                                                  \
    auto it = attr.find(key);                                                          \
    if (it != attr.end()) {                                                            \
      *value = it->second.bt();                                                        \
      return true;                                                                     \
    }                                                                                  \
    return false;                                                                      \
  }

#define DEFINE_GET_ATTR_LIST_VALUE(ARG_TYPE_KEY, ARG_TYPE_VALUE, FIELD)                                 \
  FMK_FUNC_DEV_VISIBILITY bool GetAttrDefListValue(ARG_TYPE_KEY map_key, int idx, ARG_TYPE_VALUE value, \
                                                   const AttrDefMap &attr) {                            \
    auto it = attr.find(map_key);                                                                       \
    if (it == attr.end()) {                                                                             \
      return false;                                                                                     \
    }                                                                                                   \
                                                                                                        \
    const auto &list = it->second.list();                                                               \
    if (idx < 0 || idx > list.FIELD##_size() - 1) {                                                     \
      return false;                                                                                     \
    }                                                                                                   \
                                                                                                        \
    *value = list.FIELD(idx);                                                                           \
    return true;                                                                                        \
  }

DEFINE_GET_ATTR_VALUE(const std::string &, std::string *, s);
DEFINE_GET_ATTR_VALUE(const std::string &, int32_t *, i);
DEFINE_GET_ATTR_VALUE(const std::string &, int64_t *, i);
DEFINE_GET_ATTR_VALUE(const std::string &, uint32_t *, u);
DEFINE_GET_ATTR_VALUE(const std::string &, float *, f);
DEFINE_GET_ATTR_VALUE(const std::string &, double *, f);
DEFINE_GET_ATTR_VALUE(const std::string &, bool *, b);
DEFINE_GET_ATTR_VALUE(const std::string &, AttrDef_ListValue *, list);

DEFINE_GET_ATTR_LIST_VALUE(const std::string &, int32_t *, i);
DEFINE_GET_ATTR_LIST_VALUE(const std::string &, uint32_t *, u);
DEFINE_GET_ATTR_LIST_VALUE(const std::string &, float *, f);
DEFINE_GET_ATTR_LIST_VALUE(const std::string &, double *, f);

DEFINE_GET_ATTR_POINT_REF(const std::string &, NamedAttrs, func);
DEFINE_GET_ATTR_CONST_POINT_REF(const std::string &, NamedAttrs, func);

DEFINE_GET_BYTES_ATTR_VALUE(const std::string &, std::string *);

#define DEFINE_GET_OP_ATTR(ARG_TYPE_KEY, ARG_TYPE_VALUE)                                     \
  bool GetOpAttr(ARG_TYPE_KEY map_key, ARG_TYPE_VALUE value, const OpDef *op_def) {          \
    GE_RT_FALSE_CHECK_NOTNULL(op_def);                                                       \
    return GetAttrDefValue(map_key, value, op_def->attr());                                  \
  }                                                                                          \
  bool GetModelAttr(ARG_TYPE_KEY map_key, ARG_TYPE_VALUE value, const ModelDef *model_def) { \
    GE_RT_FALSE_CHECK_NOTNULL(model_def);                                                    \
    return GetAttrDefValue(map_key, value, model_def->attr());                               \
  }

DEFINE_GET_OP_ATTR(const std::string &, std::string *);
DEFINE_GET_OP_ATTR(const std::string &, int32_t *);
DEFINE_GET_OP_ATTR(const std::string &, int64_t *);
DEFINE_GET_OP_ATTR(const std::string &, uint32_t *);
DEFINE_GET_OP_ATTR(const std::string &, float *);
DEFINE_GET_OP_ATTR(const std::string &, double *);
DEFINE_GET_OP_ATTR(const std::string &, bool *);
DEFINE_GET_OP_ATTR(const std::string &, AttrDef_ListValue *);

#define DEFINE_GET_BT_ATTR(ARG_TYPE_KEY, ARG_TYPE_VALUE)                                                         \
  bool GetBytesAttr(ARG_TYPE_KEY key, ARG_TYPE_VALUE value, const OpDef *op_def) {                               \
    GE_RT_FALSE_CHECK_NOTNULL(op_def);                                                                           \
    return GetBytesValue(key, value, op_def->attr());                                                            \
  }                                                                                                              \
  FMK_FUNC_DEV_VISIBILITY bool GetBytesAttr(ARG_TYPE_KEY key, ARG_TYPE_VALUE value, const ModelDef *model_def) { \
    GE_RT_FALSE_CHECK_NOTNULL(model_def);                                                                        \
    return GetBytesValue(key, value, model_def->attr());                                                         \
  }

DEFINE_GET_BT_ATTR(const std::string &, std::string *);

bool HasOpAttr(const OpDef *op_def, const std::string &attr_name) {
  if (op_def == nullptr) {
    return false;
  }
  const AttrDefMap &attr = op_def->attr();

  const AttrDefMap::const_iterator it = attr.find(attr_name);
  if (it != attr.end()) {
    return true;
  }
  return false;
}

void AddModelAttr(const std::string &map_key, const void *value, size_t size, ModelDef *model_def) {
  if (model_def == nullptr) {
    return;
  }
  AttrDef out;
  auto attr = model_def->mutable_attr();
  auto it = attr->find(map_key);
  if (it != attr->end()) {
    auto &attr_value = it->second;
    attr_value.set_bt(value, size);
  } else {
    out.set_bt(value, size);
    attr->insert(AttrDefPair(map_key, out));
  }
}

void AddOpBytesAttr(const std::string &key, const void *value, size_t size, OpDef *op_def) {
  if (op_def == nullptr) {
    return;
  }
  AttrDef out;
  auto attr = op_def->mutable_attr();
  auto it = attr->find(key);
  if (it != attr->end()) {
    auto &attr_value = it->second;
    attr_value.set_bt(value, size);
  } else {
    out.set_bt(value, size);
    attr->insert(AttrDefPair(key, out));
  }
}

#define DEFINE_GET_ATTR_LIST_SIZE(ARG_TYPE_KEY, ARG_TYPE_VALUE, FIELD)                                              \
  FMK_FUNC_DEV_VISIBILITY uint32_t GetOpAttrListSize(ARG_TYPE_KEY key, ARG_TYPE_VALUE value, const OpDef *op_def) { \
    GE_CHK_BOOL_RET_STATUS_NOLOG(op_def != nullptr, 0);                                                             \
    const AttrDefMap &attr_map = op_def->attr();                                                                    \
    auto it = attr_map.find(key);                                                                                   \
    if (it == attr_map.end()) {                                                                                     \
      return 0;                                                                                                     \
    }                                                                                                               \
    const auto &list = it->second.list();                                                                           \
    return list.FIELD##_size();                                                                                     \
  }

DEFINE_GET_ATTR_LIST_SIZE(const std::string &, const std::string &, s);
DEFINE_GET_ATTR_LIST_SIZE(const std::string &, int32_t, i);
DEFINE_GET_ATTR_LIST_SIZE(const std::string &, int64_t, i);
DEFINE_GET_ATTR_LIST_SIZE(const std::string &, uint32_t, u);
DEFINE_GET_ATTR_LIST_SIZE(const std::string &, float, f);
DEFINE_GET_ATTR_LIST_SIZE(const std::string &, double, f);
DEFINE_GET_ATTR_LIST_SIZE(const std::string &, bool, b);
}  // namespace ge
