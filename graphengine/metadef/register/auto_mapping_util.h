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

#ifndef COMMON_AUTO_MAPPING_UTIL_H_
#define COMMON_AUTO_MAPPING_UTIL_H_

#include "framework/common/debug/ge_log.h"
#include "proto/tensorflow/attr_value.pb.h"
#include "proto/tensorflow/node_def.pb.h"
#include "graph/ge_tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_log.h"
#include "register/register_error_codes.h"
#include "register/tensor_assign.h"

namespace ge {

class AutoMappingUtil {
public:
  static bool FindAttrValue(const domi::tensorflow::NodeDef *nodeDef, const string &attr_name,
                          domi::tensorflow::AttrValue &attr_value);
  static void ConvertShape(const domi::tensorflow::TensorShapeProto &shape, vector<int64_t>& shape_dims);
  static graphStatus ConvertTensor(const domi::tensorflow::TensorProto &tensor, ge::GeTensorPtr &weight);
  static void ConvertFunc(const domi::tensorflow::NameAttrList& tf_func, ge::NamedAttrs& ge_func);

  static void ConvertDataTypeList(const domi::tensorflow::AttrValue_ListValue &list,
                                  std::vector<ge::DataType> &vec);
  static void ConvertShapeList(const domi::tensorflow::AttrValue_ListValue &list,
                               std::vector<vector<int64_t>> &vec);
  static void ConvertTensorList(const domi::tensorflow::AttrValue_ListValue &list,
                                std::vector<ge::GeTensorPtr> &vec);
  static void ConvertFuncList(const domi::tensorflow::AttrValue_ListValue &list,
                              std::vector<ge::NamedAttrs> &vec);

  // Get the attribute list list of tensorflow and save it to obj according to the key
  template<typename T>
  static void ConvertList(const std::string &key, const domi::tensorflow::AttrValue &value, T &obj) {
    const domi::tensorflow::AttrValue_ListValue &list = value.list();
    if (list.s_size() > 0) {
      vector<std::string> vec;
      for (auto e : list.s()) {
        vec.push_back(e);
      }
      (void)ge::AttrUtils::SetListStr(obj, key, vec);
    } else if (list.i_size() > 0) {
      vector<int64_t> vec;
      for (auto e : list.i()) {
        vec.push_back(e);
      }
      (void)ge::AttrUtils::SetListInt(obj, key, vec);
    } else if (list.f_size() > 0) {
      vector<float> vec;
      for (auto e : list.f()) {
        vec.push_back(e);
      }
      (void)ge::AttrUtils::SetListFloat(obj, key, vec);
    } else if (list.b_size() > 0) {
      vector<bool> vec;
      for (auto e : list.b()) {
        vec.push_back(e);
      }
      (void)ge::AttrUtils::SetListBool(obj, key, vec);
    } else if (list.type_size() > 0) {
      vector<ge::DataType> vec;
      ConvertDataTypeList(list, vec);
      (void)ge::AttrUtils::SetListDataType(obj, key, vec);
    } else if (list.shape_size() > 0) {
      vector<vector<int64_t>> shape_dims_vec;
      ConvertShapeList(list, shape_dims_vec);
      (void)ge::AttrUtils::SetListListInt(obj, key, shape_dims_vec);
    } else if (list.tensor_size() > 0) {
      vector<ge::GeTensorPtr> vec;
      ConvertTensorList(list, vec);
      (void)ge::AttrUtils::SetListTensor(obj, key, vec);
    } else if (list.func_size() > 0) {
      vector<ge::NamedAttrs> vec;
      ConvertFuncList(list, vec);
      (void)ge::AttrUtils::SetListNamedAttrs(obj, key, vec);
    } else {
      GELOGD("The list has no value, key is %s.", key.c_str());
    }
  }

  // According to the property type of tensorflow, set it to the corresponding property of obj
  template<typename T>
  static void ConvertValue(const std::string &key, const domi::tensorflow::AttrValue &value, T &obj) {
    switch (value.value_case()) {
      case domi::tensorflow::AttrValue::kS:
        (void)ge::AttrUtils::SetStr(obj, key, value.s());
        break;
      case domi::tensorflow::AttrValue::kI:
        (void)ge::AttrUtils::SetInt(obj, key, static_cast<int64_t>(value.i()));
        break;
      case domi::tensorflow::AttrValue::kF:
        (void)ge::AttrUtils::SetFloat(obj, key, static_cast<float>(value.f()));
        break;
      case domi::tensorflow::AttrValue::kB:
        (void)ge::AttrUtils::SetBool(obj, key, static_cast<bool>(value.b()));
        break;
      case domi::tensorflow::AttrValue::kType: {
        ge::DataType ge_data_type = domi::TensorAssign::ConvertTensorflowDataType(static_cast<uint32_t>(value.type()));
        (void)ge::AttrUtils::SetDataType(obj, key, ge_data_type);
        break;
      }
      case domi::tensorflow::AttrValue::kList:
        ConvertList(key, value, obj);
        break;
      case domi::tensorflow::AttrValue::kShape: {
        vector<int64_t> shape_dims;
        ConvertShape(value.shape(), shape_dims);
        (void)ge::AttrUtils::SetListInt(obj, key, shape_dims);
        break;
      }
      case domi::tensorflow::AttrValue::kTensor: {
        ge::GeTensorPtr ge_tensor = nullptr;
        graphStatus ret = ConvertTensor(value.tensor(), ge_tensor);
        if (ret != GRAPH_SUCCESS) {
          GE_LOGE("Convert ge tensor failed, key is %s.", key.c_str());
          return;
        }
        (void)ge::AttrUtils::SetTensor(obj, key, ge_tensor);
        break;
      }
      case domi::tensorflow::AttrValue::kFunc: {
        ge::NamedAttrs func;
        ConvertFunc(value.func(), func);
        (void)ge::AttrUtils::SetNamedAttrs(obj, key, func);
        break;
      }
      case domi::tensorflow::AttrValue::kPlaceholder:
        (void)ge::AttrUtils::SetStr(obj, key, value.placeholder());
        break;
      case domi::tensorflow::AttrValue::VALUE_NOT_SET:
        GELOGD("the attr value of %s is not set.", key.c_str());
        break;
      default:
        GE_LOGE("the attr value type(%d) is invalid.", static_cast<int>(value.value_case()));
        break;
    }
  }

template<typename T>
static void CopyAttrValue(const std::string &key, const ge::GeAttrValue &value, T &obj_src, T &obj) {
  GeAttrValue::ValueType value_type = value.GetValueType();
  bool is_one_type = value_type == GeAttrValue::VT_STRING || value_type == GeAttrValue::VT_INT ||
                     value_type == GeAttrValue::VT_FLOAT || value_type == GeAttrValue::VT_BOOL ||
                     value_type == GeAttrValue::VT_TENSOR || value_type == GeAttrValue::VT_NAMED_ATTRS ||
                     value_type == GeAttrValue::VT_DATA_TYPE;
  if (is_one_type) {
    switch (value_type) {
#define CASE_ATTR_VALUE_TYPE(GeValueType, ValueType, FuncName)    \
      case GeAttrValue::VT_##GeValueType: {                       \
        ValueType val;                                            \
        (void) ge::AttrUtils::Get##FuncName(obj_src, key, val);   \
        (void) ge::AttrUtils::Set##FuncName(obj, key, val);       \
        break;                                                    \
      }
      CASE_ATTR_VALUE_TYPE(STRING, string, Str);
      CASE_ATTR_VALUE_TYPE(INT, int64_t, Int);
      CASE_ATTR_VALUE_TYPE(FLOAT, float, Float);
      CASE_ATTR_VALUE_TYPE(BOOL, bool, Bool);
      CASE_ATTR_VALUE_TYPE(TENSOR, ConstGeTensorPtr, Tensor);
      CASE_ATTR_VALUE_TYPE(NAMED_ATTRS, ge::NamedAttrs, NamedAttrs);
      CASE_ATTR_VALUE_TYPE(DATA_TYPE, ge::DataType, DataType);
#undef CASE_ATTR_VALUE_TYPE
      default:
        break;
    }
  } else {
    switch (value_type) {
#define CASE_ATTR_VALUE_TYPE_LIST(GeValueType, ValueType, FuncName)   \
      case GeAttrValue::VT_LIST_##GeValueType: {                      \
        vector<ValueType> value;                                      \
        (void) ge::AttrUtils::GetList##FuncName(obj_src, key, value); \
        (void) ge::AttrUtils::SetList##FuncName(obj, key, value);     \
        break;                                                        \
      }
      CASE_ATTR_VALUE_TYPE_LIST(STRING, string, Str);
      CASE_ATTR_VALUE_TYPE_LIST(INT, int64_t, Int);
      CASE_ATTR_VALUE_TYPE_LIST(FLOAT, float, Float);
      CASE_ATTR_VALUE_TYPE_LIST(BOOL, bool, Bool);
      CASE_ATTR_VALUE_TYPE_LIST(TENSOR, ConstGeTensorPtr, Tensor);
      CASE_ATTR_VALUE_TYPE_LIST(NAMED_ATTRS, ge::NamedAttrs, NamedAttrs);
      CASE_ATTR_VALUE_TYPE_LIST(DATA_TYPE, ge::DataType, DataType);
      CASE_ATTR_VALUE_TYPE_LIST(LIST_INT, vector<int64_t>, ListInt);
#undef CASE_ATTR_VALUE_TYPE_LIST
      default:
        GELOGW("[Copy][AttrValue] Attr value type %d is not supported.", static_cast<int>(value_type));
        break;
    }
  }
}
};
} // namespace domi
#endif  // COMMON_AUTO_MAPPING_UTIL_H_
