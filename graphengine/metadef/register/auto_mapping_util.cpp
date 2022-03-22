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
#include "register/auto_mapping_util.h"
#include "graph/debug/ge_util.h"

namespace ge {

// Convert tensorflow property to ge property
bool AutoMappingUtil::FindAttrValue(const domi::tensorflow::NodeDef *nodeDef, const string &attr_name,
                                    domi::tensorflow::AttrValue &attr_value) {
  GE_CHECK_NOTNULL(nodeDef);
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue> &attr = nodeDef->attr();
  const google::protobuf::Map<std::string, domi::tensorflow::AttrValue>::const_iterator it = attr.find(attr_name);
  if (it != attr.end()) {
    attr_value = it->second;
    return true;
  }
  return false;
}

// Get the attribute shape of tensorflow
void AutoMappingUtil::ConvertShape(const domi::tensorflow::TensorShapeProto &shape,
                                   vector<int64_t>& shape_dims) {
  shape_dims.clear();
  if (!shape.unknown_rank()) {
    for (auto &dim : shape.dim()) {
      shape_dims.push_back(dim.size());
    }
  } else {
   shape_dims = ge::UNKNOWN_SHAPE;
  }
}

graphStatus AutoMappingUtil::ConvertTensor(const domi::tensorflow::TensorProto &tensor, ge::GeTensorPtr &weight) {
  weight = ComGraphMakeShared<ge::GeTensor>();
  if (weight == nullptr) {
    GE_LOGE("Weight is nullptr.");
    return GRAPH_FAILED;
  }
  domi::tensorflow::DataType tf_data_type = tensor.dtype();
  ge::DataType ge_data_type = domi::TensorAssign::ConvertTensorflowDataType(tf_data_type);
  if (domi::TensorAssign::SetGeTensorDataType(ge_data_type, weight) != domi::SUCCESS) {
    GE_LOGE("Set Ge tensor data type failed.");
    return GRAPH_FAILED;
  }
  if (domi::TensorAssign::SetGeTensor(tensor, weight) != domi::SUCCESS) {
    GE_LOGE("Set Ge tensor failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

void AutoMappingUtil::ConvertTensorList(const domi::tensorflow::AttrValue_ListValue &list,
                                        std::vector<ge::GeTensorPtr> &vec) {
  vec.clear();
  for (auto &tensor : list.tensor()) {
    ge::GeTensorPtr ge_tensor = nullptr;
    graphStatus ret = ConvertTensor(tensor, ge_tensor);
    if (ret != GRAPH_SUCCESS) {
      GE_LOGE("Convert tensor failed.");
      return;
    }
    vec.push_back(ge_tensor);
  }
}

void AutoMappingUtil::ConvertFunc(const domi::tensorflow::NameAttrList& tf_func,
                                  ge::NamedAttrs& ge_func) {
  ge_func.SetName(tf_func.name());
  auto& attrs = tf_func.attr();
  for (auto &item : attrs) {
    ConvertValue(item.first, item.second, ge_func);
  }
}

void AutoMappingUtil::ConvertDataTypeList(const domi::tensorflow::AttrValue_ListValue &list,
                                          std::vector<ge::DataType> &vec) {
  vec.clear();
  for (auto &e : list.type()) {
    ge::DataType ge_data_type = domi::TensorAssign::ConvertTensorflowDataType(static_cast<uint32_t>(e));
    vec.push_back(ge_data_type);
  }
}

void AutoMappingUtil::ConvertShapeList(const domi::tensorflow::AttrValue_ListValue &list,
                                       std::vector<vector<int64_t>> &vec) {
  vec.clear();
  for (const auto &e : list.shape()) {
    vector<int64_t> shape_dims;
    ConvertShape(e, shape_dims);
    vec.push_back(shape_dims);
  }
}

void AutoMappingUtil::ConvertFuncList(const domi::tensorflow::AttrValue_ListValue &list,
                                      std::vector<ge::NamedAttrs> &vec) {
  vec.clear();
  for (const auto &e : list.func()) {
    ge::NamedAttrs func;
    ConvertFunc(e, func);
    vec.push_back(func);
  }
}

} // namespace domi
