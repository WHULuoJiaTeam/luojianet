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

#include "graph/ge_attr_value.h"
#include <set>
#include <google/protobuf/text_format.h>
#include "external/graph/graph.h"
#include "graph/utils/attr_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/model_serialize.h"
#include "graph/ge_tensor_impl.h"
#include "graph/buffer_impl.h"
#include "graph/op_desc_impl.h"
#include "proto/ge_ir.pb.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/debug/ge_attr_define.h"
#include "debug/ge_log.h"
#include "debug/ge_util.h"
#include "graph/utils/tensor_utils.h"
#include "graph/serialization/attr_serializer_registry.h"
#include "graph/serialization/tensor_desc_serializer.h"


namespace ge {
void NamedAttrs::SetName(const std::string &name) {
  name_ = name;
}

std::string NamedAttrs::GetName() const {
  return name_;
}

AnyValue NamedAttrs::GetItem(const std::string &key) const {
  AnyValue value;
  (void) GetAttr(key, value);
  return value;
}

ProtoAttrMap &NamedAttrs::MutableAttrMap() {
  return attrs_;
}

ConstProtoAttrMap &NamedAttrs::GetAttrMap() const {
  return attrs_;
}

bool AttrUtils::HasAttr(ConstAttrHolderAdapter &&obj, const std::string &name) {
  if (!obj) {
    return false;
  }
  return obj->HasAttr(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj, const std::string &name, int32_t &value) {
  int64_t int64_val = 0;
  if (!AttrUtils::GetInt(std::move(obj), name, int64_val)) {
    return false;
  }
  if (int64_val > INT32_MAX) {
    REPORT_INNER_ERROR("E19999", "%ld int64_t value cannot cast to int32_t", int64_val);
    GELOGE(GRAPH_FAILED, "[Check][Param] %ld int64_t value cannot cast to int32_t", int64_val);
    return false;
  }
  value = static_cast<int32_t>(int64_val);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj, const std::string &name, uint32_t &value) {
  int64_t int64_val = 0;
  if (!AttrUtils::GetInt(std::move(obj), name, int64_val)) {
    return false;
  }
  if (int64_val > static_cast<int64_t>(UINT32_MAX)) {
    REPORT_INNER_ERROR("E19999", "%ld int64_t value cannot cast to uint32_t", int64_val);
    GELOGE(GRAPH_FAILED, "[Check][Param] %ld int64_t value cannot cast to uint32_t", int64_val);
    return false;
  }
  // 老版本中，只判断了上限，没有判断下限，因此小于0时，这里不会报错
  // 这里维持老版本的做法，在第一次上库做完后，补上小于0的判断
  value = static_cast<uint32_t>(int64_val);
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CloneOpDesc(const ConstOpDescPtr &org_op_desc) {
  if (org_op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "org_op_desc is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] org_op_desc is null");
    return nullptr;
  }
  std::shared_ptr<proto::OpDef> op_def;
  op_def = ComGraphMakeShared<proto::OpDef>();
  if (op_def == nullptr) {
    REPORT_CALL_ERROR("E19999", "create proto::OpDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][OpDef] proto::OpDef make shared failed");
    return nullptr;  // lint !e665
  }
  ModelSerializeImp imp;
  (void) imp.SerializeOpDesc(org_op_desc, op_def.get());

  imp.SetProtobufOwner(op_def);
  OpDescPtr op_desc = nullptr;
  GE_CHK_BOOL_EXEC(imp.UnserializeOpDesc(op_desc, *op_def),
                   REPORT_CALL_ERROR("E19999", "UnserializeOpDesc failed");
                   return op_desc, "[Call][UnserializeOpDesc] op_desc unserialize failed");
  op_desc->extAttrs_ = org_op_desc->extAttrs_;

  // This function may be called by some passes of fusion engine, in this condition, do not need these attribute
  if (op_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "op_desc impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Op desc impl is nullptr.");
    return nullptr;
  }
  if (!op_desc->impl_->input_name_idx_.empty()) {
    op_desc->impl_->input_name_idx_.clear();
  }
  if (!op_desc->impl_->output_name_idx_.empty()) {
    op_desc->impl_->output_name_idx_.clear();
  }
  if (!op_desc->impl_->optional_input_names_.empty()) {
    op_desc->impl_->optional_input_names_.clear();
  }

  return op_desc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr AttrUtils::CopyOpDesc(const ConstOpDescPtr &org_op_desc) {
  if (org_op_desc == nullptr || org_op_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "org_op_desc is null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] org_op_desc is null");
    return nullptr;
  }
  const std::shared_ptr<proto::OpDef> op_def = ComGraphMakeShared<proto::OpDef>();
  if (op_def == nullptr) {
    REPORT_CALL_ERROR("E19999", "create proto::OpDef failed");
    GELOGE(GRAPH_FAILED, "[Create][OpDef] proto::OpDef make shared failed");
    return nullptr;
  }
  ModelSerializeImp imp;
  (void) imp.SerializeOpDesc(org_op_desc, op_def.get());

  imp.SetProtobufOwner(op_def);
  OpDescPtr op_desc = nullptr;
  if (!imp.UnserializeOpDesc(op_desc, *op_def)) {
    REPORT_CALL_ERROR("E19999", "UnserializeOpDesc failed.");
    return nullptr;
  }

  op_desc->extAttrs_ = org_op_desc->extAttrs_;

  if (op_desc->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "op desc impl is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] op desc impl is null.");
    return nullptr;
  }
  op_desc->impl_->input_name_idx_.insert(org_op_desc->impl_->input_name_idx_.begin(),
                                         org_op_desc->impl_->input_name_idx_.end());
  op_desc->impl_->optional_input_names_.insert(org_op_desc->impl_->optional_input_names_.begin(),
                                               org_op_desc->impl_->optional_input_names_.end());
  op_desc->impl_->output_name_idx_.insert(org_op_desc->impl_->output_name_idx_.begin(),
                                          org_op_desc->impl_->output_name_idx_.end());

  op_desc->impl_->infer_func_ = org_op_desc->impl_->infer_func_;
  op_desc->impl_->infer_format_func_ = org_op_desc->impl_->infer_format_func_;
  op_desc->impl_->verifier_func_ = org_op_desc->impl_->verifier_func_;

  return op_desc;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListInt(AttrHolderAdapter &&obj, const std::string &name, const std::vector<int64_t> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<int64_t> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetInt(AttrHolderAdapter &&obj, const std::string &name,
                                                                      const int64_t &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetInt(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name, int64_t &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetFloat(AttrHolderAdapter &&obj, const std::string &name, const float32_t &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetFloat(ConstAttrHolderAdapter &&obj,
                                                                        const std::string &name, float32_t &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListFloat(AttrHolderAdapter &&obj, const std::string &name, const std::vector<float32_t> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListFloat(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<float32_t> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetBool(AttrHolderAdapter &&obj, const std::string &name,
                                                                       const bool &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetBool(ConstAttrHolderAdapter &&obj,
                                                                       const std::string &name, bool &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListBool(AttrHolderAdapter &&obj, const std::string &name, const std::vector<bool> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListBool(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<bool> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::SetStr(AttrHolderAdapter &&obj, const std::string &name,
                                                                      const std::string &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetStr(ConstAttrHolderAdapter &&obj,
                                                                      const std::string &name, std::string &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListStr(AttrHolderAdapter &&obj, const std::string &name, const std::vector<std::string> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListStr(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<std::string> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetTensorDesc(AttrHolderAdapter &&obj, const std::string &name, const GeTensorDesc &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetTensorDesc(ConstAttrHolderAdapter &&obj, const std::string &name, GeTensorDesc &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensorDesc(AttrHolderAdapter &&obj, const std::string &name,
                                  const std::vector<GeTensorDesc> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListTensorDesc(ConstAttrHolderAdapter &&obj,
                                  const std::string &name, std::vector<GeTensorDesc> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetNamedAttrs(AttrHolderAdapter &&obj, const std::string &name, const NamedAttrs &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetNamedAttrs(ConstAttrHolderAdapter &&obj, const std::string &name, NamedAttrs &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListNamedAttrs(AttrHolderAdapter &&obj, const std::string &name,
                                  const std::vector<NamedAttrs> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListNamedAttrs(ConstAttrHolderAdapter &&obj,
                                  const std::string &name, std::vector<NamedAttrs> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetDataType(AttrHolderAdapter &&obj, const std::string &name, const DataType &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool AttrUtils::GetDataType(ConstAttrHolderAdapter &&obj,
                                                                           const std::string &name, DataType &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListDataType(AttrHolderAdapter &&obj, const std::string &name, const std::vector<DataType> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListDataType(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<DataType> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListListInt(AttrHolderAdapter &&obj, const std::string &name,
                               const std::vector<std::vector<int64_t>> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListListInt(ConstAttrHolderAdapter &&obj, const std::string &name,
                               std::vector<std::vector<int64_t>> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListListFloat(AttrHolderAdapter &&obj, const std::string &name,
                                 const std::vector<std::vector<float32_t>> &value) {
  return SetAttrValue(obj->MutableAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListListFloat(ConstAttrHolderAdapter &&obj, const std::string &name,
                                 std::vector<std::vector<float32_t>> &value) {
  return GetAttrValue(obj->GetAttrMap(), name, value);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListInt(AttrHolderAdapter &&obj, const std::string &name, const std::vector<uint32_t> &value) {
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value.begin(), value.end()));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListInt(AttrUtils::AttrHolderAdapter &&obj, const std::string &name,
                           const std::vector<int32_t> &value) {
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value.begin(), value.end()));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListInt(AttrHolderAdapter &&obj, const std::string &name, std::initializer_list<int64_t> &&value) {
  return SetListInt(std::move(obj), name, std::vector<int64_t>(value));
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<int32_t> &value) {
  value.clear();
  std::vector<int64_t> int64_list;
  if (!GetListInt(std::move(obj), name, int64_list)) {
    return false;
  }

  for (size_t i = 0UL; i < int64_list.size(); ++i) {
    if (int64_list[i] > INT32_MAX) {
      REPORT_INNER_ERROR("E19999", "index %zu %ld int64_t value cannot cast to int32_t", i, int64_list[i]);
      GELOGE(GRAPH_FAILED, "[Check][Param] index %zu %ld int64_t value cannot cast to int32_t", i, int64_list[i]);
      return false;
    }
  }
  (void) value.insert(value.begin(), int64_list.begin(), int64_list.end());
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListInt(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<uint32_t> &value) {
  value.clear();
  std::vector<int64_t> int64_list;
  if (!GetListInt(std::move(obj), name, int64_list)) {
    return false;
  }

  for (size_t i = 0UL; i < int64_list.size(); ++i) {
    if (int64_list[i] > static_cast<int64_t>(UINT32_MAX)) {
      REPORT_INNER_ERROR("E19999", "index %zu %ld int64_t value cannot cast to uint32_t", i, int64_list[i]);
      GELOGE(GRAPH_FAILED, "[Check][Param] index %zu %ld int64_t value cannot cast to uint32_t", i, int64_list[i]);
      return false;
    }
    // 老版本中，只判断了上限，没有判断下限，因此小于0时，这里不会报错
    // 这里维持老版本的做法，在第一次上库做完后，补上小于0的判断
  }
  (void) value.insert(value.begin(), int64_list.begin(), int64_list.end());
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetTensor(AttrUtils::AttrHolderAdapter &&obj, const std::string &name, const GeTensor &value) {
  // 当前GeTensor的拷贝赋值、拷贝构造函数均不是深拷贝，因此无法使用默认的方法SetAttr
  if (!obj->MutableAttrMap().SetByName(name, GeTensor())) {
    return false;
  }
  const auto tensor = obj->MutableAttrMap().MutableGetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  TensorUtils::CopyTensor(value, *tensor);
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetTensor(AttrHolderAdapter &&obj, const std::string &name, const GeTensorPtr &value) {
  return SetTensor(std::move(obj), name, *value);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetTensor(AttrHolderAdapter &&obj, const std::string &name, const ConstGeTensorPtr &value) {
  return SetTensor(std::move(obj), name, *value);
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensor(AttrUtils::AttrHolderAdapter &&obj, const std::string &name,
                              const std::vector<GeTensor> &value) {
  std::vector<GeTensor> tensors(value.size());
  if (!obj->MutableAttrMap().SetByName(name, tensors)) {
    return false;
  }
  const auto attr_tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (attr_tensors == nullptr) {
    return false;
  }
  for (size_t i = 0UL; i < value.size(); ++i) {
    TensorUtils::CopyTensor(value[i], (*attr_tensors)[i]);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj, const std::string &name,
                              const std::vector<GeTensorPtr> &value) {
  std::vector<ConstGeTensorPtr> tensors(value.size());
  (void) std::copy(value.begin(), value.end(), tensors.begin());
  return SetListTensor(std::move(obj), name, tensors);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj, const std::string &name,
                              const std::vector<ConstGeTensorPtr> &value) {
  std::vector<GeTensor> tensors(value.size());
  if (!obj->MutableAttrMap().SetByName(name, tensors)) {
    return false;
  }
  const auto attr_tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (attr_tensors == nullptr) {
    return false;
  }
  for (size_t i = 0UL; i < value.size(); ++i) {
    TensorUtils::CopyTensor(*(value[i]), (*attr_tensors)[i]);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListTensor(AttrHolderAdapter &&obj, const std::string &name,
                              std::initializer_list<ConstGeTensorPtr> &&value) {
  return SetListTensor(std::move(obj), name, std::vector<ConstGeTensorPtr>(value));
}

// 所有权UT测试，不能把属性上的GeTensor给错误释放了
// 而且这里的行为与老版本是不一样的，老版本中，即使属性的owner生命周期结束析构了，通过本接口获取的value仍然是可用的
// 但是新接口中，owner没有转移，owner析构后，value指向的内存就被释放了，这里需要排查
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::MutableTensor(AttrHolderAdapter &&obj, const std::string &name, GeTensorPtr &value) {
  const auto tensor = obj->MutableAttrMap().MutableGetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  value = std::shared_ptr<GeTensor>(tensor, [](const GeTensor *const ptr) { (void) ptr; });
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetTensor(ConstAttrHolderAdapter &&obj, const std::string &name, ConstGeTensorPtr &value) {
  const auto tensor = obj->GetAttrMap().GetByName<GeTensor>(name);
  if (tensor == nullptr) {
    return false;
  }
  value = std::shared_ptr<const GeTensor>(tensor, [](const GeTensor *const ptr) { (void) ptr; });
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListTensor(ConstAttrHolderAdapter &&obj, const std::string &name,
                              std::vector<ConstGeTensorPtr> &value) {
  const auto tensors = obj->GetAttrMap().GetByName<std::vector<GeTensor>>(name);
  if (tensors == nullptr) {
    return false;
  }
  value.resize(tensors->size());
  for (size_t i = 0UL; i < tensors->size(); ++i) {
    value[i] = std::shared_ptr<const GeTensor>(&(*tensors)[i], [](const GeTensor *const ptr) { (void) ptr; });
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::MutableListTensor(AttrHolderAdapter &&obj, const std::string &name, std::vector<GeTensorPtr> &value) {
  const auto tensors = obj->MutableAttrMap().MutableGetByName<std::vector<GeTensor>>(name);
  if (tensors == nullptr) {
    return false;
  }
  value.resize(tensors->size());
  for (size_t i = 0UL; i < tensors->size(); ++i) {
    value[i] = std::shared_ptr<GeTensor>(&(*tensors)[i], [](const GeTensor *const ptr) { (void) ptr; });
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetGraph(AttrUtils::AttrHolderAdapter &&obj, const std::string &name, const ComputeGraphPtr &value) {
  proto::GraphDef *graph_def = SetAndGetAttrValue(obj->MutableAttrMap(), name, proto::GraphDef());
  if (graph_def == nullptr) {
    return false;
  }
  ModelSerializeImp imp;
  if (!imp.SerializeGraph(value, graph_def)) {
    REPORT_CALL_ERROR("E19999", "SerializeGraph failed when add ComputeGraph to attr %s", name.c_str());
    GELOGE(GRAPH_FAILED, "[Serialize][Graph] Failed when add ComputeGraph to attr %s", name.c_str());
    (void) obj->MutableAttrMap().Delete(name);
    return false;
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListGraph(AttrUtils::AttrHolderAdapter &&obj, const std::string &name,
                             const std::vector<ComputeGraphPtr> &value) {
  std::vector<proto::GraphDef> graphs(value.size());
  if (!obj->MutableAttrMap().SetByName(name, graphs)) {
    return false;
  }
  const auto attr_graphs = obj->MutableAttrMap().MutableGetByName<std::vector<proto::GraphDef>>(name);
  if (attr_graphs == nullptr) {
    return false;
  }
  for (size_t i = 0UL; i < value.size(); ++i) {
    ModelSerializeImp imp;
    if (!imp.SerializeGraph(value[i], &attr_graphs->at(i))) {
          REPORT_CALL_ERROR("E19999", "SerializeGraph failed when add ComputeGraph to attr %s", name.c_str());
      GELOGE(GRAPH_FAILED, "[Serialize][Graph] Failed when add ComputeGraph to attr %s", name.c_str());
      (void) obj->MutableAttrMap().Delete(name);
      return false;
    }
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetGraph(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &name, ComputeGraphPtr &value) {
  const auto attr_graph_def = obj->GetAttrMap().GetByName<proto::GraphDef>(name);
  if (attr_graph_def == nullptr) {
    return false;
  }
  // 这里延续了老代码实现，先拷贝构造一个ComputeGraph，然后做反序列化，感觉直接把attr_graph_def传进去应该就可以了?
  // 下一步对这里做整改，直接传入attr_graph_def，避免这一次拷贝
  const auto graph_def = ComGraphMakeShared<proto::GraphDef>(*attr_graph_def);
  if (graph_def == nullptr) {
    REPORT_CALL_ERROR("E19999", "create proto::GraphDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][GraphDef] proto::GraphDef make shared failed");
    return false;
  }

  ModelSerializeImp imp;
  imp.SetProtobufOwner(graph_def);
  if (!imp.UnserializeGraph(value, *graph_def)) {
    REPORT_CALL_ERROR("E19999", "UnserializeGraph failed when get attr ComputeGraph by name %s", name.c_str());
    GELOGE(GRAPH_FAILED, "[Unserialize][Graph] Failed when get attr ComputeGraph by name %s", name.c_str());
    return false;
  }

  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListGraph(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &name,
                             std::vector<ComputeGraphPtr> &value) {
  const auto graph_defs = obj->GetAttrMap().GetByName<std::vector<proto::GraphDef>>(name);
  if (graph_defs == nullptr) {
    return false;
  }

  value.resize(graph_defs->size());
  for (size_t i = 0UL; i < graph_defs->size(); ++i) {
    std::shared_ptr<proto::GraphDef> graph_def;
    graph_def = ComGraphMakeShared<proto::GraphDef>(graph_defs->at(i));
    if (graph_def == nullptr) {
      REPORT_CALL_ERROR("E19999", "create proto::GraphDef failed.");
      GELOGE(GRAPH_FAILED, "[Create][GraphDef] proto::GraphDef make shared failed");
      graph_def = nullptr;
      return false;  // lint !e665
    } else {
      ComputeGraphPtr graph = nullptr;
      ModelSerializeImp imp;
      imp.SetProtobufOwner(static_cast<const ProtoMsgOwner &>(graph_def));
      if (!imp.UnserializeGraph(graph, *graph_def)) {
        REPORT_CALL_ERROR("E19999", "UnserializeGraph failed.");
        GELOGE(GRAPH_FAILED, "[Unserialize][Graph] Failed");
        return false;
      }  // lint !e514
      value[i] = graph;
    }
  }
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetBytes(AttrUtils::AttrHolderAdapter &&obj, const std::string &name, const Buffer &value) {
  const auto buffer = SetAndGetAttrValue(obj->MutableAttrMap(), name, Buffer());
  if (buffer == nullptr) {
    return false;
  }
  BufferUtils::CopyFrom(value, *buffer);
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetBytes(ConstAttrHolderAdapter &&obj, const std::string &name, Buffer &value) {
  const auto buffer = obj->GetAttrMap().GetByName<Buffer>(name);
  if (buffer == nullptr) {
    return false;
  }
  BufferUtils::CopyFrom(*buffer, value);
  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetListBytes(AttrUtils::AttrHolderAdapter &&obj, const std::string &name,
                             const std::vector<Buffer> &value) {
  std::vector<Buffer> buffers(value.size());
  const auto attr_buffers = SetAndGetAttrValue(obj->MutableAttrMap(), name, buffers);
  if (attr_buffers == nullptr) {
    return false;
  }

  for (size_t i = 0UL; i < value.size(); ++i) {
    BufferUtils::CopyFrom(value[i], (*attr_buffers)[i]);
  }

  return true;
}
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetListBytes(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &name,
                             std::vector<Buffer> &value) {
  const auto buffers = obj->GetAttrMap().GetByName<std::vector<Buffer>>(name);
  if (buffers == nullptr) {
    return false;
  }
  value.resize(buffers->size());
  for (size_t i = 0UL; i < buffers->size(); ++i) {
    BufferUtils::CopyFrom(buffers->at(i), value[i]);
  }
  return true;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetZeroCopyBytes(AttrHolderAdapter &&obj, const std::string &name, Buffer &&buffer) {
  // Value will be shared
  return SetAttrValue(obj->MutableAttrMap(), name, std::move(buffer));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetZeroCopyBytes(ConstAttrHolderAdapter &&obj, const std::string &name, Buffer &buffer) {
  // Value will be shared
  return GetAttrValue<Buffer>(obj->GetAttrMap(), name, buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::SetZeroCopyListBytes(AttrHolderAdapter &&obj, const std::string &name,
                                     std::vector<Buffer> &list_buffer) {
  // Value will be shared
  return SetAttrValue(obj->MutableAttrMap(), name, list_buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
bool AttrUtils::GetZeroCopyListBytes(ConstAttrHolderAdapter &&obj, const std::string &name,
                                     std::vector<Buffer> &list_buffer) {
  // Value will be shared
  return GetAttrValue<std::vector<Buffer>>(obj->GetAttrMap(), name, list_buffer);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
std::map<std::string, AnyValue> AttrUtils::GetAllAttrs(ConstAttrHolderAdapter &&obj) {
  const auto holder = obj.get();
  if (holder == nullptr) {
    const std::map<std::string, AnyValue> empty;
    return empty;
  }
  return holder->GetAllAttrs();
}


std::string AttrUtils::GetAttrsStrAfterRid(ConstAttrHolderAdapter &&obj,
                                           const std::set<std::string> &un_compute_attrs) {

  const std::map<std::string, AnyValue> attr_map = GetAllAttrs(std::move(obj));
  if (attr_map.empty()) {
    return "";
  }
  std::map<std::string, std::string> ordered_attrs;
  for (auto &attr : attr_map) {
    proto::AttrDef attr_def;
    auto *serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr.second.GetValueTypeId());
    if ((serializer == nullptr) || (serializer->Serialize(attr.second, attr_def) != GRAPH_SUCCESS)) {
      ordered_attrs[attr.first] = "";
      continue;
    }

    ordered_attrs[attr.first] = attr_def.SerializeAsString();
  }

  std::stringstream ss;
  for (auto &attr : ordered_attrs) {
    if (un_compute_attrs.find(attr.first) != un_compute_attrs.end()) {
      continue;
    }
    ss << attr.first << ":" << attr.second << ";";
  }
  return ss.str();
}
std::string AttrUtils::GetAllAttrsStr(ConstAttrHolderAdapter &&obj) {
  const auto attr_map = GetAllAttrs(std::move(obj));
  if (attr_map.empty()) {
    return "";
  }
  std::map<std::string, std::string> ordered_attrs;
  for (auto &attr : attr_map) {
    proto::AttrDef attr_def;
    auto *serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr.second.GetValueTypeId());
    if ((serializer == nullptr) || (serializer->Serialize(attr.second, attr_def) != GRAPH_SUCCESS)) {
      ordered_attrs[attr.first] = "";
      continue;
    }

    if (attr_def.has_t()) {
      // print tensor desc message as an ordered string.
      std::string ordered_tensor_desc;
      (void) google::protobuf::TextFormat::PrintToString(attr_def.t().desc(), &ordered_tensor_desc);
      ordered_attrs[attr.first] = ordered_tensor_desc + attr_def.t().data();
    } else if (attr_def.has_td()) {
      // print tensor desc message as an ordered string.
      std::string ordered_attr;
      (void) google::protobuf::TextFormat::PrintToString(attr_def.td(), &ordered_attr);
      ordered_attrs[attr.first] = ordered_attr;
    } else {
      ordered_attrs[attr.first] = attr_def.SerializeAsString();
    }
  }

  std::stringstream ss;
  for (auto &attr : ordered_attrs) {
    ss << attr.first << ":" << attr.second << ";";
  }
  return ss.str();
}
}  // namespace ge
