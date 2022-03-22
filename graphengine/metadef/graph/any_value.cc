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

#include "graph/any_value.h"
#include <vector>
#include <unordered_map>

#include "external/graph/types.h"
#include "graph/ge_attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_tensor.h"
#include "graph/buffer.h"

namespace ge {
template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<bool>() {
  return reinterpret_cast<TypeId>(1);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::string>() {
  return reinterpret_cast<TypeId>(2);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<float>() {
  return reinterpret_cast<TypeId>(3);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<int64_t>() {
  return reinterpret_cast<TypeId>(4);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<GeTensorDesc>() {
  return reinterpret_cast<TypeId>(5);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<GeTensor>() {
  return reinterpret_cast<TypeId>(6);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<Buffer>() {
  return reinterpret_cast<TypeId>(7);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<proto::GraphDef>() {
  return reinterpret_cast<TypeId>(8);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<NamedAttrs>() {
  return reinterpret_cast<TypeId>(9);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::vector<int64_t>>>() {
  return reinterpret_cast<TypeId>(10);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<DataType>() {
  return reinterpret_cast<TypeId>(11);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::vector<float>>>() {
  return reinterpret_cast<TypeId>(12);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<std::string>>() {
  return reinterpret_cast<TypeId>(13);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<float>>() {
  return reinterpret_cast<TypeId>(14);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<bool>>() {
  return reinterpret_cast<TypeId>(15);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<int64_t>>() {
  return reinterpret_cast<TypeId>(16);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<GeTensorDesc>>() {
  return reinterpret_cast<TypeId>(17);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<GeTensor>>() {
  return reinterpret_cast<TypeId>(18);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<Buffer>>() {
  return reinterpret_cast<TypeId>(19);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<proto::GraphDef>>() {
  return reinterpret_cast<TypeId>(20);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<NamedAttrs>>() {
  return reinterpret_cast<TypeId>(21);
}

template<>
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY TypeId GetTypeId<std::vector<DataType>>() {
  return reinterpret_cast<TypeId>(22);
}

namespace {
std::unordered_map<TypeId, AnyValue::ValueType> type_ids_to_value_type = {
    {nullptr, AnyValue::VT_NONE},
    {GetTypeId<std::string>(), AnyValue::VT_STRING},
    {GetTypeId<float>(), AnyValue::VT_FLOAT},
    {GetTypeId<bool>(), AnyValue::VT_BOOL},
    {GetTypeId<int64_t>(), AnyValue::VT_INT},
    {GetTypeId<GeTensorDesc>(), AnyValue::VT_TENSOR_DESC},
    {GetTypeId<GeTensor>(), AnyValue::VT_TENSOR},
    {GetTypeId<Buffer>(), AnyValue::VT_BYTES},
    {GetTypeId<proto::GraphDef>(), AnyValue::VT_GRAPH},
    {GetTypeId<NamedAttrs>(), AnyValue::VT_NAMED_ATTRS},
    {GetTypeId<std::vector<std::vector<int64_t>>>(), AnyValue::VT_LIST_LIST_INT},
    {GetTypeId<DataType>(), AnyValue::VT_DATA_TYPE},
    {GetTypeId<std::vector<std::vector<float>>>(), AnyValue::VT_LIST_LIST_FLOAT},
    {GetTypeId<std::vector<std::string>>(), AnyValue::VT_LIST_STRING},
    {GetTypeId<std::vector<float>>(), AnyValue::VT_LIST_FLOAT},
    {GetTypeId<std::vector<bool>>(), AnyValue::VT_LIST_BOOL},
    {GetTypeId<std::vector<int64_t>>(), AnyValue::VT_LIST_INT},
    {GetTypeId<std::vector<GeTensorDesc>>(), AnyValue::VT_LIST_TENSOR_DESC},
    {GetTypeId<std::vector<GeTensor>>(), AnyValue::VT_LIST_TENSOR},
    {GetTypeId<std::vector<Buffer>>(), AnyValue::VT_LIST_BYTES},
    {GetTypeId<std::vector<proto::GraphDef>>(), AnyValue::VT_LIST_GRAPH},
    {GetTypeId<std::vector<NamedAttrs>>(), AnyValue::VT_LIST_NAMED_ATTRS},
    {GetTypeId<std::vector<DataType>>(), AnyValue::VT_LIST_DATA_TYPE},
};
}  // namespace

void AnyValue::Swap(AnyValue &other) noexcept {
  AnyValue tmp;
  if (!other.IsEmpty()) {
    other.operate_(OperateType::kOpMove, &other, &tmp);
  }

  other.Clear();
  if (!IsEmpty()) {
    operate_(OperateType::kOpMove, this, &other);
  }

  Clear();
  if (!tmp.IsEmpty()) {
    tmp.operate_(OperateType::kOpMove, &tmp, this);
  }
}

AnyValue::AnyValue(AnyValue &&other) noexcept {
  if (!other.IsEmpty()) {
    other.operate_(OperateType::kOpMove, &other, this);
  }
}
AnyValue &AnyValue::operator=(AnyValue &&other) noexcept {
  Clear();
  if (!other.IsEmpty()) {
    other.operate_(OperateType::kOpMove, &other, this);
  }
  return *this;
}
AnyValue &AnyValue::operator=(const AnyValue &other) {
  Clear();
  if (!other.IsEmpty()) {
    other.operate_(OperateType::kOpClone, &other, this);
  }
  return *this;
}
TypeId AnyValue::GetValueTypeId() const noexcept {
  TypeId vt{kInvalidTypeId};
  if (!IsEmpty()) {
    operate_(OperateType::kGetTypeId, this, &vt);
  }
  return vt;
}
AnyValue::ValueType AnyValue::GetValueType() const noexcept {
  auto vt = GetValueTypeId();
  auto iter = type_ids_to_value_type.find(vt);
  if (iter == type_ids_to_value_type.end()) {
    return AnyValue::VT_NONE;
  }
  return iter->second;
}
AnyValue AnyValue::Copy() const {
  AnyValue av(*this);
  return av;
}
const void *AnyValue::GetAddr() const {
  void *addr = nullptr;
  operate_(OperateType::kOpGetAddr, this, &addr);
  return addr;
}
}  // namespace ge