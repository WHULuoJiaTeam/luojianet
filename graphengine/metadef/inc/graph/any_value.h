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

#ifndef EXECUTE_GRAPH_ANY_VALUE_H
#define EXECUTE_GRAPH_ANY_VALUE_H
#include <functional>
#include <memory>
#include <vector>

#include "graph/types.h"
#include "type_utils.h"
#include "external/graph/ge_error_codes.h"
namespace ge {
class Buffer;
class GeTensor;
class GeTensorDesc;
class ComputeGraph;
class NamedAttrs;
using GeTensorPtr = std::shared_ptr<GeTensor>;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
class AnyValue {
 public:
  // 后续删除，新增代码请勿使用这堆using
  using INT = int64_t;
  using FLOAT = float;
  using BOOL = bool;
  using STR = std::string;
  using TENSOR = GeTensorPtr;
  using TENSOR_DESC = GeTensorDesc;
  using GRAPH = ComputeGraphPtr;
  using BYTES = Buffer;
  using NAMED_ATTRS = ge::NamedAttrs;
  using DATA_TYPE = ge::DataType;

  using LIST_INT = std::vector<INT>;
  using LIST_FLOAT = std::vector<FLOAT>;
  using LIST_BOOL = std::vector<BOOL>;
  using LIST_STR = std::vector<STR>;
  using LIST_TENSOR = std::vector<TENSOR>;
  using LIST_TENSOR_DESC = std::vector<TENSOR_DESC>;
  using LIST_GRAPH = std::vector<GRAPH>;
  using LIST_BYTES = std::vector<BYTES>;
  using LIST_NAMED_ATTRS = std::vector<NAMED_ATTRS>;
  using LIST_DATA_TYPE = std::vector<DATA_TYPE>;
  using LIST_LIST_INT = std::vector<std::vector<int64_t>>;
  using LIST_LIST_FLOAT = std::vector<std::vector<float>>;
  using NamedAttrs = ge::NamedAttrs;
  // public type definitions
  // 这堆ValueType的预定义本质上是反向依赖，AnyValue不应该反向依赖ComputeGraph等数据结构
  // 后续整改掉
  enum ValueType {
    VT_NONE = 0,
    VT_STRING,
    VT_FLOAT,
    VT_BOOL,
    VT_INT,
    VT_TENSOR_DESC,
    VT_TENSOR,
    VT_BYTES,
    VT_GRAPH,
    VT_NAMED_ATTRS,
    VT_LIST_LIST_INT,
    VT_DATA_TYPE,
    VT_LIST_LIST_FLOAT,

    VT_LIST_BASE = 1000,
    VT_LIST_STRING = VT_LIST_BASE + VT_STRING,
    VT_LIST_FLOAT = VT_LIST_BASE + VT_FLOAT,
    VT_LIST_BOOL = VT_LIST_BASE + VT_BOOL,
    VT_LIST_INT = VT_LIST_BASE + VT_INT,
    VT_LIST_TENSOR_DESC = VT_LIST_BASE + VT_TENSOR_DESC,
    VT_LIST_TENSOR = VT_LIST_BASE + VT_TENSOR,
    VT_LIST_BYTES = VT_LIST_BASE + VT_BYTES,
    VT_LIST_GRAPH = VT_LIST_BASE + VT_GRAPH,
    VT_LIST_NAMED_ATTRS = VT_LIST_BASE + VT_NAMED_ATTRS,
    VT_LIST_DATA_TYPE = VT_LIST_BASE + VT_DATA_TYPE,
  };

 public:
  AnyValue() = default;
  AnyValue(AnyValue &&other) noexcept;
  AnyValue(const AnyValue &other) {
    if (!other.IsEmpty()) {
      other.operate_(OperateType::kOpClone, &other, this);
    }
  }
  AnyValue &operator=(AnyValue &&other) noexcept;
  AnyValue &operator=(const AnyValue &other);
  ~AnyValue() {
    Clear();
  }

  template<class T>
  static AnyValue CreateFrom(T &&value);
  // 如果只有万能引用，那么Set<int>(左值)这种调用方法会出错，因此有了这个函数
  template<typename T>
  static AnyValue CreateFrom(const T &value);

  template<class T>
  graphStatus SetValue(T &&value);

  // 如果只有万能引用，那么Set<int>(左值)这种调用方法会出错，因此有了这个函数
  template<typename T>
  graphStatus SetValue(const T &value);

  template<typename T>
  graphStatus SetValue(std::initializer_list<T> values);

  template<typename T>
  graphStatus GetValue(T &value) const;
  template<class T>
  const T *Get() const;
  template<class T>
  T *MutableGet();

  template<class T>
  bool SameType() const noexcept;

  void Swap(AnyValue &other) noexcept;

  void Clear() {
    if (operate_ == nullptr) {
      return;
    }
    operate_(OperateType::kOpClear, nullptr, this);
  }

  bool IsEmpty() const noexcept {
    return operate_ == nullptr;
  }

  ValueType GetValueType() const noexcept;
  TypeId GetValueTypeId() const noexcept;
  AnyValue Copy() const;

 private:
  template<typename T>
  void InnerSet(T &&value);
  const void *GetAddr() const;

 private:
  enum class OperateType { kOpClear, kOpGetAddr, kOpClone, kOpMove, kGetTypeId, kOperateTypeEnd };

  template<typename T>
  struct InlineOperations {
    static void Operate(OperateType ot, const AnyValue *av, void *out);
    static void Construct(const T &value, AnyValue *av);
    static void Construct(T &&value, AnyValue *av);
  };

  template<typename T>
  struct AllocateOperations {
    static void Operate(OperateType ot, const AnyValue *av, void *out);
    static void Construct(const T &value, AnyValue *av);
    static void Construct(T &&value, AnyValue *av);
  };

 private:
  using ValueHolder = union {
    void *pointer;
    std::aligned_storage<sizeof(void *)>::type inline_buf;
  };
  ValueHolder holder_{};

  void (*operate_)(OperateType ot, const AnyValue *av, void *out){nullptr};
};
using GeAttrValue = AnyValue;

template<typename T>
void AnyValue::AllocateOperations<T>::Construct(const T &value, AnyValue *av) {
  av->holder_.pointer = new (std::nothrow) T(value);
  av->operate_ = AnyValue::AllocateOperations<T>::Operate;
}
template<typename T>
void AnyValue::AllocateOperations<T>::Construct(T &&value, AnyValue *av) {
  av->holder_.pointer = ::new (std::nothrow) T(std::forward<T>(value));
  av->operate_ = AnyValue::AllocateOperations<T>::Operate;
}
template<typename T>
void AnyValue::AllocateOperations<T>::Operate(AnyValue::OperateType ot, const AnyValue *av, void *out) {
  switch (ot) {
    case OperateType::kOpClear: {
      auto av_p = static_cast<AnyValue *>(out);
      delete static_cast<T *>(av_p->holder_.pointer);
      av_p->holder_.pointer = nullptr;
      av_p->operate_ = nullptr;
      break;
    }
    case OperateType::kOpGetAddr:
      *static_cast<void **>(out) = const_cast<void *>(av->holder_.pointer);
      break;
    case OperateType::kOpClone:
      static_cast<AnyValue *>(out)->holder_.pointer =
          new (std::nothrow) T(*static_cast<const T *>(av->holder_.pointer));
      static_cast<AnyValue *>(out)->operate_ = av->operate_;
      break;
    case OperateType::kOpMove: {
      auto av_p = static_cast<AnyValue *>(out);
      av_p->holder_.pointer = av->holder_.pointer;
      av_p->operate_ = av->operate_;
      const_cast<AnyValue *>(av)->holder_.pointer = nullptr;
      break;
    }
    case OperateType::kGetTypeId:
      *static_cast<TypeId *>(out) = GetTypeId<T>();
      break;
    default:
      break;
  }
}
template<typename T>
void AnyValue::InlineOperations<T>::Construct(const T &value, AnyValue *av) {
  ::new (&(av->holder_.inline_buf)) T(value);
  av->operate_ = AnyValue::InlineOperations<T>::Operate;
}
template<typename T>
void AnyValue::InlineOperations<T>::Construct(T &&value, AnyValue *av) {
  Construct(value, av);
}
template<typename T>
void AnyValue::InlineOperations<T>::Operate(AnyValue::OperateType ot, const AnyValue *av, void *out) {
  switch (ot) {
    case OperateType::kOpClear: {
      auto av_p = static_cast<AnyValue *>(out);
      reinterpret_cast<T *>(&av_p->holder_.inline_buf)->~T();
      av_p->operate_ = nullptr;
      break;
    }
    case OperateType::kOpGetAddr:
      *static_cast<void **>(out) = const_cast<void *>(reinterpret_cast<const void *>(&av->holder_.inline_buf));
      break;
    case OperateType::kOpClone: {
      auto av_p = static_cast<AnyValue *>(out);
      new (&av_p->holder_.inline_buf) T(*reinterpret_cast<const T *>(&av->holder_.inline_buf));
      av_p->operate_ = av->operate_;
      break;
    }
    case OperateType::kOpMove: {
      auto av_p = static_cast<AnyValue *>(out);
      auto moved_t_p = const_cast<T *>(reinterpret_cast<const T *>(&av->holder_.inline_buf));
      new (&av_p->holder_.inline_buf) T(std::move(*moved_t_p));
      av_p->operate_ = av->operate_;
      break;
    }
    case OperateType::kGetTypeId:
      *static_cast<TypeId *>(out) = GetTypeId<T>();
      break;
    default:
      break;
  }
}

template<class T>
AnyValue AnyValue::CreateFrom(T &&value) {
  AnyValue av;
  av.InnerSet(std::forward<T>(value));
  return av;
}
template<typename T>
AnyValue AnyValue::CreateFrom(const T &value) {
  AnyValue av;
  av.InnerSet(value);
  return av;
}
template<typename T>
void AnyValue::InnerSet(T &&value) {
  using PureT = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
  using Inline = std::integral_constant<bool, sizeof(PureT) <= sizeof(holder_)>;
  using Operations =
      typename std::conditional<Inline{}, AnyValue::InlineOperations<PureT>, AnyValue::AllocateOperations<PureT>>::type;

  Operations::Construct(std::forward<T>(value), this);
}
template<class T>
graphStatus AnyValue::SetValue(T &&value) {
  Clear();
  InnerSet(std::forward<T>(value));
  return GRAPH_SUCCESS;
}
template<typename T>
graphStatus AnyValue::SetValue(const T &value) {
  Clear();
  InnerSet(value);
  return GRAPH_SUCCESS;
}

// TODO 补充UT
template<typename T>
graphStatus AnyValue::SetValue(std::initializer_list<T> values) {
  Clear();
  InnerSet(std::vector<T>(std::move(values)));
  return GRAPH_SUCCESS;
}
template<class T>
const T *AnyValue::Get() const {
  if (!SameType<T>()) {
    return nullptr;
  }
  if (IsEmpty()) {
    return nullptr;
  }
  return static_cast<const T *>(GetAddr());
}
template<typename T>
graphStatus AnyValue::GetValue(T &value) const {
  auto p = Get<T>();
  if (p == nullptr) {
    return GRAPH_FAILED;
  }
  value = *p;
  return GRAPH_SUCCESS;
}
template<class T>
T *AnyValue::MutableGet() {
  return const_cast<T *>(Get<T>());
}
template<class T>
bool AnyValue::SameType() const noexcept {
  if (operate_ == nullptr) {
    return false;
  }
  TypeId tid = kInvalidTypeId;
  operate_(OperateType::kGetTypeId, this, &tid);
  return tid == GetTypeId<T>();
}
}  // namespace ge

#endif  //EXECUTE_GRAPH_ANY_VALUE_H
