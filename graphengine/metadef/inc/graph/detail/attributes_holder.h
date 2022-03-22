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

#ifndef INC_GRAPH_DETAIL_ATTRIBUTES_HOLDER_H_
#define INC_GRAPH_DETAIL_ATTRIBUTES_HOLDER_H_

#include <map>
#include <memory>
#include <string>
#include <set>
#include <unordered_set>
#include <utility>
#include <vector>
#include "graph/detail/any_map.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/attr_store.h"

namespace google {
namespace protobuf {
class Message;
template<typename Key, typename T>
class Map;
}  // namespace protobuf
}  // namespace google

namespace ge {
namespace proto {
class AttrDef;
class TensorDef;
class TensorDescriptor;
class ShapeDef;
class NamedAttrs;
class ModelDef;
class OpDef;
class GraphDef;
}  // namespace proto

using ProtoAttrMap = AttrStore;
using ConstProtoAttrMap = const AttrStore;
using ProtoMsgOwner = std::shared_ptr<::google::protobuf::Message>;

template<class ProtoType>
class GeIrProtoHelper {
 public:
  GeIrProtoHelper(const ProtoMsgOwner &protoOwner, ProtoType *const protoMsg)
      : protoOwner_(protoOwner), protoMsg_(protoMsg) {}

  GeIrProtoHelper() {
    protoOwner_ = std::shared_ptr<::google::protobuf::Message>(nullptr);
    protoMsg_ = nullptr;
  }
  virtual ~GeIrProtoHelper() = default;

  template<typename T>
  GeIrProtoHelper(const GeIrProtoHelper<T> &other) {
    protoOwner_ = other.protoOwner_;
    protoMsg_ = other.protoMsg_;
  }
  template<typename T>
  GeIrProtoHelper &operator=(const GeIrProtoHelper<T> &other) {
    protoOwner_ = other.protoOnwer_;
    protoMsg_ = other.protoMsg_;
    return *this;
  }
  void InitDefault();
  template<typename T>
  bool operator==(const GeIrProtoHelper<T> &other) const {
    return protoOwner_ == other.protoOwner_ && protoMsg_ == other.protoMsg_;
  }

  inline const ProtoMsgOwner &GetProtoOwner() const {
    return protoOwner_;
  }
  inline ProtoType *GetProtoMsg() const {
    return protoMsg_;
  }
  void CopyValueFrom(const GeIrProtoHelper<const ProtoType> &other) {
    if ((other.protoMsg_ != nullptr) && (protoMsg_ != nullptr)) {
      *protoMsg_ = *other.protoMsg_;
    }
  }
  void MoveValueFrom(GeIrProtoHelper<ProtoType> &&other) {
    if ((other.protoMsg_ != nullptr) && (protoMsg_ != nullptr)) {
      *protoMsg_ = std::move(*other.protoMsg_);
    }
  }

  void Swap(GeIrProtoHelper<ProtoType> &other) {
    protoOwner_.swap(other.protoOwner_);

    ProtoType *const temp = protoMsg_;
    protoMsg_ = other.protoMsg_;
    other.protoMsg_ = temp;
  }

  // protoMsg_ is part of protoOwner_, they have the same runtime
  ProtoMsgOwner protoOwner_ = nullptr;
  ProtoType *protoMsg_ = nullptr;
  friend class GeIrProtoHelper<typename std::conditional<
      std::is_const<ProtoType>::value, typename std::remove_const<ProtoType>::type, const ProtoType>::type>;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrHolder {
 public:
  AttrHolder() = default;
  virtual ~AttrHolder() = default;

  graphStatus SetAttr(const std::string &name, const AnyValue &value);

  graphStatus TrySetAttr(const std::string &name, const AnyValue &value);

  graphStatus GetAttr(const std::string &name, AnyValue &value) const;

  bool HasAttr(const std::string &name) const;

  graphStatus DelAttr(const std::string &name);

  void CopyAttrsFrom(const AttrHolder &holder);

  void CopyFrom(const AttrHolder &holder);

  void Swap(AttrHolder &holder) {
    requiredAttrs_.swap(holder.requiredAttrs_);
    extAttrs_.Swap(holder.extAttrs_);
  }

  template<class T>
  bool SetExtAttr(const std::string &name, const T &value) {
    return extAttrs_.Set(name, value);
  }
  template<class T>
  T TryGetExtAttr(const std::string &name, const T defaultValue) const {
    T ret(defaultValue);
    (void) extAttrs_.Get(name, ret);
    return ret;
  }

 protected:
  graphStatus AddRequiredAttr(const std::string &name);
  const std::set<std::string> GetAllAttrNames() const;
  const std::map<std::string, AnyValue> GetAllAttrs() const;

  virtual ProtoAttrMap &MutableAttrMap() = 0;
  virtual ConstProtoAttrMap &GetAttrMap() const = 0;

  friend class ModelSerializeImp;
  friend class AttrUtils;
  friend class AttrUtilsHelper;

 private:
  std::vector<std::string> requiredAttrs_;
  AnyMap extAttrs_;
};
}  // namespace ge
#endif  // INC_GRAPH_DETAIL_ATTRIBUTES_HOLDER_H_
