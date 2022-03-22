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

#ifndef INC_GRAPH_GE_ATTR_VALUE_H_
#define INC_GRAPH_GE_ATTR_VALUE_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "graph/buffer.h"
#include "detail/attributes_holder.h"
#include "graph/ge_error_codes.h"
#include "graph/ge_tensor.h"
#include "graph/any_value.h"

using std::map;
using std::string;
using std::vector;

namespace ge {
class GeTensor;

using GeTensorPtr = std::shared_ptr<GeTensor>;
using ConstGeTensorPtr = std::shared_ptr<const GeTensor>;

class ComputeGraph;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
using ConstComputeGraphPtr = std::shared_ptr<const ComputeGraph>;

class GeTensorDesc;

template<typename T>
bool SetAttrValue(AttrStore &attrs, const std::string &name, T &&value) {
  return attrs.SetByName(name, std::forward<T>(value));
}

template<typename T>
bool GetAttrValue(const AttrStore &attrs, const std::string &name, T &value) {
  const auto p = attrs.GetByName<T>(name);
  if (p == nullptr) {
    return false;
  }
  value = *p;
  return true;
}

template<typename T, typename RT = typename std::decay<T>::type>
RT *SetAndGetAttrValue(AttrStore &attrs, const std::string &name, T &&value) {
  if (!attrs.SetByName(name, std::forward<T>(value))) {
    return nullptr;
  }
  return attrs.MutableGetByName<RT>(name);
}

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NamedAttrs : public AttrHolder {
 public:
  NamedAttrs() = default;
  virtual ~NamedAttrs() = default;
  void SetName(const std::string &name);
  std::string GetName() const;
  AnyValue GetItem(const std::string &key) const;

 protected:
  ProtoAttrMap &MutableAttrMap() override;
  ConstProtoAttrMap &GetAttrMap() const override;

 private:
  AttrStore attrs_;
  std::string name_;

  friend class GeAttrValueImp;
};

class AttrValueImpl {
 public:
  AttrValueImpl() = default;
  ~AttrValueImpl() = default;

  friend class AttrValue;
  friend class AttrHolder;
  friend class Operator;

private:
  AnyValue geAttrValue_;
};
}  // namespace ge
#endif  // INC_GRAPH_GE_ATTR_VALUE_H_
