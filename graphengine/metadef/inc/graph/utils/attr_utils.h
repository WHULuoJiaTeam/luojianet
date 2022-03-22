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

#ifndef INC_GRAPH_UTILS_ATTR_UTILS_H_
#define INC_GRAPH_UTILS_ATTR_UTILS_H_

#include <memory>
#include <string>
#include <vector>
#include <set>
#include "graph/detail/attributes_holder.h"
#include "graph/ge_attr_value.h"
#include "graph/types.h"
#include "graph/op_desc.h"

namespace ge {

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrUtils {
 public:
  class ConstAttrHolderAdapter;
  class AttrHolderAdapter;
  // Set
  static bool HasAttr(ConstAttrHolderAdapter &&obj, const std::string &name);

  static bool SetInt(AttrHolderAdapter &&obj, const std::string &name, const int64_t &value);
  static bool SetListInt(AttrHolderAdapter &&obj, const std::string &name, const std::vector<int64_t> &value);
  static bool SetListInt(AttrHolderAdapter &&obj, const std::string &name, const std::vector<uint32_t> &value);
  static bool SetListInt(AttrHolderAdapter &&obj, const std::string &name, const std::vector<int32_t> &value);
  static bool SetListInt(AttrHolderAdapter &&obj, const std::string &name, std::initializer_list<int64_t> &&value);

  static bool SetFloat(AttrHolderAdapter &&obj, const std::string &name, const float32_t &value);
  static bool SetListFloat(AttrHolderAdapter &&obj, const std::string &name, const std::vector<float32_t> &value);
  static bool SetBool(AttrHolderAdapter &&obj, const std::string &name, const bool &value);
  static bool SetListBool(AttrHolderAdapter &&obj, const std::string &name, const std::vector<bool> &value);
  static bool SetStr(AttrHolderAdapter &&obj, const std::string &name, const std::string &value);
  static bool SetListStr(AttrHolderAdapter &&obj, const std::string &name, const std::vector<std::string> &value);
  static bool SetTensorDesc(AttrHolderAdapter &&obj, const std::string &name, const GeTensorDesc &value);
  static bool SetListTensorDesc(AttrHolderAdapter &&obj, const std::string &name, const std::vector<GeTensorDesc> &value);
  static bool SetTensor(AttrHolderAdapter &&obj, const std::string &name, const GeTensorPtr &value);
  static bool SetTensor(AttrHolderAdapter &&obj, const std::string &name, const ConstGeTensorPtr &value);
  static bool SetTensor(AttrHolderAdapter &&obj, const std::string &name, const GeTensor &value);
  static bool SetListTensor(AttrHolderAdapter &&obj, const std::string &name, const std::vector<GeTensorPtr> &value);
  static bool SetListTensor(AttrHolderAdapter &&obj, const std::string &name, const std::vector<ConstGeTensorPtr> &value);
  static bool SetListTensor(AttrHolderAdapter &&obj, const std::string &name,
                            std::initializer_list<ConstGeTensorPtr> &&value);
  static bool SetListTensor(AttrHolderAdapter &&obj, const std::string &name, const std::vector<GeTensor> &value);
  static bool SetGraph(AttrHolderAdapter &&obj, const std::string &name, const ComputeGraphPtr &value);
  static bool SetListGraph(AttrHolderAdapter &&obj, const std::string &name, const std::vector<ComputeGraphPtr> &value);
  static bool SetBytes(AttrHolderAdapter &&obj, const std::string &name, const Buffer &value);
  static bool SetListBytes(AttrHolderAdapter &&obj, const std::string &name, const std::vector<Buffer> &value);
  static bool SetNamedAttrs(AttrHolderAdapter &&obj, const std::string &name, const NamedAttrs &value);
  static bool SetListNamedAttrs(AttrHolderAdapter &&obj, const std::string &name,
                                const std::vector<NamedAttrs> &value);

  // Get
  static bool GetInt(ConstAttrHolderAdapter &&obj, const std::string &name, int64_t &value);
  static bool GetInt(ConstAttrHolderAdapter &&obj, const std::string &name, int32_t &value);
  static bool GetInt(ConstAttrHolderAdapter &&obj, const std::string &name, uint32_t &value);
  static bool GetListInt(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<int64_t> &value);
  static bool GetListInt(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<int32_t> &value);
  static bool GetListInt(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<uint32_t> &value);
  static bool GetFloat(ConstAttrHolderAdapter &&obj, const std::string &name, float32_t &value);
  static bool GetListFloat(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<float32_t> &value);
  static bool GetBool(ConstAttrHolderAdapter &&obj, const std::string &name, bool &value);
  static bool GetListBool(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<bool> &value);
  static bool GetStr(ConstAttrHolderAdapter &&obj, const std::string &name, std::string &value);
  static bool GetListStr(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<std::string> &value);
  static bool GetTensorDesc(ConstAttrHolderAdapter &&obj, const std::string &name, GeTensorDesc &value);
  static bool GetListTensorDesc(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<GeTensorDesc> &value);
  static bool GetTensor(ConstAttrHolderAdapter &&obj, const std::string &name, ConstGeTensorPtr &value);
  static bool MutableTensor(AttrHolderAdapter &&obj, const std::string &name, GeTensorPtr &value);
  static bool GetListTensor(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<ConstGeTensorPtr> &value);
  static bool MutableListTensor(AttrHolderAdapter &&obj, const std::string &name, std::vector<GeTensorPtr> &value);
  static bool GetGraph(ConstAttrHolderAdapter &&obj, const std::string &name, ComputeGraphPtr &value);
  static bool GetListGraph(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<ComputeGraphPtr> &value);
  static bool GetBytes(ConstAttrHolderAdapter &&obj, const std::string &name, Buffer &value);
  static bool GetListBytes(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<Buffer> &value);
  static bool GetNamedAttrs(ConstAttrHolderAdapter &&obj, const std::string &name, NamedAttrs &value);
  static bool GetListNamedAttrs(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<NamedAttrs> &value);
  // Value will be moved
  static bool SetZeroCopyBytes(AttrHolderAdapter &&obj, const std::string &name, Buffer &&buffer);
  static bool GetZeroCopyBytes(ConstAttrHolderAdapter &&obj, const std::string &name, Buffer &buffer);
  // Value will be moved
  static bool SetZeroCopyListBytes(AttrHolderAdapter &&obj, const std::string &name,
                                   std::vector<Buffer> &list_buffer);
  static bool GetZeroCopyListBytes(ConstAttrHolderAdapter &&obj, const std::string &name,
                                   std::vector<Buffer> &list_buffer);

  static bool SetListListInt(AttrHolderAdapter &&obj, const std::string &name,
                             const std::vector<std::vector<int64_t>> &value);
  static bool GetListListInt(ConstAttrHolderAdapter &&obj, const std::string &name,
                             std::vector<std::vector<int64_t>> &value);

  static bool SetListListFloat(AttrHolderAdapter &&obj, const std::string &name,
                               const std::vector<std::vector<float32_t>> &value);
  static bool GetListListFloat(ConstAttrHolderAdapter &&obj, const std::string &name,
                               std::vector<std::vector<float32_t>> &value);

  static bool SetListDataType(AttrHolderAdapter &&obj, const std::string &name, const std::vector<ge::DataType> &value);
  static bool GetListDataType(ConstAttrHolderAdapter &&obj, const std::string &name, std::vector<ge::DataType> &value);

  static bool SetDataType(AttrHolderAdapter &&obj, const std::string &name, const ge::DataType &value);
  static bool GetDataType(ConstAttrHolderAdapter &&obj, const std::string &name, ge::DataType &value);

  static OpDescPtr CloneOpDesc(const ConstOpDescPtr &org_op_desc);

  static OpDescPtr CopyOpDesc(const ConstOpDescPtr &org_op_desc);
  static std::string GetAllAttrsStr(ConstAttrHolderAdapter &&obj);
  static std::map<std::string, AnyValue> GetAllAttrs(ConstAttrHolderAdapter &&obj);
  static std::string GetAttrsStrAfterRid(ConstAttrHolderAdapter &&obj, const std::set<std::string> &un_compute_attrs);
  class AttrHolderAdapter {
   public:
    AttrHolderAdapter(AttrHolder *const obj) : obj_(obj) {}
    ~AttrHolderAdapter() {}
    template <class T>
    AttrHolderAdapter(const std::shared_ptr<T> &obj) : obj_(obj.get()) {}
    AttrHolderAdapter(const AttrHolderAdapter &obj) : obj_(obj.get()) {}
    AttrHolderAdapter(AttrHolder &obj) : obj_(&obj) {}
    AttrHolder *operator->() const { return obj_; }
    AttrHolder *get() const { return obj_; }

    AttrHolderAdapter &operator=(const AttrHolderAdapter &rls) {
      if (&rls != this) {
        obj_ = rls.obj_;
      }
      return *this;
    }

   private:
    AttrHolder *obj_;
  };

  class ConstAttrHolderAdapter {
   public:
    ConstAttrHolderAdapter(const AttrHolder *const obj) : obj_(obj) {}
    ~ConstAttrHolderAdapter() {}
    template <class T>
    ConstAttrHolderAdapter(const std::shared_ptr<T> obj) : obj_(obj.get()) {}
    ConstAttrHolderAdapter(const ConstAttrHolderAdapter &obj) : obj_(obj.get()) {}
    ConstAttrHolderAdapter(const AttrHolder &obj) : obj_(&obj) {}
    operator bool() const { return obj_ != nullptr; }
    const AttrHolder *operator->() const { return obj_; }
    const AttrHolder *get() const { return obj_; }

    ConstAttrHolderAdapter &operator=(const ConstAttrHolderAdapter &rls) {
      if (&rls != this) {
        obj_ = rls.obj_;
      }
      return *this;
    }

   private:
    const AttrHolder *obj_;
  };
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_ATTR_UTILS_H_
