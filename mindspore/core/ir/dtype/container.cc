/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "ir/dtype/container.h"
#include <cstdlib>
#include <algorithm>
#include "utils/log_adapter.h"

namespace mindspore {
static std::string DumpTypeVector(const std::vector<TypePtr> &elements, bool is_dumptext) {
  std::ostringstream oss;
  bool begin = true;
  size_t cnt = 0;
  // write 'Tuple[Bool, Bool, Bool, Int, Float, Float]' as 'Tuple[Bool...3, Int, Float...2]'
  for (size_t i = 0; i < elements.size(); ++i) {
    TypePtr elem = elements[i];
    cnt += 1;
    bool print = false;
    if (i + 1 < elements.size()) {
      TypePtr next = elements[i + 1];
      if (*elem != *next) {
        print = true;
      }
    } else {
      // encounter last element
      print = true;
    }
    if (!print) {
      continue;
    }
    if (!begin) {
      oss << ",";
    } else {
      begin = false;
    }
    oss << (is_dumptext ? elem->DumpText() : elem->ToString());
    if (cnt > 1) {
      oss << "*" << cnt;
    }
    cnt = 0;
  }
  return oss.str();
}

TypePtr List::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<List>();
  } else {
    TypePtrList elements;
    (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(elements),
                         [](const TypePtr &ele) { return ele->DeepCopy(); });
    auto copy = std::make_shared<List>(elements);
    return copy;
  }
}

const TypePtr List::operator[](std::size_t dim) const {
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "Index " << dim << " is out range of the List size " << size() << ".";
  }
  return elements_[dim];
}

bool List::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  const List &other_list = static_cast<const List &>(other);
  if (elements_.size() != other_list.elements_.size()) {
    return false;
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    if (*elements_[i] != *other_list.elements_[i]) {
      return false;
    }
  }
  return true;
}

Class::Class(const Named &tag, const ClassAttrVector &attributes,
             const mindspore::HashMap<std::string, ValuePtr> &methods)
    : Object(kObjectTypeClass, false), attributes_(attributes), tag_(tag), methods_(methods) {}

std::string List::DumpContent(bool is_dumptext) const {
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "List";
  } else {
    buffer << "List[";
    buffer << DumpTypeVector(elements_, is_dumptext);
    buffer << "]";
  }
  return buffer.str();
}

bool Class::operator==(const Type &other) const {
  // Class is cached for each pyobj in ParseDataClass, so ClassPtr is one by one map to pyobj.
  return &other == this;
}

TypePtr Class::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<Class>();
  } else {
    auto copy = std::make_shared<Class>(tag_, attributes_, methods_);
    return copy;
  }
}

std::string Class::DumpContent(bool is_dumptext) const {
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "Cls";
  } else {
    bool begin = true;
    buffer << "Cls." << tag_ << "[";
    for (auto &attr : attributes_) {
      if (!begin) {
        buffer << ", ";
      } else {
        begin = false;
      }
      auto sub_content = is_dumptext ? attr.second->DumpText() : attr.second->ToString();
      buffer << attr.first << ":" << sub_content;
    }
    buffer << "]";
  }
  return buffer.str();
}

TypePtr Tuple::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<Tuple>();
  } else {
    TypePtrList elements;
    (void)std::transform(elements_.begin(), elements_.end(), std::back_inserter(elements),
                         [](const TypePtr &ele) { return ele->DeepCopy(); });
    auto copy = std::make_shared<Tuple>(elements);
    return copy;
  }
}

bool Tuple::operator==(const Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }
  auto other_tuple = static_cast<const Tuple &>(other);
  if (elements_.size() != other_tuple.elements_.size()) {
    return false;
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    if (*elements_[i] != *other_tuple.elements_[i]) {
      return false;
    }
  }
  return true;
}

const TypePtr Tuple::operator[](std::size_t dim) const {
  if (dim >= size()) {
    MS_LOG(EXCEPTION) << "Index " << dim << " is out range of the Tuple size " << size() << ".";
  }
  return elements_[dim];
}

std::string Tuple::DumpContent(bool is_dumptext) const {
  std::ostringstream buffer;
  if (IsGeneric()) {
    buffer << "Tuple";
  } else {
    buffer << "Tuple[";
    buffer << DumpTypeVector(elements_, is_dumptext);
    buffer << "]";
  }
  return buffer.str();
}

TypePtr Dictionary::DeepCopy() const {
  if (IsGeneric()) {
    return std::make_shared<Dictionary>();
  } else {
    std::vector<std::pair<std::string, TypePtr>> kv;
    (void)std::transform(
      key_values_.begin(), key_values_.end(), std::back_inserter(kv),
      [](const std::pair<std::string, TypePtr> &item) { return std::make_pair(item.first, item.second->DeepCopy()); });
    return std::make_shared<Dictionary>(kv);
  }
}

std::string DumpKeyVector(std::vector<std::string> keys) {
  std::ostringstream buffer;
  for (auto key : keys) {
    buffer << key << ",";
  }
  return buffer.str();
}

std::string Dictionary::DumpContent(bool) const {
  std::ostringstream buffer;
  std::vector<std::string> keys;
  std::vector<TypePtr> values;
  for (const auto &kv : key_values_) {
    keys.push_back(kv.first);
    values.push_back(kv.second);
  }
  if (IsGeneric()) {
    buffer << "Dictionary";
  } else {
    buffer << "Dictionary[";
    buffer << "[" << DumpKeyVector(keys) << "],";
    buffer << "[" << DumpTypeVector(values, false) << "]";
    buffer << "]";
  }
  return buffer.str();
}

bool Dictionary::operator==(const mindspore::Type &other) const {
  if (!IsSameObjectType(*this, other)) {
    return false;
  }

  const auto &other_dict = static_cast<const Dictionary &>(other);
  if (key_values_.size() != other_dict.key_values_.size()) {
    return false;
  }
  for (size_t index = 0; index < key_values_.size(); index++) {
    if (key_values_[index].first != other_dict.key_values_[index].first) {
      return false;
    }
    if (*key_values_[index].second != *other_dict.key_values_[index].second) {
      return false;
    }
  }
  return true;
}
}  // namespace mindspore
