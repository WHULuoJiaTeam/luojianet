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

#ifndef INC_GRAPH_RANGE_VISTOR_H_
#define INC_GRAPH_RANGE_VISTOR_H_

#include <vector>
#include <list>

template <class E, class O>
class RangeVistor {
 public:
  using Iterator = typename std::vector<E>::iterator;
  using ConstIterator = typename std::vector<E>::const_iterator;

  RangeVistor(O owner, const std::vector<E> &vs) : owner_(owner), elements_(vs) {}
  RangeVistor(O owner, const std::list<E> &vs) : owner_(owner), elements_(vs.begin(), vs.end()) {}

  ~RangeVistor() {}

  Iterator begin() { return elements_.begin(); }

  Iterator end() { return elements_.end(); }

  ConstIterator begin() const { return elements_.begin(); }

  ConstIterator end() const { return elements_.end(); }

  std::size_t size() const { return elements_.size(); }

  bool empty() const { return elements_.empty(); }

  E &at(std::size_t index) { return elements_.at(index); }

  const E &at(std::size_t index) const { return elements_.at(index); }

 private:
  O owner_;
  std::vector<E> elements_;
};

#endif  // INC_GRAPH_RANGE_VISTOR_H_
