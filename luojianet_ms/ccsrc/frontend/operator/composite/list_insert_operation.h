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

#ifndef LUOJIANET_MS_CCSRC_FRONTEND_OPERATOR_COMPOSITE_LIST_INSERT_OPERATION_H_
#define LUOJIANET_MS_CCSRC_FRONTEND_OPERATOR_COMPOSITE_LIST_INSERT_OPERATION_H_

#include <vector>
#include <string>
#include <memory>

#include "ir/meta_func_graph.h"

namespace luojianet_ms {
// namespace to support composite operators definition
namespace prim {
class ListInsert : public MetaFuncGraph {
 public:
  explicit ListInsert(const std::string &name) : MetaFuncGraph(name) {}
  ~ListInsert() override = default;
  MS_DECLARE_PARENT(ListInsert, MetaFuncGraph)
  FuncGraphPtr GenerateFuncGraph(const abstract::AbstractBasePtrList &a_list) override;
  friend std::ostream &operator<<(std::ostream &os, const ListInsert &list_insert) {
    os << list_insert.name_;
    return os;
  }
  friend bool operator==(const ListInsert &lhs, const ListInsert &rhs) { return lhs.name_ == rhs.name_; }
};
using ListInsertPtr = std::shared_ptr<ListInsert>;
}  // namespace prim
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_FRONTEND_OPERATOR_COMPOSITE_LIST_INSERT_OPERATION_H_
