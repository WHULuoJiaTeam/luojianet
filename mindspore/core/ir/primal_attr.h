/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_PRIMAL_ATTR_H_
#define MINDSPORE_CORE_IR_PRIMAL_ATTR_H_
#include <string>
#include <memory>
#include <stack>
#include "utils/hash_map.h"
#include "utils/visible.h"

namespace mindspore {
class Value;
using ValuePtr = std::shared_ptr<Value>;

class MS_CORE_API PrimalAttrManager {
 public:
  static PrimalAttrManager &GetInstance() noexcept;
  PrimalAttrManager(const PrimalAttrManager &) = delete;
  PrimalAttrManager &operator=(const PrimalAttrManager &) = delete;
  ~PrimalAttrManager() = default;
  void SetPrimalAttr(const mindspore::HashMap<std::string, ValuePtr> &primal_attrs) { primal_attrs_ = primal_attrs; }
  void ClearPrimalAttr() noexcept { primal_attrs_.clear(); }
  mindspore::HashMap<std::string, ValuePtr> GetCurrentPrimalAttr() { return primal_attrs_; }

 private:
  PrimalAttrManager() = default;
  mindspore::HashMap<std::string, ValuePtr> primal_attrs_;
};

// PrimalAttrGuard is a class that help generate the back propagation cnode
// with specified primal attrs in the current c++ action scope.
class PrimalAttrGuard {
 public:
  explicit PrimalAttrGuard(const mindspore::HashMap<std::string, ValuePtr> &primal_attrs) {
    PrimalAttrManager::GetInstance().SetPrimalAttr(primal_attrs);
  }
  ~PrimalAttrGuard() { PrimalAttrManager::GetInstance().ClearPrimalAttr(); }
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_PRIMAL_ATTR_H_
