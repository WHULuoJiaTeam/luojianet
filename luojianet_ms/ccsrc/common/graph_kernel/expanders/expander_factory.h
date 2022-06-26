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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_EXPANDER_FACTORY_H_
#define LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_EXPANDER_FACTORY_H_

#include <functional>
#include <string>
#include <memory>

#include "utils/hash_map.h"
#include "common/graph_kernel/expanders/utils.h"

namespace luojianet_ms::graphkernel::expanders {
class OpExpanderFactory {
 public:
  static OpExpanderFactory &Instance() {
    static OpExpanderFactory instance = OpExpanderFactory();
    return instance;
  }
  std::shared_ptr<OpDesc> GetExpander(const std::string &op) {
    if (auto iter = creators.find(op); iter != creators.end()) {
      auto expander_ptr = iter->second();
      expander_ptr->op_ = op;
      return expander_ptr;
    }
    return nullptr;
  }
  OpExpanderFactory() = default;
  ~OpExpanderFactory() = default;

  using RegFunc = std::function<std::shared_ptr<OpDesc>()>;
  void Register(const std::string &op, const RegFunc &func) { creators[op] = func; }

 private:
  luojianet_ms::HashMap<std::string, RegFunc> creators;
};

class OpExpanderRegister {
 public:
  OpExpanderRegister(const std::string &name, const OpExpanderFactory::RegFunc &func) : func_(func) {
    OpExpanderFactory::Instance().Register(name, func);
  }
  ~OpExpanderRegister() = default;

 private:
  // for pclint-plus
  OpExpanderFactory::RegFunc func_;
};

#define OP_EXPANDER_REGISTER(name, cls)                 \
  const OpExpanderRegister g_##cls##_expander_reg(name, \
                                                  []() -> std::shared_ptr<OpDesc> { return std::make_shared<cls>(); })
}  // namespace luojianet_ms::graphkernel::expanders
#endif  // LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_EXPANDERS_EXPANDER_FACTORY_H_
