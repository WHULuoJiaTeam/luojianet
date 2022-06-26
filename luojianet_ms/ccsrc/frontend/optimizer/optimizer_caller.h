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

#ifndef LUOJIANET_MS_CCSRC_OPTIMIZER_OPTIMIZER_CALLER_H_
#define LUOJIANET_MS_CCSRC_OPTIMIZER_OPTIMIZER_CALLER_H_

#include <memory>

#include "ir/anf.h"
#include "ir/visitor.h"

namespace luojianet_ms {
namespace opt {
class Optimizer;
using OptimizerPtr = std::shared_ptr<Optimizer>;
using OptimizerWeakPtr = std::weak_ptr<Optimizer>;
using PredicateFuncType = luojianet_ms::PredicateFuncType;
}  // namespace opt

class OptimizerCaller {
 public:
  virtual AnfNodePtr operator()(const opt::OptimizerPtr &, const AnfNodePtr &) { return nullptr; }
  virtual ~OptimizerCaller() = default;
};
using OptimizerCallerPtr = std::shared_ptr<OptimizerCaller>;
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_OPTIMIZER_OPTIMIZER_CALLER_H_
