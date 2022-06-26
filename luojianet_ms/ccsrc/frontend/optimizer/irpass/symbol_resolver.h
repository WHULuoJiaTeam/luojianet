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

#ifndef LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_
#define LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_

#include <string>
#include <memory>

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "ir/pattern_matcher.h"
#include "pipeline/jit/parse/data_converter.h"
#include "include/common/utils/python_adapter.h"
#include "pipeline/jit/parse/parse_base.h"

namespace luojianet_ms {
namespace opt {
namespace irpass {
// Put GetAttr pattern and Resolve pattern together to ensure that GetAttr pattern always takes precedence over Resolve
// pattern. After matching GetAttr pattern, there may be new nodes that can match GetAttr pattern and Resolve pattern.
// The same is true for matching Resolve pattern.
//
// {prim::kPrimGetAttr, {prim::kPrimTupleGetItem, {prim::kPrimResolve, namespace, symbol}, index}, attr}
// {prim::kPrimGetAttr, {prim::kPrimResolve, namespace, symbol}, attr}
// {prim::kPrimGetAttr, namespace, attr}
// {prim::kPrimGetAttr, MsClassObject, attr}
// {prim::kPrimGetAttr, bool, attr}
// {prim::kPrimResolve, namespace, symbol}
class Resolver : public OptimizerCaller {
 public:
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) override;
};
}  // namespace irpass
}  // namespace opt
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_IRPASS_SYMBOL_RESOLVER_H_
