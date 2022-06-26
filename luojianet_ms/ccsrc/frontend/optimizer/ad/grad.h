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

#ifndef LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_AD_GRAD_H_
#define LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_AD_GRAD_H_

#include <memory>
#include <string>

#include "ir/anf.h"
#include "ir/meta_func_graph.h"
#include "frontend/optimizer/optimizer.h"

namespace luojianet_ms {
namespace ad {
FuncGraphPtr Grad(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer, bool is_top = true);
FuncGraphPtr Kprim(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources);
MetaFuncGraphPtr Kmeta(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &);
void CleanRes();
}  // namespace ad
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_AD_GRAD_H_
