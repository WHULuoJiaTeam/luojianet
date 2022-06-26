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

#include "src/train/graph_fusion.h"
#include "tools/converter/optimizer.h"
#include "tools/converter/legacy_optimizer/fusion/matmul_biasadd_fusion_pass.h"
#include "tools/converter/legacy_optimizer/graph/isolated_node_remove_pass.h"

namespace luojianet_ms {
namespace lite {
STATUS GraphFusion::Run(schema::MetaGraphT *graph) {
  if (graph == nullptr) {
    MS_LOG(ERROR) << "graph is nullptr.";
    return RET_ERROR;
  }
  Optimizer fusion_optimizer;
  fusion_optimizer.AddPass(new (std::nothrow) MatMulBiasAddFusionPass());
  fusion_optimizer.AddPass(new (std::nothrow) IsolatedNodeRemovePass());
  auto status = fusion_optimizer.Run(graph);
  if (status != RET_OK && status != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "graph fusion failed.";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace lite
}  // namespace luojianet_ms
