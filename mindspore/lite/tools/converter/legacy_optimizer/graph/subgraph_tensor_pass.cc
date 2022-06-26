/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <algorithm>
#include <memory>
#include "tools/converter/legacy_optimizer/graph/subgraph_tensor_pass.h"
#include "src/common/log_adapter.h"
#include "src/common/utils.h"
#include "tools/common/graph_util.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
bool SubgraphTensorPass::IsUsing(const schema::MetaGraphT &graph, const uint32_t &tensor_idx) {
  for (const auto &node : graph.nodes) {
    if (IsContain<uint32_t>(node->inputIndex, tensor_idx)) {
      return true;
    }
    if (IsContain<uint32_t>(node->outputIndex, tensor_idx)) {
      return true;
    }
  }
  for (const auto &subgraph : graph.subGraph) {
    if (IsContain<uint32_t>(subgraph->inputIndices, tensor_idx)) {
      return true;
    }
    if (IsContain<uint32_t>(subgraph->outputIndices, tensor_idx)) {
      return true;
    }
  }
  return false;
}

void SubgraphTensorPass::UpdateTensorIdx(schema::MetaGraphT *graph, const uint32_t &tensor_idx) {
  for (const auto &subgraph : graph->subGraph) {
    UpdateVec<uint32_t>(&(subgraph->inputIndices), tensor_idx);
    UpdateVec<uint32_t>(&(subgraph->outputIndices), tensor_idx);
  }
  for (const auto &node : graph->nodes) {
    UpdateVec<uint32_t>(&(node->inputIndex), tensor_idx);
    UpdateVec<uint32_t>(&(node->outputIndex), tensor_idx);
  }
  UpdateVec<uint32_t>(&(graph->inputIndex), tensor_idx);
  UpdateVec<uint32_t>(&(graph->outputIndex), tensor_idx);
}

void SubgraphTensorPass::RemoveUselessTensors(schema::MetaGraphT *graph) {
  for (auto it = graph->allTensors.begin(); it != graph->allTensors.end();) {
    uint32_t idx = it - graph->allTensors.begin();
    if (IsUsing(*graph, idx)) {
      it++;
    } else {
      it = graph->allTensors.erase(it);
      UpdateTensorIdx(graph, idx);
    }
  }
}

void SubgraphTensorPass::SyncMainGraphInputAndOutput(const schema::MetaGraphT &graph) {
  MS_ASSERT(graph.subGraph.size() > 0);
  graph.subGraph[0]->inputIndices.assign(graph.inputIndex.begin(), graph.inputIndex.end());
}

STATUS SubgraphTensorPass::Run(schema::MetaGraphT *graph) {
  CHECK_NULL_RETURN(graph);

  RemoveUselessTensors(graph);

  SetSubgraphTensorIndices(graph);

  SyncMainGraphInputAndOutput(*graph);

  return RET_OK;
}
}  // namespace lite
}  // namespace mindspore
