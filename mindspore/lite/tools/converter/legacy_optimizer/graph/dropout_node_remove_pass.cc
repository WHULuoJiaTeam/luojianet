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

#include "tools/converter/legacy_optimizer/graph/dropout_node_remove_pass.h"
#include <queue>
#include "src/common/log_adapter.h"
#include "tools/common/meta_graph_utils.h"
#include "include/errorcode.h"
#include "schema/inner/model_generated.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
STATUS IsolateDropoutNode(schema::MetaGraphT *graphT, size_t nodeIdx) {
  MS_ASSERT(graphT != nullptr);
  CNodeT *node = graphT->nodes.at(nodeIdx).get();
  auto inputTensorIdxes = node->inputIndex;
  auto outputTensorIdxes = node->outputIndex;
  auto preNodeIdxes = GetInputNodeIdx(*graphT, nodeIdx);
  if (preNodeIdxes.size() > 1 || outputTensorIdxes.size() > 2) {
    MS_LOG(ERROR) << "Only support node who has no more than one input and two output";
    return RET_ERROR;
  }
  if (inputTensorIdxes.empty()) {
    MS_LOG(ERROR) << "Error, " << nodeIdx << "th node has no input tensor";
    return RET_ERROR;
  }
  if (outputTensorIdxes.size() == 2) {
    auto outDataTensorIdx = outputTensorIdxes.at(1);
    auto &gOutTensorIdx = graphT->outputIndex;
    if (std::any_of(gOutTensorIdx.begin(), gOutTensorIdx.end(),
                    [&outDataTensorIdx](const unsigned int &idx) { return (idx == outDataTensorIdx); })) {
      MS_LOG(ERROR) << "Unsupported Dropout: " << node->name.c_str() << " with mask output.";
      return RET_ERROR;
    }
    auto postNodeIdxes = GetOutputNodeIdx(*graphT, nodeIdx, 1);
    if (!postNodeIdxes.empty()) {
      MS_LOG(WARNING) << "Unsupported Dropout: " << node->name.c_str() << " with mask output.";
      return RET_OK;
    }
  }
  auto inDataTensorIdx = inputTensorIdxes.front();
  if (!outputTensorIdxes.empty()) {
    auto outDataTensorIdx = outputTensorIdxes.front();
    MS_ASSERT(graphT->allTensors.size() > inDataTensorIdx);
    MS_ASSERT(graphT->allTensors.at(inDataTensorIdx) != nullptr);
    auto &gOutTensorIdx = graphT->outputIndex;
    auto matchedTensor =
      std::find_if(gOutTensorIdx.begin(), gOutTensorIdx.end(),
                   [&outDataTensorIdx](const unsigned int &idx) { return (idx == outDataTensorIdx); });
    *matchedTensor = inDataTensorIdx;
    // find poseNode
    auto postNodeIdxes = GetOutputNodeIdx(*graphT, nodeIdx, 0);
    for (auto postNodeIdx : postNodeIdxes) {
      MS_ASSERT(graphT->nodes.size() > postNodeIdx);
      auto &postNode = graphT->nodes.at(postNodeIdx);
      MS_ASSERT(postNode != nullptr);
      auto iter = std::find_if(postNode->inputIndex.begin(), postNode->inputIndex.end(),
                               [&outDataTensorIdx](const unsigned int &idx) { return (idx == outDataTensorIdx); });
      *iter = inDataTensorIdx;
    }
  }

  // now all node's outputTensors are useless
  // remove all node's outputTensors
  auto status = RemoveTensor(graphT, outputTensorIdxes);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "RemoveOutputTensors of node " << node->name.c_str() << "failed";
    return RET_ERROR;
  }

  node->inputIndex.clear();
  node->outputIndex.clear();
  return RET_OK;
}
}  // namespace

STATUS DropoutNodeRemovePass::Run(schema::MetaGraphT *graph) {
  CHECK_NULL_RETURN(graph);
  bool ifChanged = false;
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    auto &node = graph->nodes.at(i);
    if (node->primitive == nullptr || node->primitive->value.type == schema::PrimitiveType_Dropout) {
      MS_LOG(INFO) << "node->primitive is nullptr, node name: " << node->name;
      ifChanged = true;
      auto status = IsolateDropoutNode(graph, i);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "IsolateDropoutNode failed, subGraph: " << graph->name << ", node: " << node->name
                      << ", error: " << status;
        return status;
      }
    }
  }
  return ifChanged ? RET_OK : RET_NO_CHANGE;
}
}  // namespace lite
}  // namespace mindspore
