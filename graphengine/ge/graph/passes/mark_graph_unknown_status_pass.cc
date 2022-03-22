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

#include "graph/passes/mark_graph_unknown_status_pass.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
const char *const kOwnerGraphIsUnknown = "OwnerGraphIsUnknown";
}

Status MarkGraphUnknownStatusPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);
  bool is_unknown_shape = false;
  bool forced_unknown = false;
  for (const auto &node : graph->GetDirectNode()) {
    GE_CHK_GRAPH_STATUS_RET(ge::NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown_shape),
                            "[Get][ShapeStatus] of node[%s] failed!", node->GetName().c_str());
    if (is_unknown_shape) {
      break;
    }
    if (AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_FORCE_UNKNOWN_SHAPE, forced_unknown) && forced_unknown) {
      GELOGD("node %s was marked as unknown shape.", node->GetName().c_str());
      is_unknown_shape = true;
      break;
    }
  }

  const auto &node = graph->GetParentNode();
  if (!is_unknown_shape && node != nullptr && node->GetType() == PARTITIONEDCALL) {
    GE_CHK_GRAPH_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown_shape),
                            "[Get][ShapeStatus] of node[%s] failed!", node->GetName().c_str());
  }

  for (const auto &node : graph->GetDirectNode()) {
    GELOGD("Set OwnerGraphIsUnknown attr to node[%s]", node->GetName().c_str());
    (void)AttrUtils::SetBool(node->GetOpDesc(), kOwnerGraphIsUnknown, is_unknown_shape);
  }
  graph->SetGraphUnknownFlag(is_unknown_shape);
  GELOGD("mark graph [%s] unknown status success! value is %d", graph->GetName().c_str(), is_unknown_shape);
  return SUCCESS;
}
}  // namespace ge