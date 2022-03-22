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

#include "graph/passes/link_gen_mask_nodes_pass.h"

#include <set>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"
#include "init/gelib.h"

using std::set;
using std::vector;

namespace ge {
namespace {
const size_t kGenMaskInputIndex = 1;
const size_t kDefaultMaxParallelNum = 1;
}  // namespace

LinkGenMaskNodesPass::LinkGenMaskNodesPass(const map<string, int> &stream_max_parallel_num)
    : GraphPass(), stream_max_parallel_num_(stream_max_parallel_num) {}

// GenMask is the second input of DoMask and one GenMask's output may be used by multiple DoMask.
// We will control the order of GenMask according to the order of the first DoMask.
Status LinkGenMaskNodesPass::Run(ComputeGraphPtr graph) {
  GE_CHECK_NOTNULL(graph);

  vector<NodePtr> gen_mask_nodes;
  GetAllGenMaskNodes(graph, gen_mask_nodes);

  size_t gen_mask_group_size = gen_mask_nodes.size();
  Status status = GetGenMaskGroupSize(gen_mask_nodes, gen_mask_group_size);
  if (status != SUCCESS) {
    GELOGE(FAILED, "[Get][GenMaskGroupSize] failed.");
    return FAILED;
  }

  if (gen_mask_group_size < 1) {
    gen_mask_group_size = 1;
  }

  for (size_t index = 1; index < gen_mask_nodes.size(); ++index) {
    if (index % gen_mask_group_size == 0) {
      GELOGI("skiped index: %zu.", index);
      continue;
    }

    NodePtr &src_node = gen_mask_nodes[index - 1];
    auto src_anchor = src_node->GetOutControlAnchor();
    GE_CHECK_NOTNULL(src_anchor);

    NodePtr &dest_node = gen_mask_nodes[index];
    auto dest_anchor = dest_node->GetInControlAnchor();
    GE_CHECK_NOTNULL(dest_anchor);

    graphStatus status_link_to = src_anchor->LinkTo(dest_anchor);
    if (status_link_to != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Op:%s(%s) link control to op:%s(%s) failed",
                        src_node->GetName().c_str(), src_node->GetType().c_str(),
                        dest_node->GetName().c_str(), dest_node->GetType().c_str());
      GELOGE(FAILED, "[Add][Edge] Op:%s(%s) link control to op:%s(%s) failed",
             src_node->GetName().c_str(), src_node->GetType().c_str(),
             dest_node->GetName().c_str(), dest_node->GetType().c_str());
      return FAILED;
    }
    GELOGD("Link from %s to %s.", src_node->GetName().c_str(), dest_node->GetName().c_str());
  }

  return SUCCESS;
}

// [pointer can not be null]
bool LinkGenMaskNodesPass::AreAllInputsConst(const NodePtr &node) const {
  for (const NodePtr &in_node : node->GetInDataNodes()) {
    string op_type = in_node->GetType();
    if ((op_type != CONSTANT) && (op_type != CONSTANTOP)) {
      return false;
    }
  }
  return true;
}

void LinkGenMaskNodesPass::GetAllGenMaskNodes(ComputeGraphPtr graph, vector<NodePtr> &gen_mask_nodes) const {
  set<NodePtr> nodes_set;
  for (const NodePtr &node : graph->GetDirectNode()) {
    bool not_domask = node->GetType() != DROPOUTDOMASK && node->GetType() != DROPOUTDOMASKV3 &&
                      node->GetType() != DROPOUTDOMASKV3D && node->GetType() != SOFTMAXV2WITHDROPOUTDOMASKV3D;
    if (not_domask) {
      continue;
    }

    if ((node->GetOpDesc() == nullptr) || (node->GetOpDesc()->HasAttr(ATTR_NAME_STREAM_LABEL))) {
      continue;
    }

    auto in_data_nodes = node->GetInDataNodes();
    if (in_data_nodes.size() > kGenMaskInputIndex) {
      NodePtr &gen_mask = in_data_nodes.at(kGenMaskInputIndex);
      for (auto &in_data_node : in_data_nodes) {
        // node gen_mask is located at different place in the fused node
        if (in_data_node->GetName().find(DROPOUTGENMASK) != in_data_node->GetName().npos) {
          gen_mask = in_data_node;
          GELOGD("The fused node type [%s], paired with the input node name [%s].",
                 node->GetType().c_str(), gen_mask->GetName().c_str());
          break;
        }
      }

      if ((gen_mask->GetOpDesc() == nullptr) || (gen_mask->GetOpDesc()->HasAttr(ATTR_NAME_STREAM_LABEL))) {
        continue;
      }
      if (AreAllInputsConst(gen_mask) && nodes_set.count(gen_mask) == 0) {
        gen_mask_nodes.emplace_back(gen_mask);
        nodes_set.emplace(gen_mask);
      }
    }
  }
}

Status LinkGenMaskNodesPass::GetGenMaskGroupSize(vector<NodePtr> &gen_mask_nodes, size_t &gen_mask_group_size) const {
  if (gen_mask_nodes.empty()) {
    return SUCCESS;
  }

  NodePtr gen_mask_node = gen_mask_nodes.front();
  GE_CHECK_NOTNULL(gen_mask_node);
  OpDescPtr gen_mask_op = gen_mask_node->GetOpDesc();
  GE_CHECK_NOTNULL(gen_mask_op);

  auto ge_lib = GELib::GetInstance();
  if ((ge_lib != nullptr) && ge_lib->InitFlag()) {
    (void)ge_lib->DNNEngineManagerObj().GetDNNEngineName(gen_mask_node);
  }

  size_t gen_mask_group_num = kDefaultMaxParallelNum;
  string engine_name = gen_mask_op->GetOpEngineName();
  auto iter = stream_max_parallel_num_.find(engine_name);
  if (iter != stream_max_parallel_num_.end()) {
    gen_mask_group_num = static_cast<size_t>(iter->second);
  }
  GELOGI("gen_mask_group_num: %zu.", gen_mask_group_num);

  if (gen_mask_group_num > 0) {
    gen_mask_group_size = (gen_mask_nodes.size() + 1) / gen_mask_group_num;
  }
  GELOGI("gen_mask_group_size: %zu.", gen_mask_group_size);

  return SUCCESS;
}
}  // namespace ge
