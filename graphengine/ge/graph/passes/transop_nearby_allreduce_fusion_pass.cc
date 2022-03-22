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

#include "graph/passes/transop_nearby_allreduce_fusion_pass.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/types.h"
#include "graph/utils/graph_utils.h"
#include "common/transop_util.h"

namespace ge {
Status TransOpNearbyAllreduceFusionPass::Run(NodePtr &node) {
  if (node == nullptr) {
    GELOGW("null node is existed in graph");
    return SUCCESS;
  }

  if (node->GetType() == HCOMALLREDUCE || node->GetType() == HVDCALLBACKALLREDUCE) {
    GELOGI("found allreduce op %s", node->GetName().c_str());
    Status ret = RemoveNearbyPairedTransOps(node);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Remove][PairedTransOp] for allreduce op:%s", node->GetName().c_str());
      return FAILED;
    }
    GELOGI("successfully remove paired transop for allreduce op (%s)", node->GetName().c_str());
  }

  return SUCCESS;
}

bool TransOpNearbyAllreduceFusionPass::IsSymmetricTransOps(const NodePtr &node1, const NodePtr &node2) {
  if (node1 == nullptr || node2 == nullptr || node1->GetOpDesc() == nullptr || node2->GetOpDesc() == nullptr) {
    return false;
  }

  if (node1->GetType() != TRANSDATA || node2->GetType() != TRANSDATA) {
    return false;
  }

  // two symmetric trans ops should have same type
  if (node1->GetType() != node2->GetType()) {
    return false;
  }

  const auto &node1_input_desc = node1->GetOpDesc()->MutableInputDesc(0);
  const auto &node1_output_desc = node1->GetOpDesc()->MutableOutputDesc(0);
  GE_CHECK_NOTNULL_EXEC(node1_input_desc, return false);
  GE_CHECK_NOTNULL_EXEC(node1_output_desc, return false);

  const auto &node2_input_desc = node2->GetOpDesc()->MutableInputDesc(0);
  const auto &node2_output_desc = node2->GetOpDesc()->MutableOutputDesc(0);
  GE_CHECK_NOTNULL_EXEC(node2_input_desc, return false);
  GE_CHECK_NOTNULL_EXEC(node2_output_desc, return false);

  // two symmetric trans ops should have symmetric input/output datatype
  GELOGD("format: nod1_input=%d, nod1_output=%d, nod2_input=%d, nod2_output=%d",
         node1_input_desc->GetFormat(), node1_output_desc->GetFormat(), node2_input_desc->GetFormat(),
         node2_output_desc->GetFormat());
  if (node1_input_desc->GetFormat() != node2_output_desc->GetFormat() ||
      node1_output_desc->GetFormat() != node2_input_desc->GetFormat()) {
    return false;
  }

  // two symmetric trans ops should have symmetric input/output format
  GELOGD("datatype: nod1_input=%d, nod1_output=%d, nod2_input=%d, nod2_output=%d",
         node1_input_desc->GetDataType(), node1_output_desc->GetDataType(), node2_input_desc->GetDataType(),
         node2_output_desc->GetDataType());
  if (node1_input_desc->GetDataType() != node2_output_desc->GetDataType() ||
      node1_output_desc->GetDataType() != node2_input_desc->GetDataType()) {
    return false;
  }

  // two symmetric trans ops should have symmetric input/output shape
  if (node1_input_desc->GetShape().GetDims() != node2_output_desc->GetShape().GetDims() ||
      node1_output_desc->GetShape().GetDims() != node2_input_desc->GetShape().GetDims()) {
    return false;
  }
  return true;
}

Status TransOpNearbyAllreduceFusionPass::RemoveNearbyPairedTransOps(const NodePtr &node) {
  if (node == nullptr) {
    return FAILED;
  }
  GELOGI("find allReduce node %s", node->GetName().c_str());
  auto in_data_anchors = node->GetAllInDataAnchors();
  auto out_data_anchors = node->GetAllOutDataAnchors();
  if (in_data_anchors.size() != out_data_anchors.size()) {
    REPORT_INNER_ERROR("E19999", "In data anchors size:%zu not equal to out data anchors size:%zu in node:%s(%s), "
                       "check invalid", in_data_anchors.size(), out_data_anchors.size(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] in and out data anchor size are not equal, node=%s, in_size=%zu, out_size=%zu",
           node->GetName().c_str(), in_data_anchors.size(), out_data_anchors.size());
    return FAILED;
  }

  size_t data_anchor_size = in_data_anchors.size();
  GELOGI("node = %s, data_anchor_size = %zu", node->GetName().c_str(), data_anchor_size);

  size_t removed_node_count = 0;
  for (size_t i = 0; i < data_anchor_size; i++) {
    if (in_data_anchors.at(i) == nullptr || out_data_anchors.at(i) == nullptr) {
      GELOGW("node=%s has a null anchor at idx=%zu", node->GetName().c_str(), i);
      continue;
    }
    if (in_data_anchors.at(i)->GetPeerAnchors().size() != 1) {
      GELOGW("nodes=%s has abnormal in peer anchors at %zu", node->GetName().c_str(), i);
      continue;
    }
    if (out_data_anchors.at(i)->GetPeerAnchors().size() != 1) {
      GELOGW("nodes=%s has abnormal out peer anchors at %zu", node->GetName().c_str(), i);
      continue;
    }
    auto in_first_peer_anchor = in_data_anchors.at(i)->GetFirstPeerAnchor();
    if (in_first_peer_anchor == nullptr) {
      GELOGW("node=%s, input anchor idx=%zu, first peer anchor is null", node->GetName().c_str(), i);
      continue;
    }
    auto out_first_peer_anchor = out_data_anchors.at(i)->GetFirstPeerAnchor();
    if (out_first_peer_anchor == nullptr) {
      GELOGW("node=%s, output anchor idx=%zu, first peer anchor is null", node->GetName().c_str(), i);
      continue;
    }
    auto in_node = in_first_peer_anchor->GetOwnerNode();
    auto out_node = out_first_peer_anchor->GetOwnerNode();

    GELOGI("in_node=%s, out_node=%s", in_node->GetName().c_str(), out_node->GetName().c_str());
    if (!IsSymmetricTransOps(in_node, out_node)) {
      GELOGD("ignore asymmetric transop %s and %s for node %s",
             in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str());
      continue;
    }

    // delete in_node
    if (IsolateAndDeleteNode(in_node, {0}) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Isolate and delete node:%s(%s) failed",
                        in_node->GetName().c_str(), in_node->GetType().c_str());
      GELOGE(FAILED, "[Remove][Node] %s failed", in_node->GetName().c_str());
      return FAILED;
    }
    removed_node_count++;

    // delete out_node
    if (IsolateAndDeleteNode(out_node, {0}) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Isolate and delete node:%s(%s) failed",
                        out_node->GetName().c_str(), out_node->GetType().c_str());
      GELOGE(FAILED, "[Remove][Node] %s failed", out_node->GetName().c_str());
      return FAILED;
    }
    removed_node_count++;

    // update allreduce input/output desc
    GE_CHECK_NOTNULL(node->GetOpDesc());
    GE_CHECK_NOTNULL(in_node->GetOpDesc());
    GE_CHECK_NOTNULL(out_node->GetOpDesc());
    auto input_desc = in_node->GetOpDesc()->GetInputDesc(0);
    auto output_desc = out_node->GetOpDesc()->GetOutputDesc(0);
    if (node->GetOpDesc()->UpdateInputDesc(static_cast<uint32_t>(i), input_desc) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Update input:%zu desc in op:%s(%s) failed",
                        i, node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Update][InputDesc] in op:%s(%s) failed, input index:%zu",
             node->GetName().c_str(), node->GetType().c_str(), i);
    }
    if (node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(i), output_desc) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Update output:%zu desc in op:%s(%s) failed",
                        i, node->GetName().c_str(), node->GetType().c_str());
      GELOGE(FAILED, "[Update][OutputDesc] in op:%s(%s) failed, input index:%zu",
             node->GetName().c_str(), node->GetType().c_str(), i);
    }
    GELOGI("successfully remove paired transop (%s and %s) for node %s",
           in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str());
  }
  GELOGI("successfully remove %zu pair of transops in total for node %s", removed_node_count, node->GetName().c_str());
  return SUCCESS;
}
}  // namespace ge
