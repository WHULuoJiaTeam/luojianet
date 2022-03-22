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

#include "graph/utils/graph_utils.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <queue>
#include <atomic>
#include <mutex>

#include "graph/ge_context.h"
#include "graph/debug/ge_util.h"
#include "proto/ge_ir.pb.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/dumper/ge_graph_dumper.h"
#include "graph/debug/ge_op_types.h"
#include "external/ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph_impl.h"
#include "graph/op_desc_impl.h"
#include "mmpa/mmpa_api.h"

namespace ge {
enum class DumpGraphLevel {
  kDumpLevel1 = 1,
  kDumpLevel2,
  kDumpLevel3,
  kDumpLevelOther,
};

namespace{
const int32_t kBaseOfIntegerValue = 10;
#ifdef FMK_SUPPORT_DUMP
const char_t *const kDumpGeGraph = "DUMP_GE_GRAPH";
const int32_t kDumpGraphIndexWidth = 8;
#endif

const char_t *const kNpuCollectPath = "NPU_COLLECT_PATH";
const char_t *const kDumpGraphPath = "DUMP_GRAPH_PATH";
const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
const char_t *const kDumpStrBuild = "Build";
const char_t *const kDumpStrPartition = "partition";
const char_t *const kDumpStrOptimizeSubgraph = "OptimizeSubGraph";
const char_t *const kDumpStrSubgraphFunc = "sub_graph";
const char_t *const kDumpStrAicpu = "Aicpu";
#ifdef __GNUC__
const char_t *const KDumpSeparator = "/";
#else
const char_t *const KDumpSeparator = "\\";
#endif
const size_t kNameMax = 255U;
const int32_t kCopyGraphMaxRecursionDepth = 10;
const int32_t kNameWidth = 5;
const std::set<std::string> kMergeInputSkipTypes{ STREAMACTIVE, STREAMSWITCH, CONSTANT, CONSTANTOP };
};

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const InDataAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "addedge failed because param src is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const AnchorPtr &src,
                                                                               const AnchorPtr &dst) {
  const OutDataAnchorPtr src_data = Anchor::DynamicAnchorCast<OutDataAnchor>(src);
  const InDataAnchorPtr dst_data = Anchor::DynamicAnchorCast<InDataAnchor>(dst);
  const OutControlAnchorPtr src_control = Anchor::DynamicAnchorCast<OutControlAnchor>(src);
  const InControlAnchorPtr dst_control = Anchor::DynamicAnchorCast<InControlAnchor>(dst);
  if ((src_data != nullptr) && (dst_data != nullptr) && (src_data->LinkTo(dst_data) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_data != nullptr) && (dst_control != nullptr) && (src_data->LinkTo(dst_control) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_control != nullptr) && (dst_control != nullptr) && (src_control->LinkTo(dst_control) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_control != nullptr) && (dst_data != nullptr) && (src_control->LinkTo(dst_data) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "AddEdge failed because src or dst is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutControlAnchorPtr &src,
                                                                               const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "AddEdge failed because src is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "AddEdge failed because src is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,
                                                                                  const InDataAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(GRAPH_FAILED, "[Remove][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const AnchorPtr &src,
                                                                                  const AnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(GRAPH_FAILED, "[Remove][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutControlAnchorPtr &src,
                                                                                  const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(GRAPH_FAILED, "[Remove][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,
                                                                                  const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "[Remove][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::ReplaceEdgeSrc(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                       const OutDataAnchorPtr &new_src) {
  if ((RemoveEdge(src, dst) == GRAPH_SUCCESS) && (AddEdge(new_src, dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "[Replace][EdgeSrc] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::ReplaceEdgeSrc(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                       const OutControlAnchorPtr &new_src) {
  if ((RemoveEdge(src, dst) == GRAPH_SUCCESS) && (AddEdge(new_src, dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "[Replace][EdgeSrc] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::ReplaceEdgeDst(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                       const InDataAnchorPtr &new_dst) {
  if ((RemoveEdge(src, dst) == GRAPH_SUCCESS) && (AddEdge(src, new_dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "[Replace][EdgeDst] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::ReplaceEdgeDst(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                       const InControlAnchorPtr &new_dst) {
  if ((RemoveEdge(src, dst) == GRAPH_SUCCESS) && (AddEdge(src, new_dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "[Replace][EdgeDst] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::InsertNodeBetweenDataAnchors(
    const OutDataAnchorPtr &src, const InDataAnchorPtr &dst, const NodePtr &new_node) {
  GE_CHECK_NOTNULL(src);
  GE_CHECK_NOTNULL(dst);
  GE_CHECK_NOTNULL(new_node);

  const InDataAnchorPtr node_in_anchor = new_node->GetInDataAnchor(0);
  GE_CHK_BOOL_RET_STATUS(node_in_anchor != nullptr, GRAPH_FAILED,
                         "[Invoke][GetInDataAnchor] this node has not inDataAnchor");
  const OutDataAnchorPtr node_out_anchor = new_node->GetOutDataAnchor(0);
  GE_CHK_BOOL_RET_STATUS(node_out_anchor != nullptr, GRAPH_FAILED,
                         "[Invoke][GetOutDataAnchor] this node has not outDataAnchor");
  GE_CHK_STATUS_RET(src->ReplacePeer(dst, node_in_anchor, node_out_anchor), "[Replace][Peer] Failed");
  return GRAPH_SUCCESS;
}


GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveSubgraphRecursively(const ComputeGraphPtr &compute_graph,
                                      const NodePtr &remove_node) {
  GE_CHECK_NOTNULL(compute_graph);
  if (remove_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param remove node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  // Check if this node is belong to this compute graph, maybe a little slow
  const auto &all_nodes_in_graph = compute_graph->GetDirectNode();
  if (std::find(all_nodes_in_graph.begin(), all_nodes_in_graph.end(), remove_node) == all_nodes_in_graph.end()) {
    REPORT_INNER_ERROR("E19999", "Can not find node %s in graph %s.",
                       remove_node->GetName().c_str(), compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Can not find node %s in graph %s.",
           remove_node->GetName().c_str(), compute_graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  // Find all subgraph of this node
  const auto &root_graph = GraphUtils::FindRootGraph(compute_graph);
  std::vector<ComputeGraphPtr> subgraphs;
  std::vector<NodePtr> all_nodes;
  std::deque<NodePtr> candidates;
  NodePtr remove_node_new = remove_node;
  candidates.emplace_back(remove_node_new);
  while (!candidates.empty()) {
    const NodePtr node = candidates.front();
    all_nodes.emplace_back(node);
    candidates.pop_front();

    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }

    const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
    for (auto name_iter = subgraph_names.rbegin(); name_iter != subgraph_names.rend(); ++name_iter) {
      auto subgraph = root_graph->GetSubgraph(*name_iter);
      if (subgraph != nullptr && subgraph->impl_ != nullptr) {
        subgraphs.emplace_back(subgraph);
        (void)candidates.insert(candidates.begin(), subgraph->impl_->nodes_.begin(), subgraph->impl_->nodes_.end());
      }
    }
  }
  // Remove all subgraph
  for (const auto &remove_graph : subgraphs) {
    if (root_graph->RemoveSubGraph(remove_graph) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "RemoveSubGraph failed, sub graph name is %s, compute graph is %s.",
                        remove_node->GetName().c_str(), compute_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Remove][SubGraph] failed, sub graph name is %s, compute graph is %s.",
             remove_node->GetName().c_str(), compute_graph->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveNodeWithoutRelink(const ComputeGraphPtr &compute_graph, const NodePtr &node) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(compute_graph->impl_);
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  // If the node save as input node, delete it
  (void)compute_graph->RemoveInputNode(node);

  // If the node save as output node, delete it
  (void)compute_graph->RemoveOutputNode(node);

  // If the node has sub-graphs, delete them
  const auto ret = RemoveSubgraphRecursively(compute_graph, node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Remove][SubGraph] recursively failed.");
    return GRAPH_FAILED;
  }

  const auto iter = find(compute_graph->impl_->nodes_.begin(), compute_graph->impl_->nodes_.end(), node);
  if (iter != compute_graph->impl_->nodes_.end()) {
    compute_graph->EraseFromNodeList(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

///
/// @brief Insert node: src->insert_node:input_index, insert_node:output_index->dst
/// @param [in] src
/// @param [in] dsts
/// @param [in] insert_node
/// @param [in] input_index
/// @param [in] output_index
/// @return graphStatus
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::InsertNodeAfter(const OutDataAnchorPtr &src,
    const std::vector<InDataAnchorPtr> &dsts, const NodePtr &insert_node, uint32_t input_index, uint32_t output_index) {
  GE_CHECK_NOTNULL(src);
  GE_CHECK_NOTNULL(insert_node);

  const NodePtr src_node = src->GetOwnerNode();
  GE_CHECK_NOTNULL(src_node);
  if (src_node->GetOwnerComputeGraph() != insert_node->GetOwnerComputeGraph()) {
    REPORT_INNER_ERROR("E19999", "src:%s and insert_node:%s not exist in the same graph.",
                       src_node->GetName().c_str(), insert_node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] src:%s and insert_node:%s not exist in the same graph.",
           src_node->GetName().c_str(), insert_node->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (AddEdge(src, insert_node->GetInDataAnchor(static_cast<int32_t>(input_index))) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddEdge %s->%s failed.", src_node->GetName().c_str(), insert_node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][Edge] %s->%s failed.", src_node->GetName().c_str(), insert_node->GetName().c_str());
    return GRAPH_FAILED;
  }

  const OutControlAnchorPtr src_out_ctrl_anchor = src_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(src_out_ctrl_anchor);

  bool ctrl_edge_flag = true;
  const std::string type = NodeUtils::GetNodeType(src->GetOwnerNode());
  if ((type == SWITCH) || (type == REFSWITCH) || (type == SWITCHN)) {
    ctrl_edge_flag = false;
  }

  for (auto &dst : dsts) {
    GE_CHECK_NOTNULL(dst);
    const NodePtr dst_node = dst->GetOwnerNode();
    GELOGI("Insert node %s between %s->%s.",
           insert_node->GetName().c_str(), src_node->GetName().c_str(), dst_node->GetName().c_str());
    if (src_node->GetOwnerComputeGraph() != dst_node->GetOwnerComputeGraph()) {
      REPORT_INNER_ERROR("E19999", "src:%s and dst:%s not exist in the same graph.",
                         src_node->GetName().c_str(), dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] src:%s and dst:%s not exist in the same graph.",
             src_node->GetName().c_str(), dst_node->GetName().c_str());
      return GRAPH_FAILED;
    }

    (void)RemoveEdge(src, dst);
    if (AddEdge(insert_node->GetOutDataAnchor(static_cast<int32_t>(output_index)), dst) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "ReplaceEdge from %s->%s to %s->%s failed.", src_node->GetName().c_str(),
                        dst_node->GetName().c_str(), insert_node->GetName().c_str(), dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Replace][Edge] from %s->%s to %s->%s failed.", src_node->GetName().c_str(),
             dst_node->GetName().c_str(), insert_node->GetName().c_str(), dst_node->GetName().c_str());
      return GRAPH_FAILED;
    }

    if (!ctrl_edge_flag) { continue; }
    for (const InControlAnchorPtr& peer_in_ctrl_anchor : src_out_ctrl_anchor->GetPeerInControlAnchors()) {
      if ((RemoveEdge(src_out_ctrl_anchor, peer_in_ctrl_anchor) != GRAPH_SUCCESS) ||
          (AddEdge(insert_node->GetOutControlAnchor(), peer_in_ctrl_anchor) != GRAPH_SUCCESS)) {
        REPORT_CALL_ERROR("E19999", "ReplaceEdge from %s->%s to %s->%s failed.",
                          src_node->GetName().c_str(), peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                          insert_node->GetName().c_str(), peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Replace][Edge] from %s->%s to %s->%s failed.",
               src_node->GetName().c_str(), peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
               insert_node->GetName().c_str(), peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus GraphUtils::InsertNodeBefore(const InDataAnchorPtr &dst,
                                         const NodePtr &insert_node,
                                         uint32_t input_index,
                                         uint32_t output_index) {
  GE_CHECK_NOTNULL(dst);
  GE_CHECK_NOTNULL(insert_node);
  const auto dst_node = dst->GetOwnerNode();
  GE_CHECK_NOTNULL(dst_node);
  if (dst_node->GetOwnerComputeGraph() != insert_node->GetOwnerComputeGraph()) {
    GELOGE(GRAPH_FAILED, "[INSERT][NODE] dst:%s and insert_node:%s not exist in the same graph.",
           dst_node->GetName().c_str(), insert_node->GetName().c_str());
    return GRAPH_FAILED;
  }

  const auto src_node_out_anchor = dst->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(src_node_out_anchor);
  const auto src_node = src_node_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(src_node);

  // insert node
  if ((RemoveEdge(src_node_out_anchor, dst) != GRAPH_SUCCESS) ||
      (AddEdge(src_node_out_anchor,
               insert_node->GetInDataAnchor(static_cast<int32_t>(input_index))) != GRAPH_SUCCESS) ||
      (AddEdge(insert_node->GetOutDataAnchor(static_cast<int32_t>(output_index)), dst) != GRAPH_SUCCESS)) {
    GELOGE(GRAPH_FAILED, "[INSERT][NODE] %s between %s->%s failed",
           insert_node->GetName().c_str(),
           src_node->GetName().c_str(),
           dst_node->GetName().c_str());
    return GRAPH_FAILED;
  }
  GELOGI("[INSERT][NODE] %s between %s->%s",
         insert_node->GetName().c_str(),
         src_node->GetName().c_str(),
         dst_node->GetName().c_str());

  // update control edges
  const auto in_ctrl_anchor = dst_node->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  const auto insert_node_in_ctrl_anchor = insert_node->GetInControlAnchor();
  for (const auto &peer_out_ctrl_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
    const auto peer_node = peer_out_ctrl_anchor->GetOwnerNode();
    const auto node_type = NodeUtils::GetNodeType(peer_node);
    if (node_type == ATOMICADDRCLEAN) {
      continue;
    }
    if ((RemoveEdge(peer_out_ctrl_anchor, in_ctrl_anchor) != GRAPH_SUCCESS) ||
        (AddEdge(peer_out_ctrl_anchor, insert_node_in_ctrl_anchor) != GRAPH_SUCCESS)) {
      GELOGE(GRAPH_FAILED, "[INSERT][NODE] replace control edge from %s->%s to %s->%s failed.",
             peer_node != nullptr ? peer_node->GetName().c_str() : "NULL",
             dst_node->GetName().c_str(),
             peer_node != nullptr ? peer_node->GetName().c_str() : "NULL",
             insert_node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveJustNode(ComputeGraph &compute_graph,
                                                                                      const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should be not null.");
    return GRAPH_FAILED;
  }
  if (compute_graph.impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "The compute graph impl should be not null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] The compute graph impl should be not null.");
    return GRAPH_FAILED;
  }
  const auto iter = find(compute_graph.impl_->nodes_.begin(), compute_graph.impl_->nodes_.end(), node);
  if (iter != compute_graph.impl_->nodes_.end()) {
    compute_graph.EraseFromNodeList(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveJustNode(ComputeGraphPtr compute_graph,
                                                                                      const NodePtr &node) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(node);
  const graphStatus ret = ((RemoveJustNode(*compute_graph, node) == GRAPH_SUCCESS) ? GRAPH_SUCCESS : GRAPH_FAILED);
  return ret;
}

void GraphUtils::RecordOriginalNames(std::vector<ge::NodePtr> original_nodes, const ge::NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
                   return, "[Check][Param] node is null.");
  std::vector<std::string> original_names;
  for (const auto &node_tmp : original_nodes) {
    std::vector<std::string> names_tmp;
    const ge::OpDescPtr opdesc_tmp = node_tmp->GetOpDesc();
    if (opdesc_tmp == nullptr) {
      GELOGE(GRAPH_FAILED, "[Check][Param] Node %s get opdesc is nullptr", node_tmp->GetName().c_str());
      continue;
    }
    const auto ret = ge::AttrUtils::GetListStr(opdesc_tmp, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names_tmp);
    if (!ret) {
      GELOGW("[Get][Attr] Get attr _datadump_original_op_names failed");
      continue;
    }
    if (names_tmp.size() != 0UL) {
      (void)original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
    } else {
      original_names.push_back(opdesc_tmp->GetName());
    }
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names),
                   REPORT_INNER_ERROR("E19999", "Set original_op_names to node:%s fail.", node->GetName().c_str());
                   return, "[Invoke][SetListStr] Set original_op_names to node:%s fail.", node->GetName().c_str());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::RecordOriginalNames(std::vector<std::string> names_tmp,
                                                                                    const ge::NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
                   return, "[Check][Param] node is null.");
  std::vector<std::string> original_names;
  if (names_tmp.size() != 0UL) {
    (void)original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
  } else {
    const std::string tmp;
    original_names.push_back(tmp);
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names),
                   REPORT_INNER_ERROR("E19999", "Set original_op_names to node %s fail.", node->GetName().c_str());
                   return, "[Invoke][SetListStr] Set original_op_names to node %s fail.", node->GetName().c_str());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::MatchDumpStr(const std::string &suffix) {
  char_t dump_level[MMPA_MAX_PATH] = { '\0' };
  const INT32 res = mmGetEnv(kDumpGraphLevel, &(dump_level[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  const int64_t dump_graph_level = (res == EN_OK) ? std::strtol(&(dump_level[0U]), nullptr, kBaseOfIntegerValue)
                                                  : static_cast<int64_t>(DumpGraphLevel::kDumpLevel2);

  if (dump_graph_level == static_cast<int64_t>(DumpGraphLevel::kDumpLevel1)) {
    return false;
  }

  if ((dump_graph_level == static_cast<int64_t>(DumpGraphLevel::kDumpLevel2)) &&
      ((suffix.find(kDumpStrPartition) != std::string::npos) ||
       (suffix.find(kDumpStrOptimizeSubgraph) != std::string::npos) ||
       (suffix.find(kDumpStrAicpu) != std::string::npos) ||
       (suffix.find(kDumpStrSubgraphFunc) != std::string::npos))) {
    return true;
  }

  if ((dump_graph_level == static_cast<int64_t>(DumpGraphLevel::kDumpLevel3)) && (suffix.compare(kDumpStrBuild) != 0)) {
    return true;
  }

  return false;
}

namespace {
void GetDumpGraphPrefix(std::stringstream& stream_file_name) {
  static std::string path_prefix;
  if (path_prefix.empty()) {
    char_t npu_collect_path[MMPA_MAX_PATH] = { '\0' };
    INT32 res = mmGetEnv(kNpuCollectPath, &(npu_collect_path[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    if (res == EN_OK) {
      const std::string base_path_str(npu_collect_path);
      stream_file_name << base_path_str << "/extra-info/graph/" << mmGetPid() << "_" << GetContext().DeviceId() << "/";
    } else {
      char_t dump_graph_path[MMPA_MAX_PATH] = { '\0' };
      res = mmGetEnv(kDumpGraphPath, &(dump_graph_path[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
      if (res == EN_OK) {
        const std::string dump_graph_path_str(dump_graph_path);
        stream_file_name << (dump_graph_path_str.empty() ? "" : dump_graph_path_str + "/");
      } else {
        stream_file_name << "./";
      }
    }
    path_prefix = stream_file_name.str();
  } else {
    stream_file_name << path_prefix;
  }
}

inline graphStatus CheckDumpGraphNum(int64_t file_index) {
  thread_local int64_t max_dump_file_num = 0;
  if (max_dump_file_num == 0) {
    std::string opt = "0";
    (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_NUM, opt);
    max_dump_file_num = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
  }
  if ((max_dump_file_num != 0) && (file_index > max_dump_file_num)) {
    GELOGW("[DumpGraph][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%ld, max_dump_file_num=%ld",
           file_index, max_dump_file_num);
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraph(const ge::ComputeGraphPtr &graph,
                                                                            const std::string &suffix,
                                                                            bool is_always_dump,
                                                                            const std::string &user_graph_name) {
#ifdef FMK_SUPPORT_DUMP
  GraphDumperRegistry::GetDumper().Dump(graph, suffix);
  char_t dump_ge_graph[MMPA_MAX_PATH] = { '\0' };
  const INT32 res = mmGetEnv(kDumpGeGraph, &(dump_ge_graph[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  if ((res != EN_OK) && (!is_always_dump)) {
    return;
  }

  // dump the graph according to different graph level
  const bool not_dump = GraphUtils::MatchDumpStr(suffix) && (!is_always_dump);
  if (not_dump) {
    return;
  }

  // file name
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump om txt: %ld", file_index);
  if (CheckDumpGraphNum(file_index) != GRAPH_SUCCESS) { return; }

  std::stringstream stream_file_name;
  {
    static std::mutex mutex;
    const std::lock_guard<std::mutex> lock(mutex);
    GetDumpGraphPrefix(stream_file_name);
    if (mmAccess2(stream_file_name.str().c_str(), M_F_OK) != EN_OK) {
      const int32_t ret = CreateDirectory(stream_file_name.str());
      if (ret != 0) {
        GELOGW("[DumpGraph][CreateDirectory] Create dump graph dir failed, path:%s", stream_file_name.str().c_str());
        stream_file_name.str("");
        stream_file_name << "./";
      }
    }
  }

  stream_file_name << "ge_proto_" << std::setw(kDumpGraphIndexWidth) << std::setfill('0') << file_index;
  stream_file_name << "_" << suffix << ".txt";
  const std::string proto_file = user_graph_name.empty() ? stream_file_name.str() : user_graph_name;

  // Create buffer
  ge::Model model("", "");
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(graph)));
  Buffer buffer;
  const int64_t dump_level =
      (dump_ge_graph != nullptr) ? std::strtol(&(dump_ge_graph[0U]), nullptr, kBaseOfIntegerValue)
                                 : ge::OnnxUtils::NO_DUMP;
  (void)model.Save(buffer, (dump_level != ge::OnnxUtils::DUMP_ALL) && (!is_always_dump));

  // Write file
  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    const std::string str(reinterpret_cast<const char_t *>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      GELOGE(GRAPH_FAILED, "[Invoke][Parse] parse from std::string failed.");
      return;
    }
    char_t real_path[MMPA_MAX_PATH] = {'\0'};
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strnlen(proto_file.c_str(), sizeof(real_path)) >= sizeof(real_path),
                                   REPORT_INNER_ERROR("E19999", "file path is too longer! file:%s", proto_file.c_str());
                                   return, "[Check][Param] file path is too longer!");
    GE_IF_BOOL_EXEC(mmRealPath(proto_file.c_str(), &(real_path[0U]), MMPA_MAX_PATH) != EN_OK,
                    GELOGI("file %s does not exist, it will be created.", proto_file.c_str()));

    GraphUtils::WriteProtoToTextFile(ge_proto, &(real_path[0U]));
  }
#else
  GELOGW("[DumpGraph][Check] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::DumpGEGraphByPath(const ge::ComputeGraphPtr &graph, const std::string &file_path,
                              const int64_t dump_level) {
  const auto sep = file_path.rfind(KDumpSeparator);
  if (sep == std::string::npos) {
    REPORT_INPUT_ERROR("E19026", std::vector<std::string>({"pathname", "reason"}),
                       std::vector<std::string>({
                       file_path.c_str(),
                       "Separator is not found in file_path."}));
    GELOGE(GRAPH_FAILED, "[CheckParam] Separator is not found in file_path.file_path:%s", file_path.c_str());
    return GRAPH_FAILED;
  }
  const std::string file_name = file_path.substr(sep + 1UL, file_path.length());
  const std::string path_dir = file_path.substr(0UL, sep + 1UL);
  if ((file_name.length() == 0) || (path_dir.length() == 0)) {
    REPORT_INPUT_ERROR("E19026", std::vector<std::string>({"pathname", "reason"}),
                       std::vector<std::string>({
                       file_path.c_str(),
                       "Path or filename is not set."}));
    GELOGE(GRAPH_FAILED, "[Invalid]path or name invalid.file_path:%s", file_path.c_str());
    return GRAPH_FAILED;
  }

  // Create buffer
  ge::Model model("", "");
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(graph)));
  Buffer buffer;
  (void)model.Save(buffer, dump_level != ge::OnnxUtils::DUMP_ALL);

  // Write file
  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    const std::string str(reinterpret_cast<const char_t *>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      GELOGE(GRAPH_FAILED, "[Invoke][Parse] parse from std::string failed.");
      return GRAPH_FAILED;
    }
    char_t real_path[MMPA_MAX_PATH] = {'\0'};
    if (mmRealPath(path_dir.c_str(), &(real_path[0U]), MMPA_MAX_PATH) != EN_OK) {
      REPORT_INPUT_ERROR("E19026", std::vector<std::string>({"pathname", "reason"}),
                         std::vector<std::string>({
                         path_dir.c_str(),
                         "Directory does not exist."}));
      GELOGE(GRAPH_FAILED, "[Get][RealPath]Directory %s does not exist.", path_dir.c_str());
      return GRAPH_FAILED;
    }
    const std::string path = real_path;
    const std::string real_path_name = path + std::string(KDumpSeparator) + file_name;
    GraphUtils::WriteProtoToTextFile(ge_proto, real_path_name.c_str());
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGrph(const ge::ComputeGraphPtr &graph,
                                                                           const std::string &path,
                                                                           const std::string &suffix) {
  // file name
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump om txt: %ld", file_index);
  if (CheckDumpGraphNum(file_index) != GRAPH_SUCCESS) { return; }

  std::stringstream stream_file_name;
  stream_file_name << path.c_str() << "/ge_proto_" << std::setw(kNameWidth) << std::setfill('0')
                   << file_index;
  stream_file_name << "_" << suffix << ".txt";
  const std::string proto_file = stream_file_name.str();
  (void)DumpGEGraphByPath(graph, proto_file, ge::OnnxUtils::NO_DUMP);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraph(const char_t *file,
                                                                            ge::ComputeGraph &compute_graph) {
  ge::proto::ModelDef model_def;
  // Get ModelDef object from file generated by DumpGEGraph()
  if (!ReadProtoFromTextFile(file, &model_def)) {
    GELOGE(GRAPH_FAILED, "[Get][ModelDef] failed from file:%s", file);
    return false;
  }
  ge::Model model;
  // Get Model object from ModelDef by deserialize ModelDef
  if (model.Load(model_def) == GRAPH_SUCCESS) {
    GE_CHK_BOOL_EXEC(GraphUtils::GetComputeGraph(model.GetGraph()) != nullptr,
                     REPORT_INNER_ERROR("E19999", "Get computer graph is nullptr, model file:%s.", file);
                     return false, "[Get][ComputerGraph] is nullptr");
    compute_graph = *(GraphUtils::GetComputeGraph(model.GetGraph()));
    return true;
  } else {
    REPORT_CALL_ERROR("E19999", "Get Model failed from ModelDef:%s.", file);
    GELOGE(GRAPH_FAILED, "[Get][Model] failed from ModelDef:%s", file);
    return false;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraph(const char_t *file,
                                                                            ge::ComputeGraphPtr &compute_graph) {
  ge::proto::ModelDef model_def;
  // Get ModelDef object from file generated by DumpGEGraph()
  if (!ReadProtoFromTextFile(file, &model_def)) {
    GELOGE(GRAPH_FAILED, "[Get][ModelDef] failed from file:%s", file);
    return false;
  }
  ge::Model model;
  // Get Model object from ModelDef by deserialize ModelDef
  if (model.Load(model_def) == GRAPH_SUCCESS) {
    GE_CHK_BOOL_EXEC(GraphUtils::GetComputeGraph(model.GetGraph()) != nullptr,
                     REPORT_INNER_ERROR("E19999", "Get computer graph is nullptr, model file:%s.", file);
                     return false, "[Get][ComputerGraph] is nullptr");
    compute_graph = GraphUtils::GetComputeGraph(model.GetGraph());
    for (const auto &node : compute_graph->GetDirectNode()) {
      if (node == nullptr) {
        REPORT_INNER_ERROR("E19999", "ModeDef %s has nullptr node.", file);
        GELOGE(GRAPH_FAILED, "[Get][Node]Nullptr node in graph:%s, model:%s", compute_graph->GetName().c_str(), file);
        return false;
      }
      GELOGI("Node %s set owner graph", node->GetName().c_str());
      if (node->SetOwnerComputeGraph(compute_graph) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "SetOwnerComputeGraph failed for node:%s", node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Invoke][SetGraph]Node %s set owner graph failed", node->GetName().c_str());
        return false;
      }
    }
    return true;
  } else {
    REPORT_CALL_ERROR("E19999", "Get Model failed from ModelDef:%s.", file);
    GELOGE(GRAPH_FAILED, "[Get][Model] failed from ModelDef:%s", file);
    return false;
  }
}

// Printing protocol messages in text format is useful for debugging and human editing of messages.
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::WriteProtoToTextFile(
    const google::protobuf::Message &proto, const char_t *real_path) {
#ifdef FMK_SUPPORT_DUMP
  const MODE FILE_AUTHORITY = 384U; // 0600U in octal
  const int32_t fd = mmOpen2(real_path, M_WRONLY | M_CREAT | O_TRUNC, FILE_AUTHORITY);
  if (fd < 0) {
    REPORT_CALL_ERROR("E19999", "open file:%s failed, errormessage:%s", real_path, strerror(errno));
    GELOGE(GRAPH_FAILED, "[Open][File] failed for %s, reason:%s", real_path, strerror(errno));
    return;
  }
  google::protobuf::io::FileOutputStream *output = new (std::nothrow) google::protobuf::io::FileOutputStream(fd);
  if (output == nullptr) {
    REPORT_CALL_ERROR("E19999", "create FileOutputStream failed.");
    GELOGE(GRAPH_FAILED, "[Create][FileOutputStream] Output is nullptr");
    if (mmClose(fd) != 0) {
      REPORT_CALL_ERROR("E19999", "close FileOutputStream failed, reason:%s.", strerror(errno));
      GELOGE(GRAPH_FAILED, "[Close][FileOutputStream] failed, reason:%s", strerror(errno));
    }
    return;
  }
  const bool ret = google::protobuf::TextFormat::Print(proto, output);
  if (!ret) {
    REPORT_CALL_ERROR("E19999", "write file:%s failed.", real_path);
    GELOGE(GRAPH_FAILED, "[Invoke][Print] Fail to write the file: %s", real_path);
    delete output;
    output = nullptr;
    GE_CHK_BOOL_EXEC(mmClose(fd) == 0,
                     REPORT_CALL_ERROR("E19999", "close FileOutputStream failed, reason:%s.", strerror(errno));
                     return, "[Close][FileOutputStream] failed, reason:%s", strerror(errno));
    return;
  }
  delete output;
  output = nullptr;
  GE_CHK_BOOL_EXEC(mmClose(fd) == 0,
                   REPORT_CALL_ERROR("E19999", "close FileOutputStream failed, reason:%s.", strerror(errno));
                   return, "[Close][FileOutputStream] failed, reason:%s.", strerror(errno));

  FILE *const file = fopen(real_path, "rb");
  if (file == nullptr) {
    REPORT_CALL_ERROR("E19999", "open file:%s failed, errormessage:%s", real_path, strerror(errno));
    GELOGE(GRAPH_FAILED, "[Invoke][FOpen] fail to open the file: %s, %s", real_path, strerror(errno));
    return;
  }
  if (fseek(file, 0L, SEEK_END) == 0) {
    const int64_t fileSize = ftell(file);
    thread_local int64_t max_dump_file_size = 0;
    if (max_dump_file_size == 0) {
      std::string opt = "0";
      // Can not check return value
      (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_SIZE, opt);
      max_dump_file_size = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
    }
    if ((max_dump_file_size != 0) && (fileSize != -1) && (fileSize > max_dump_file_size)) {
      GELOGW("[WriteProto][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%ld, max_dump_file_num=%ld",
             fileSize, max_dump_file_size);
      GE_IF_BOOL_EXEC(remove(real_path) != 0, GELOGW("[WriteProto][RemovePath] Remove path %s failed", real_path));
      GE_CHK_BOOL_EXEC(fclose(file) == 0,
                       REPORT_CALL_ERROR("E19999", "close file:%s failed, error:%s", real_path, strerror(errno));
                       return, "[FClose][File] %s failed error:%s", real_path, strerror(errno));
      return;
    }
  }
  GE_CHK_BOOL_EXEC(fclose(file) == 0,
                   REPORT_CALL_ERROR("E19999", "close file:%s failed error:%s", real_path, strerror(errno));
                   return, "[FClose][File] %s failed error:%s", real_path, strerror(errno));
#else
  GELOGW("[Write][Proto] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::ReadProtoFromTextFile(
    const char_t *file, google::protobuf::Message *proto) {
  if ((file == nullptr) || (proto == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param file or proto is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] incorrect parameter. file path or message is invalid");
    return false;
  }
  std::ifstream fs(file, std::ifstream::in);
  if (!fs.is_open()) {
    REPORT_CALL_ERROR("E19999", "open file:%s failed.", file);
    GELOGE(GRAPH_FAILED, "[Invoke][OpenFile]proto file '%s' open fail.", file);
    return false;
  }
  google::protobuf::io::IstreamInputStream input(&fs);
  const bool ret = google::protobuf::TextFormat::Parse(&input, proto);
  if (!ret) {
    REPORT_INNER_ERROR("E19999", "parse proto from text ret fail, please check your text file '%s'.", file);
    GELOGE(GRAPH_FAILED, "[Parse][Proto] from text ret fail, please check your text file '%s'.", file);
  }
  fs.close();
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraphToOnnx(const ge::ComputeGraph &compute_graph,
                                                                                  const std::string &suffix) {
#ifdef FMK_SUPPORT_DUMP
  char_t dump_ge_graph[MMPA_MAX_PATH] = { '\0' };
  const INT32 res = mmGetEnv(kDumpGeGraph, &(dump_ge_graph[0]), static_cast<uint32_t>(MMPA_MAX_PATH));
  const int64_t dump_ge_graph_level =
      (res == EN_OK) ? std::strtol(&(dump_ge_graph[0U]), nullptr, kBaseOfIntegerValue) : OnnxUtils::NO_DUMP;
  if ((dump_ge_graph_level == OnnxUtils::NO_DUMP) || (dump_ge_graph_level >= OnnxUtils::DUMP_LEVEL_END)) {
    GELOGD("Skip DumpGEGraphToOnnx with dump_ge_graph_level %ld.", dump_ge_graph_level);
    return;
  }

  // dump the graph according to different graph level
  if (GraphUtils::MatchDumpStr(suffix)) {
    return;
  }

  // 1.Get ge::onnx::ModelProto from ge::Model
  ge::Model model("GE", "");
  const std::shared_ptr<ge::ComputeGraph> compute_graph_ptr = ComGraphMakeShared<ge::ComputeGraph>(compute_graph);
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(compute_graph_ptr)));
  onnx::ModelProto model_proto;
  if (!OnnxUtils::ConvertGeModelToModelProto(model, model_proto)) {
    GELOGE(GRAPH_FAILED, "[Convert][GeModel] DumpGEGraphToOnnx failed.");
    return;
  }

  // 2.Set file name
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump ge onnx file: %ld", file_index);
  if (CheckDumpGraphNum(file_index) != GRAPH_SUCCESS) { return; }

  std::stringstream stream_file_name;
  GetDumpGraphPrefix(stream_file_name);
  if (mmAccess2(stream_file_name.str().c_str(), M_F_OK) != EN_OK) {
    const int32_t ret = CreateDirectory(stream_file_name.str());
    if (ret != 0) {
      GELOGW("[DumpGraph][CreateDirectory] Create dump graph dir failed, path:%s", stream_file_name.str().c_str());
      stream_file_name.str("");
      stream_file_name << "./";
    }
  }

  stream_file_name << "ge_onnx_" << std::setw(kDumpGraphIndexWidth) << std::setfill('0') << file_index;
  stream_file_name << "_graph_" << compute_graph.GetGraphID();
  stream_file_name << "_" << suffix << ".pbtxt";
  const std::string proto_file = stream_file_name.str();
  if ((proto_file.length()) >= kNameMax) {
    GELOGE(GRAPH_FAILED, "[Check][Param] File name is too longer!, file:%s", proto_file.c_str());
    return;
  }
  std::unique_ptr<char[]> real_path(new (std::nothrow) char[MMPA_MAX_PATH]{0});
  GE_CHECK_NOTNULL_EXEC(real_path, return);

  /// Returning nullptr means 3 case as follows:
  /// a.path is MMPA_MAX_PATH chars or more
  /// b.the file does not exist
  /// c.the path has no permissions
  /// Distinguish between last the two cases in the function WriteProtoToTextFile call open()
  if (mmRealPath(proto_file.c_str(), real_path.get(), MMPA_MAX_PATH) != EN_OK) {
    // Case a has been checked above
    GELOGI("File %s does not exist, it will be created.", proto_file.c_str());
  }

  // 3. Serialize to file in current path
  GraphUtils::WriteProtoToTextFile(model_proto, real_path.get());
#else
  GELOGW("[DumpGraph][Check] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGrphToOnnx(const ge::ComputeGraph &compute_graph,
                                                                               const std::string &path,
                                                                               const std::string &suffix) {
  // 1.Get ge::onnx::ModelProto from ge::Model
  ge::Model model("GE", "");
  const std::shared_ptr<ge::ComputeGraph> compute_graph_ptr = ComGraphMakeShared<ge::ComputeGraph>(compute_graph);
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(compute_graph_ptr)));
  onnx::ModelProto model_proto;
  if (!OnnxUtils::ConvertGeModelToModelProto(model, model_proto)) {
    GELOGE(GRAPH_FAILED, "[Convert][GeModel] DumpGEGraphToOnnx failed.");
    return;
  }

  // 2.Set file name
  static std::atomic<int64_t> atomic_file_index(0);
  const auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump ge onnx file: %ld", file_index);
  if (CheckDumpGraphNum(file_index) != GRAPH_SUCCESS) { return; }

  std::stringstream stream_file_name;
  stream_file_name << path.c_str() << "/ge_onnx_" << std::setw(5) << std::setfill('0') << file_index;
  stream_file_name << "_graph_" << compute_graph.GetGraphID();
  stream_file_name << "_" << suffix << ".pbtxt";
  const std::string proto_file = stream_file_name.str();
  if ((proto_file.length()) >= kNameMax) {
    GELOGE(GRAPH_FAILED, "[Check][Param] File name is too longer!, file:%s", proto_file.c_str());
    return;
  }
  std::unique_ptr<char[]> real_path(new (std::nothrow) char[MMPA_MAX_PATH]{0});
  if (real_path == nullptr) {
    GELOGE(GRAPH_FAILED, "[New][RealPath] failed.");
    return;
  }
  /// Returning nullptr means 3 case as follows:
  /// a.path is PATH_MAX chars or more
  /// b.the file does not exist
  /// c.the path has no permissions
  /// Distinguish between last the two cases in the function WriteProtoToTextFile call open()
  if (mmRealPath(proto_file.c_str(), real_path.get(), MMPA_MAX_PATH) != EN_OK) {
    // Case a has been checked above
    GELOGI("File %s does not exist, it will be created.", proto_file.c_str());
  }

  // 3. Serialize to file in current path
  GraphUtils::WriteProtoToTextFile(model_proto, real_path.get());
}

namespace {
using InNodesToOut = std::unordered_map<NodePtr, std::unordered_set<NodePtr>>;

inline std::string GetNodeNameByAnchor(const Anchor *const anchor) {
  if (anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "param anchor is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Anchor is nullptr");
    return "Null";
  }
  const auto node = anchor->GetOwnerNode();
  return node == nullptr ? "Null" : node->GetName();
}

graphStatus ReplaceOutDataAnchor(const OutDataAnchorPtr &new_anchor, const OutDataAnchorPtr &old_anchor,
                                 InNodesToOut *const in_nodes_to_out = nullptr) {
  if (new_anchor == nullptr || old_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "param new_anchor or old_anchor is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] new_anchor or old_anchor is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  const auto new_node = new_anchor->GetOwnerNode();
  for (const auto &peer_in_anchor : old_anchor->GetPeerInDataAnchors()) {
    auto ret = peer_in_anchor->Unlink(old_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed to unlink old anchor link from %s(%d) to %s(%d)",
                        GetNodeNameByAnchor(old_anchor.get()).c_str(), old_anchor->GetIdx(),
                        GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Remove][Link] Failed to unlink old anchor link from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(old_anchor.get()).c_str(), old_anchor->GetIdx(),
             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
    ret = peer_in_anchor->LinkFrom(new_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "[Create][Link] Failed to relink new anchors from %s(%d) to %s(%d)",
                        GetNodeNameByAnchor(new_anchor.get()).c_str(), new_anchor->GetIdx(),
                        GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Create][Link] Failed to relink new anchors from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(new_anchor.get()).c_str(), new_anchor->GetIdx(),
             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }

    if (in_nodes_to_out != nullptr) {
      (void)(*in_nodes_to_out)[new_node].insert(peer_in_anchor->GetOwnerNode());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus RelinkDataIO(const NodePtr &node, const std::vector<int> &io_map, InNodesToOut &in_nodes_to_out) {
  GE_CHECK_NOTNULL(node);
  auto in_data_anchors = node->GetAllInDataAnchors();
  auto out_data_anchors = node->GetAllOutDataAnchors();
  const size_t out_data_anchors_size = out_data_anchors.size();
  if (out_data_anchors_size < io_map.size()) {
    REPORT_INNER_ERROR("E19999", "param io_map size:%zu > the actual size:%zu, node:%s type:%s",
                       io_map.size(), out_data_anchors.size(), node->GetName().c_str(), node->GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The io_map specified for node %s type %s is larger %zu than "
           "the actual size %zu", node->GetName().c_str(), node->GetType().c_str(),
           io_map.size(), out_data_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  for (size_t i = 0U; i < out_data_anchors_size; ++i) {
    const auto out_data_anchor = out_data_anchors.at(i);
    if (out_data_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to relink for node %s type %s, the out data anchor "
                         "at index %zu is null", node->GetName().c_str(), node->GetType().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to relink for node %s type %s, the out data anchor "
             "at index %zu is null", node->GetName().c_str(), node->GetType().c_str(), i);
      return GRAPH_FAILED;
    }

    int32_t in_index = -1;
    if (i < io_map.size()) {
      in_index = io_map.at(i);
    }
    if (in_index < 0) {
      out_data_anchor->UnlinkAll();
    } else {
      if (in_index >= static_cast<int32_t>(in_data_anchors.size())) {
        REPORT_INNER_ERROR("E19999", "Failed to relink for node %s type %s, invalid index %d specified for input(%zu)",
                           node->GetName().c_str(), node->GetType().c_str(), in_index, in_data_anchors.size());
        GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Failed to relink for node %s type %s, invalid index %d "
               "specified for input(%zu)", node->GetName().c_str(), node->GetType().c_str(),
               in_index, in_data_anchors.size());
        return GRAPH_PARAM_INVALID;
      }
      const auto in_anchor = in_data_anchors.at(static_cast<size_t>(in_index));
      if (in_anchor == nullptr) {
        GELOGW("[Relink][Check] %d\'th in_data_anchor of node %s type %s is null, ignore it.", in_index,
               node->GetName().c_str(), node->GetType().c_str());
        continue;
      }
      const auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
      if (peer_out_anchor == nullptr) {
        continue;
      }
      if (peer_out_anchor->Unlink(in_anchor) != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Failed relink node %s type %s, failed to unlink the data link "
                            "from %s(%d) to it at input-index %d", node->GetName().c_str(), node->GetType().c_str(),
                            GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
                            in_index);
        GELOGE(GRAPH_FAILED, "[Remove][Link] Failed relink node %s type %s, failed to unlink the data link "
               "from %s(%d) to it at input-index %d", node->GetName().c_str(), node->GetType().c_str(),
               GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(), in_index);
        return GRAPH_FAILED;
      }
      const auto ret = ReplaceOutDataAnchor(peer_out_anchor, out_data_anchor, &in_nodes_to_out);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "[Replace][OutDataAnchor] Failed to relink node %s type %s for relinking data anchors",
               node->GetName().c_str(), node->GetType().c_str());
        return GRAPH_FAILED;
      }
    }
  }

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    in_anchor->UnlinkAll();
  }
  return GRAPH_SUCCESS;
}

InNodesToOut GetFullConnectIONodes(const NodePtr &node) {
  InNodesToOut in_nodes_to_out;
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node is nullptr");
    return in_nodes_to_out;
  }
  const auto in_nodes_list = node->GetInNodes();
  auto out_nodes_list = node->GetOutNodes();
  auto out_nodes = std::unordered_set<NodePtr>(out_nodes_list.begin(), out_nodes_list.end());

  for (const auto &in_node : in_nodes_list) {
    (void)in_nodes_to_out.insert(std::make_pair(in_node, out_nodes));
  }
  return in_nodes_to_out;
}

graphStatus RelinkControlNodeIfNeed(const NodePtr &node, InNodesToOut &in_nodes_to_out,
                                    InNodesToOut &connected_data_in_to_out) {
  GE_CHECK_NOTNULL(node);
  for (const auto &in_node_to_out : in_nodes_to_out) {
    auto &in_node = in_node_to_out.first;
    GE_CHECK_NOTNULL(in_node);
    auto &connected_data_out = connected_data_in_to_out[in_node];
    for (const auto &out_node : in_node_to_out.second) {
      GE_CHECK_NOTNULL(out_node);
      if (connected_data_out.count(out_node) == 0UL) {
        GE_CHECK_NOTNULL(in_node->GetOutControlAnchor());
        if (in_node->GetOutControlAnchor()->IsLinkedWith(out_node->GetInControlAnchor())) {
          continue;
        }
        // Some pass, such as SameTransdataBreadFusionPass will generate a ring, so add a
        // ring breaking operation here, and notice, this is an operation which will be
        // delete later, so do not use this interface to break a ring
        if (in_node == out_node) {
          GELOGW("[Relink][CtrlNode] There is a cycle between %s to %s when isolating node %s type %s",
                 in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                 node->GetType().c_str());
          continue;
        }
        const auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), out_node->GetInControlAnchor());
        if (ret != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Add ControlEdge from %s to %s failed, when isolating node %s type %s",
                            in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                            node->GetType().c_str());
          GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed, when isolating node %s type %s",
                 in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                 node->GetType().c_str());
          return GRAPH_FAILED;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplaceOutDataAnchors(const Node::Vistor<OutDataAnchorPtr> &new_outs,
                                  const Node::Vistor<OutDataAnchorPtr> &old_outs,
                                  const std::vector<int32_t> &outputs_map) {
  const auto new_out_size = new_outs.size();
  if (new_out_size < outputs_map.size()) {
    REPORT_INNER_ERROR("E19999", "Failed to replace out data anchors, the actual size %zu is less than "
                       "the mapping size %zu", new_out_size, outputs_map.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Failed to replace out data anchors, the actual size %zu is less than "
           "the mapping size %zu", new_out_size, outputs_map.size());
    return GRAPH_PARAM_INVALID;
  }
  for (size_t i = 0U; i < new_out_size; ++i) {
    auto &new_out_anchor = new_outs.at(i);
    if (new_out_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to replace out data anchors, "
                         "the out data anchor on new node is null, index %zu", i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace out data anchors, "
             "the out data anchor on new node is null, index %zu", i);
      return GRAPH_FAILED;
    }
    if (i >= outputs_map.size()) {
      continue;
    }
    const auto old_index = outputs_map.at(i);
    if (old_index < 0) {
      continue;
    }

    const OutDataAnchorPtr &old_out_anchor = old_outs.at(static_cast<size_t>(old_index));
    if (old_out_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to replace out data anchors, "
                         "the out data anchor on old node is null, index %d", old_index);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace out data anchors, "
             "the out data anchor on old node is null, index %d", old_index);
      return GRAPH_FAILED;
    }
    const auto ret = ReplaceOutDataAnchor(new_out_anchor, old_out_anchor);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus ReplaceInDataAnchors(const Node::Vistor<InDataAnchorPtr> &new_ins,
                                 const Node::Vistor<InDataAnchorPtr> &old_ins,
                                 const std::vector<int32_t> &inputs_map) {
  const auto new_in_size = new_ins.size();
  if (new_in_size < inputs_map.size()) {
    REPORT_INNER_ERROR("E19999", "Failed to replace in data anchors, "
                       "the actual size %zu is less than the mapping size %zu", new_in_size, inputs_map.size());
    GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace in data anchors, "
           "the actual size %zu is less than the mapping size %zu", new_in_size, inputs_map.size());
    return GRAPH_PARAM_INVALID;
  }

  for (size_t i = 0U; i < new_in_size; ++i) {
    auto &new_in_anchor = new_ins.at(i);
    if (new_in_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to replace in data anchors, "
                         "the out data anchor on new node is null, index %zu", i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace in data anchors, "
             "the out data anchor on new node is null, index %zu", i);
      return GRAPH_FAILED;
    }
    if (i >= inputs_map.size()) {
      continue;
    }
    const auto old_index = inputs_map.at(i);
    if (old_index < 0) {
      continue;
    }
    const InDataAnchorPtr &old_in_anchor = old_ins.at(static_cast<size_t>(old_index));
    if (old_in_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to replace in data anchors, "
                         "the out data anchor on old node is null, index %d", old_index);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace in data anchors, "
             "the out data anchor on old node is null, index %d", old_index);
      return GRAPH_FAILED;
    }

    const auto peer_out_anchor = old_in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto ret = peer_out_anchor->Unlink(old_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed to unlink old anchors, unlink from %s(%d) to %s(%d)",
                        GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
                        GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Remove][Link] Failed to unlink old anchors, unlink from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
             GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
    ret = peer_out_anchor->LinkTo(new_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed to link new anchors, link from %s(%d) to %s(%d)",
                        GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
                        GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Create][Link]Failed to link new anchors, link from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
             GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplaceControlAnchors(const NodePtr &new_node, const NodePtr &old_node) {
  GE_CHECK_NOTNULL(new_node);
  GE_CHECK_NOTNULL(new_node->GetInControlAnchor());
  GE_CHECK_NOTNULL(old_node);
  GE_CHECK_NOTNULL(old_node->GetInControlAnchor());
  const auto peer_out_anchors = old_node->GetInControlAnchor()->GetPeerAnchors();
  const auto new_in_control_anchor = new_node->GetInControlAnchor();
  const auto exists_out_anchors = new_in_control_anchor->GetPeerAnchors();
  const auto exists_out_anchors_set = std::set<AnchorPtr>(exists_out_anchors.begin(), exists_out_anchors.end());
  for (const auto &peer_out_anchor : peer_out_anchors) {
    if (peer_out_anchor == nullptr) {
      continue;
    }
    if (exists_out_anchors_set.count(peer_out_anchor) > 0U) {
      continue;
    }
    const auto ret = GraphUtils::AddEdge(peer_out_anchor, new_in_control_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge from %s to %s failed, ret:%d",
                        peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                        new_in_control_anchor->GetOwnerNode()->GetName().c_str(), ret);
      GELOGE(GRAPH_FAILED, "[Add][Edge] from %s to %s failed, ret:%d",
             peer_out_anchor->GetOwnerNode()->GetName().c_str(),
             new_in_control_anchor->GetOwnerNode()->GetName().c_str(), ret);
      return GRAPH_FAILED;
    }
  }
  const auto old_out_control_anchor = old_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(old_out_control_anchor);
  const auto peer_in_anchors = old_out_control_anchor->GetPeerAnchors();
  const auto new_out_control_anchor = new_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(new_out_control_anchor);
  auto exists_in_anchors = new_out_control_anchor->GetPeerAnchors();
  const auto exists_in_anchors_set = std::set<AnchorPtr>(exists_in_anchors.begin(), exists_in_anchors.end());
  for (const auto &peer_in_anchor : peer_in_anchors) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    if (exists_in_anchors_set.count(peer_in_anchor) > 0U) {
      continue;
    }
    const auto ret = GraphUtils::AddEdge(new_out_control_anchor, peer_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddEdge from %s to %s failed, ret:%d",
                        new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(), ret);
      GELOGE(GRAPH_FAILED, "[Add][Edge] from %s to %s failed, ret:%d",
             new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(), ret);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::IsolateNode(const NodePtr &node,
                                                                                   const std::vector<int32_t> &io_map) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Failed to isolate node(null)");
    return GRAPH_PARAM_INVALID;
  }

  /// We must get full connections info before re-link data io, because the data
  /// edges may be unlinked when relink data io
  auto in_nodes_to_out = GetFullConnectIONodes(node);

  InNodesToOut data_in_to_out;
  auto ret = RelinkDataIO(node, io_map, data_in_to_out);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Relink][DataIO] failed, node %s type %s", node->GetName().c_str(), node->GetType().c_str());
    return ret;
  }

  ret = RelinkControlNodeIfNeed(node, in_nodes_to_out, data_in_to_out);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  NodeUtils::UnlinkAll(*node);

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::IsolateNode(const NodePtr &node, const std::initializer_list<int32_t> &io_map) {
  return IsolateNode(node, std::vector<int32_t>(io_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::IsolateNodeOneIO(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] incorrect parameter. node is invalid");
    return GRAPH_PARAM_INVALID;
  }
  if (node->GetAllInDataAnchorsSize() != 1U) {
    return GRAPH_PARAM_INVALID;
  }
  if (node->GetAllOutDataAnchorsSize() != 1U) {
    return GRAPH_PARAM_INVALID;
  }
  return IsolateNode(node, {0});
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node,
                               const std::vector<int32_t> &inputs_map,
                               const std::vector<int32_t> &outputs_map) {
  if ((new_node == nullptr) || (old_node == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param new_node or old_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto ret = ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  if (ret != GRAPH_SUCCESS) {
    // The error log was printed in `ReplaceNodeDataAnchors`
    return GRAPH_FAILED;
  }
  ret = ReplaceControlAnchors(new_node, old_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Replace][ControlAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s", old_node->GetName().c_str(), old_node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceNodeAnchors(
    const NodePtr &new_node, const NodePtr &old_node, const std::initializer_list<int32_t> inputs_map,
    const std::initializer_list<int32_t> outputs_map) {
  return ReplaceNodeAnchors(new_node, old_node,
                            std::vector<int32_t>(inputs_map), std::vector<int32_t>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                   std::initializer_list<int32_t> inputs_map,
                                   std::initializer_list<int32_t> outputs_map) {
  return ReplaceNodeDataAnchors(new_node, old_node,
                                std::vector<int32_t>(inputs_map), std::vector<int32_t>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                   const std::vector<int32_t> &inputs_map,
                                   const std::vector<int32_t> &outputs_map) {
  if (new_node == nullptr || old_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param new_node or old_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }

  auto ret = ReplaceOutDataAnchors(new_node->GetAllOutDataAnchors(), old_node->GetAllOutDataAnchors(), outputs_map);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Replace][OutDataAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s", old_node->GetName().c_str(), old_node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  ret = ReplaceInDataAnchors(new_node->GetAllInDataAnchors(), old_node->GetAllInDataAnchors(), inputs_map);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Replace][InDataAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s", old_node->GetName().c_str(), old_node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyInCtrlEdges(const NodePtr &src_node,
                                                                                       NodePtr &dst_node) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  const auto src_ctrl_in_nodes = src_node->GetInControlNodes();
  if (src_ctrl_in_nodes.empty()) {
    return GRAPH_SUCCESS;
  }

  std::unordered_set<NodePtr> exist_in_ctrl_nodes_set;
  auto exist_in_ctrl_nodes = dst_node->GetInControlNodes();
  if (!exist_in_ctrl_nodes.empty()) {
    exist_in_ctrl_nodes_set.insert(exist_in_ctrl_nodes.begin(), exist_in_ctrl_nodes.end());
  }

  const auto dst_ctrl = dst_node->GetInControlAnchor();
  for (const auto &in_node : src_ctrl_in_nodes) {
    if (exist_in_ctrl_nodes_set.count(in_node) > 0U) {
      continue;
    }
    const auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), dst_ctrl);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add ControlEdge from %s to %s failed, when copy control dependencies from %s to %s",
                        in_node->GetName().c_str(), dst_node->GetName().c_str(), src_node->GetName().c_str(),
                        dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed, when copy control dependencies from %s to %s",
             in_node->GetName().c_str(), dst_node->GetName().c_str(), src_node->GetName().c_str(),
             dst_node->GetName().c_str());
      return ret;
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveInCtrlEdges(const NodePtr &src_node,
                                                                                       NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_FAILED;
  }
  const auto ret = CopyInCtrlEdges(src_node, dst_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][InCtrlEdges] failed, ret:%d, src_node:%s, dst_node:%s",
           ret, src_node->GetName().c_str(), dst_node->GetName().c_str());
    return ret;
  }
  GE_CHECK_NOTNULL(src_node->GetInControlAnchor());
  src_node->GetInControlAnchor()->UnlinkAll();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyOutCtrlEdges(const NodePtr &src_node,
                                                                                        NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_FAILED;
  }
  const auto out_ctrl_nodes = src_node->GetOutControlNodes();
  if (out_ctrl_nodes.empty()) {
    return GRAPH_SUCCESS;
  }

  std::unordered_set<Node *> exists_out_ctrl_nodes_set;
  for (const auto &node : dst_node->GetOutControlNodes()) {
    (void)exists_out_ctrl_nodes_set.insert(node.get());
  }

  const auto dst_out_ctrl = dst_node->GetOutControlAnchor();
  for (const auto &node : out_ctrl_nodes) {
    if (exists_out_ctrl_nodes_set.count(node.get()) > 0U) {
      continue;
    }
    const auto ret = GraphUtils::AddEdge(dst_out_ctrl, node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add ControlEdge from %s to %s failed when copy control dependencies from %s to %s",
                        dst_node->GetName().c_str(), node->GetName().c_str(), src_node->GetName().c_str(),
                        dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed when copy control dependencies from %s to %s",
             dst_node->GetName().c_str(), node->GetName().c_str(), src_node->GetName().c_str(),
             dst_node->GetName().c_str());
      return ret;
    }
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveOutCtrlEdges(NodePtr &src_node,
                                                                                        NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_FAILED;
  }
  const auto ret = CopyOutCtrlEdges(src_node, dst_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][OutCtrlEdges] failed, ret:%d", ret);
    return ret;
  }
  GE_CHECK_NOTNULL(src_node->GetOutControlAnchor());
  src_node->GetOutControlAnchor()->UnlinkAll();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AppendInputNode(const ComputeGraphPtr &graph,
                                                                                       const NodePtr &node) {
  if (graph->AddInputNode(node) == nullptr) {
    REPORT_CALL_ERROR("E19999", "AddInputNode %s(%s) failed, graph:%s", node->GetName().c_str(),
                      node->GetType().c_str(), graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][InputNode] %s(%s) failed, graph:%s", node->GetName().c_str(),
           node->GetType().c_str(), graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  graph->SetInputSize(graph->GetInputSize() + 1U);
  if (graph->impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "Graph impl is nullptr.");
    return GRAPH_FAILED;
  }
  graph->impl_->inputs_order_.emplace_back(node->GetName());
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
ComputeGraphPtr GraphUtils::FindRootGraph(ComputeGraphPtr graph) {
  ComputeGraphPtr result = nullptr;
  while (graph != nullptr) {
    result = std::move(graph);
    graph = result->GetParentGraph();
  }
  return result;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyGraph(const Graph &src_graph, Graph &dst_graph) {
  std::string graph_name = dst_graph.GetName();
  if (graph_name.empty()) {
    graph_name = src_graph.GetName();
  }
  ComputeGraphPtr new_compute_graph = ComGraphMakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL(new_compute_graph);
  const ComputeGraphPtr src_compute_graph = GetComputeGraph(src_graph);
  GE_CHECK_NOTNULL(src_compute_graph);
  if (src_compute_graph->GetParentGraph() != nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][RootGraph] Only support copy root graph, current graph name:%s, "
                         "parent graph name:%s.", src_compute_graph->GetName().c_str(),
           src_compute_graph->GetParentGraph()->GetName().c_str());
    return GRAPH_FAILED;
  }
  const int32_t depth = 0;
  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;
  graphStatus ret = CopyComputeGraph(src_compute_graph, new_compute_graph,
                                     node_old_2_new, op_desc_old_2_new, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][Graph] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  Graph tmp_graph = CreateGraphFromComputeGraph(new_compute_graph);
  ret = CopyGraphImpl(src_graph, tmp_graph,
                      node_old_2_new, op_desc_old_2_new);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][GraphImpl] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  std::swap(dst_graph, tmp_graph);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyComputeGraph(const ComputeGraphPtr &src_compute_graph,
                                         ComputeGraphPtr &dst_compute_graph) {
  GE_CHECK_NOTNULL(src_compute_graph);
  if (src_compute_graph->GetParentGraph() != nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][RootGraph] Only support copy root graph, current graph name:%s, "
                         "parent graph name:%s.", src_compute_graph->GetName().c_str(),
           src_compute_graph->GetParentGraph()->GetName().c_str());
    return GRAPH_FAILED;
  }

  const int32_t depth = 0;
  std::map<ConstNodePtr, NodePtr> old_2_new_node;
  std::map<ConstOpDescPtr, OpDescPtr> old_2_new_op_desc;
  const graphStatus ret = CopyComputeGraph(src_compute_graph, dst_compute_graph,
                                           old_2_new_node, old_2_new_op_desc, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][ComputeGraphPtr] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyOpAndSubgraph(const ComputeGraphPtr &src_compute_graph,
                                          ComputeGraphPtr &dst_compute_graph,
                                          std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                          std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                          std::unordered_map<std::string, NodePtr> &all_new_nodes,
                                          int32_t depth) {
  GE_CHECK_NOTNULL(src_compute_graph);
  GE_CHECK_NOTNULL(dst_compute_graph);
  const auto dst_root_compute_graph = FindRootGraph(dst_compute_graph);
  GE_CHECK_NOTNULL(dst_root_compute_graph);
  const auto src_root_compute_graph = FindRootGraph(src_compute_graph);
  GE_CHECK_NOTNULL(src_root_compute_graph);
  for (const auto &n : src_compute_graph->GetDirectNode()) {
    const OpDescPtr op_desc = AttrUtils::CopyOpDesc(n->GetOpDesc());
    if (op_desc == nullptr || op_desc->impl_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "CopyOpDesc failed from node:%s", n->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Copy][OpDesc] from node:%s failed", n->GetName().c_str());
      return GRAPH_FAILED;
    }
    if (CopyTensorAttrs(op_desc, n) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Copy][TensorAttrs] from node:%s failed.", n->GetName().c_str());
      return GRAPH_FAILED;
    }

    if (n->GetType() == CONSTANT || n->GetType() == CONSTANTOP) {
      GeTensorPtr weight = nullptr;
      if (AttrUtils::MutableTensor(n->GetOpDesc(), ATTR_NAME_WEIGHTS, weight)) {
        const GeTensor copy_weight = weight->Clone();
        if (!AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, copy_weight)) {
          REPORT_CALL_ERROR("E19999", "copy ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
          GELOGE(INTERNAL_ERROR, "[Set][Tensor]copy ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
          return GRAPH_FAILED;
        }
        GELOGD("Clone ATTR_NAME_WEIGHTS for node:%s success.", op_desc->GetName().c_str());
      }
    }

    op_desc->SetName(n->GetName());
    const NodePtr node = dst_compute_graph->AddNode(op_desc, n->GetOpDesc()->GetId());
    if (node == nullptr) {
      REPORT_CALL_ERROR("E19999", "AddNode %s to graph:%s failed",
                        op_desc->GetName().c_str(), dst_compute_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][Node][%s] to graph:%s failed",
             op_desc->GetName().c_str(), dst_compute_graph->GetName().c_str());
      return GRAPH_FAILED;
    }
    all_new_nodes[node->GetName()] = node;
    node_old_2_new[n] = node;
    op_desc_old_2_new[n->GetOpDesc()] = op_desc;

    // copy subgraph from old graph to new graph
    const auto &subgraph_names = n->GetOpDesc()->GetSubgraphInstanceNames();
    const size_t subgraph_num = subgraph_names.size();
    for (size_t subgraph_idx = 0U; subgraph_idx < subgraph_num; ++subgraph_idx) {
      const auto src_subgraph = src_root_compute_graph->GetSubgraph(subgraph_names[subgraph_num - 1U - subgraph_idx]);
      GE_CHECK_NOTNULL(src_subgraph);
      ComputeGraphPtr dst_subgraph = ComGraphMakeShared<ComputeGraph>(src_subgraph->GetName());
      GE_CHECK_NOTNULL(dst_subgraph);
      dst_subgraph->SetParentGraph(dst_compute_graph);
      std::map<ConstNodePtr, NodePtr> sub_node_old_2_new;
      std::map<ConstOpDescPtr, OpDescPtr> sub_op_desc_old_2_new;
      const graphStatus ret = CopyComputeGraph(src_subgraph, dst_subgraph, sub_node_old_2_new,
                                               sub_op_desc_old_2_new, depth + 1);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "[Copy][SubGraph] %s of parent node:%s failed.",
               src_subgraph->GetName().c_str(), node->GetName().c_str());
        return GRAPH_FAILED;
      }
      (void)dst_root_compute_graph->AddSubGraph(dst_subgraph);
      dst_subgraph->SetParentNode(node);
      op_desc->impl_->subgraph_ir_names_to_type_ = n->GetOpDesc()->impl_->subgraph_ir_names_to_type_;
      op_desc->impl_->subgraph_names_to_index_ = n->GetOpDesc()->impl_->subgraph_names_to_index_;
      op_desc->impl_->subgraph_instance_names_ = n->GetOpDesc()->impl_->subgraph_instance_names_;
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyComputeGraph(const ComputeGraphPtr &src_compute_graph,
                                         ComputeGraphPtr &dst_compute_graph,
                                         std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                         std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                         int32_t depth) {
  GE_CHECK_NOTNULL(dst_compute_graph);
  GE_CHECK_NOTNULL(src_compute_graph);

  if (depth >= kCopyGraphMaxRecursionDepth) {
    REPORT_INNER_ERROR("E19999", "param depth:%d >= %d(allow max subgraphs)", depth, kCopyGraphMaxRecursionDepth);
    GELOGE(GRAPH_FAILED, "[Check][Param]exist too much subgraphs:%d > %d(allow max subgraphs)",
           depth, kCopyGraphMaxRecursionDepth);
    return GRAPH_FAILED;
  }
  // copy op and subgraph from old graph to new graph
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  graphStatus ret = CopyOpAndSubgraph(src_compute_graph, dst_compute_graph,
                                      node_old_2_new, op_desc_old_2_new,
                                      all_new_nodes, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][OpAndSubGraph] failed.");
    return GRAPH_FAILED;
  }

  for (const auto &n : src_compute_graph->GetDirectNode()) {
    if (RelinkGraphEdges(n, "", all_new_nodes) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Relink][Edges] failed.");
      return GRAPH_FAILED;
    }
  }
  // To keep subgraph consistent with the source graph
  std::vector<ComputeGraphPtr> new_subgraphs;
  const auto old_subgraphs = src_compute_graph->GetAllSubgraphs();
  for (const auto &sub_graph : old_subgraphs) {
    const auto new_subgraph = dst_compute_graph->GetSubgraph(sub_graph->GetName());
    GE_CHK_BOOL_EXEC(new_subgraph != nullptr, return GRAPH_FAILED,
                     "[Reorder][SubGraphs] can't find subgraph:%s in new graph.", sub_graph->GetName().c_str());
    GELOGD("Copy new subgraph:%s.", sub_graph->GetName().c_str());
    new_subgraphs.push_back(new_subgraph);
  }
  dst_compute_graph->SetAllSubgraphs(new_subgraphs);

  // copy members from old graph to new graph
  ret = CopyMembers(src_compute_graph, dst_compute_graph, all_new_nodes);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][Members] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyMembers(const ComputeGraphPtr &src_compute_graph,
                                    ComputeGraphPtr &dst_compute_graph,
                                    const std::unordered_map<std::string, NodePtr> &all_new_nodes) {
  if (src_compute_graph == nullptr || src_compute_graph->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param src_compute_graph is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Src compute graph is nullptr.");
    return GRAPH_FAILED;
  }
  if (dst_compute_graph == nullptr || dst_compute_graph->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param dst_compute_graph is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Dst compute graph is nullptr.");
    return GRAPH_FAILED;
  }
  // copy info of output nodes from old graph to new graph.
  const std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info = src_compute_graph->GetGraphOutNodesInfo();
  std::vector<std::pair<NodePtr, int32_t>> new_out_nodes_info;
  for (const auto &info : out_nodes_info) {
    const auto it = all_new_nodes.find(info.first->GetName());
    if (it == all_new_nodes.end()) {
      REPORT_INNER_ERROR("E19999", "Find output node:%s failed.", info.first->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Find output node:%s failed.", info.first->GetName().c_str());
      return GRAPH_FAILED;
    }
    new_out_nodes_info.emplace_back(it->second, info.second);
  }
  dst_compute_graph->SetGraphOutNodesInfo(new_out_nodes_info);

  // copy info of input nodes from old graph to new graph.
  const ComputeGraph::Vistor<NodePtr> &input_nodes = src_compute_graph->GetInputNodes();
  for (const auto &node : input_nodes) {
    const auto it = all_new_nodes.find(node->GetName());
    if (it == all_new_nodes.end()) {
      REPORT_INNER_ERROR("E19999", "Find input node:%s failed.", node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Find input node:%s failed.", node->GetName().c_str());
      return GRAPH_FAILED;
    }
    (void)dst_compute_graph->AddInputNode(it->second);
  }

  // copy target info nodes from old graph to new graph.
  const std::vector<NodePtr> &src_traget_nodes_info = src_compute_graph->GetGraphTargetNodesInfo();
  std::vector<NodePtr> dst_traget_nodes_info;
  for (const auto &node : src_traget_nodes_info) {
    const auto it = all_new_nodes.find(node->GetName());
    if (it == all_new_nodes.end()) {
      REPORT_INNER_ERROR("E19999", "Find target info node:%s failed.", node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Find target info node:%s failed.", node->GetName().c_str());
      return GRAPH_FAILED;
    }
    dst_traget_nodes_info.emplace_back(it->second);
  }
  dst_compute_graph->SetGraphTargetNodesInfo(dst_traget_nodes_info);

  // graph属性序列化
  dst_compute_graph->impl_->attrs_ = src_compute_graph->impl_->attrs_;

  // copy other members from old graph to new graph.
  dst_compute_graph->impl_->data_format_ = src_compute_graph->impl_->data_format_;
  dst_compute_graph->impl_->is_unknown_shape_graph_ = src_compute_graph->impl_->is_unknown_shape_graph_;
  dst_compute_graph->impl_->need_iteration_ = src_compute_graph->impl_->need_iteration_;
  dst_compute_graph->impl_->is_summary_graph_ = src_compute_graph->impl_->is_summary_graph_;
  dst_compute_graph->impl_->is_valid_flag_ = src_compute_graph->impl_->is_valid_flag_;
  dst_compute_graph->impl_->input_size_ = src_compute_graph->impl_->input_size_;
  dst_compute_graph->impl_->output_size_ = src_compute_graph->impl_->output_size_;
  dst_compute_graph->impl_->direct_nodes_size_ = src_compute_graph->impl_->direct_nodes_size_;
  dst_compute_graph->impl_->inputs_order_ = src_compute_graph->impl_->inputs_order_;
  dst_compute_graph->impl_->op_name_map_ = src_compute_graph->impl_->op_name_map_;
  dst_compute_graph->impl_->out_nodes_map_ = src_compute_graph->impl_->out_nodes_map_;
  dst_compute_graph->impl_->params_share_map_ = src_compute_graph->impl_->params_share_map_;

  return GRAPH_SUCCESS;
}

///
/// Make a copy of ComputeGraph.
/// @param graph: original graph.
/// @param prefix: node name prefix of new graph.
/// @param output_nodes: output nodes of new graph.
/// @return ComputeGraphPtr
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
ComputeGraphPtr GraphUtils::CloneGraph(const ComputeGraphPtr &graph, const std::string &prefix,
                                       std::vector<NodePtr> &input_nodes, std::vector<NodePtr> &output_nodes) {
  GE_CHK_BOOL_EXEC(graph != nullptr, REPORT_INNER_ERROR("E19999", "param graph is nullptr, check invalid.");
                   return nullptr, "[Check][Param] Original graph is null");
  const ComputeGraphPtr new_graph = ComGraphMakeShared<ComputeGraph>(graph->GetName());
  GE_CHK_BOOL_EXEC(new_graph != nullptr,
                   REPORT_CALL_ERROR("E19999", "create computegraph %s failed.", graph->GetName().c_str());
                   return nullptr, "[Create][ComputeGraph] %s failed", graph->GetName().c_str());

  std::unordered_map<std::string, NodePtr> all_new_nodes;
  for (const auto &n : graph->GetDirectNode()) {
    const OpDescPtr op_desc = AttrUtils::CopyOpDesc(n->GetOpDesc());
    GE_CHK_BOOL_EXEC(op_desc != nullptr,
                     REPORT_CALL_ERROR("E19999", "Create node:%s failed.", n->GetOpDesc()->GetName().c_str());
                     return nullptr, "[Create][Node] %s failed", n->GetOpDesc()->GetName().c_str());

    if (CopyTensorAttrs(op_desc, n) != GRAPH_SUCCESS) {
      return nullptr;
    }

    const bool is_const_op = (n->GetType() == CONSTANT) || (n->GetType() == CONSTANTOP);
    if (is_const_op) {
      GeTensorPtr weight = nullptr;
      if (!AttrUtils::MutableTensor(n->GetOpDesc(), ATTR_NAME_WEIGHTS, weight)) {
        GELOGI("Can not find attr ATTR_NAME_WEIGHTS for node:%s.", n->GetName().c_str());
        continue;
      }
      const GeTensor copy_weight = weight->Clone();
      if (!AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, copy_weight)) {
        REPORT_CALL_ERROR("E19999", "Clone ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Tensor] Clone ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
        return nullptr;
      }
      GELOGD("Clone ATTR_NAME_WEIGHTS for node:%s success.", op_desc->GetName().c_str());
    }

    op_desc->SetName(n->GetName() + prefix);
    NodePtr node = new_graph->AddNode(op_desc);
    GE_CHK_BOOL_EXEC(node != nullptr,
                     REPORT_CALL_ERROR("E19999", "add node %s to graph:%s failed",
                                       op_desc->GetName().c_str(), new_graph->GetName().c_str());
                     return nullptr, "[Add][Node] [%s] to graph:%s failed",
                     op_desc->GetName().c_str(), new_graph->GetName().c_str());
    all_new_nodes[node->GetName()] = node;

    if (node->GetType() == DATA) {
      input_nodes.emplace_back(node);
    } else if (node->GetType() == NETOUTPUT) {
      output_nodes.emplace_back(node);
    } else {
      // do nothing
    }
  }

  for (const auto &n : graph->GetDirectNode()) {
    if (RelinkGraphEdges(n, prefix, all_new_nodes) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }

  std::string session_graph_id;
  if (AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    const bool ret = AttrUtils::SetStr(*new_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
    if (!ret) {
      REPORT_CALL_ERROR("E19999", "set attr ATTR_NAME_SESSION_GRAPH_ID failed, ret:%d, graph:%s",
                        static_cast<int32_t>(ret), new_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Set][Attr] ATTR_NAME_SESSION_GRAPH_ID failed, ret:%d, graph:%s.",
             static_cast<int32_t>(ret), new_graph->GetName().c_str());
      return nullptr;
    }
  }

  // copy info of output nodes from old graph to new graph.
  const std::vector<std::pair<NodePtr, int32_t>> out_nodes_info = graph->GetGraphOutNodesInfo();
  std::vector<std::pair<NodePtr, int32_t>> new_out_nodes_info;
  for (const auto &info : out_nodes_info) {
    const auto it = all_new_nodes.find(info.first->GetName());
    if (it != all_new_nodes.end()) {
      new_out_nodes_info.emplace_back(it->second, info.second);
    }
  }
  new_graph->SetGraphOutNodesInfo(new_out_nodes_info);
  return new_graph;
}

///
/// Copy tensor attribute to new node.
/// @param [in] dst_node: cloned node.
/// @param [in] src_node: original node.
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::CopyTensorAttrs(const OpDescPtr &dst_desc, const NodePtr &src_node) {
  if (dst_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dst_desc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input param dst node not valid");
    return GRAPH_FAILED;
  }
  if (src_node == nullptr || src_node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src_node is nullptr or it's opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input param src node not valid");
    return GRAPH_FAILED;
  }

  const auto &src_desc = src_node->GetOpDesc();
  dst_desc->CopyAttrsFrom(*src_desc);

  for (uint32_t i = 0U; i < src_node->GetAllInDataAnchorsSize(); ++i) {
    const auto input_desc = dst_desc->MutableInputDesc(i);
    if (input_desc == nullptr) {
      continue;
    }
    input_desc->CopyAttrsFrom(src_desc->GetInputDesc(i));
  }

  for (uint32_t i = 0U; i < src_node->GetAllOutDataAnchorsSize(); ++i) {
    const auto output_desc = dst_desc->MutableOutputDesc(i);
    if (output_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "Param dst node:%s not valid, output_desc[%d] is nullptr",
                         dst_desc->GetName().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Param dst node:%s not valid", dst_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
    output_desc->CopyAttrsFrom(src_desc->GetOutputDesc(i));
  }

  return GRAPH_SUCCESS;
}

///
/// Relink all edges for cloned ComputeGraph.
/// @param [in] node: original node.
/// @param [in] prefix: node name prefix of new node.
/// @param [in] all_nodes: all nodes in new graph.
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::RelinkGraphEdges(const NodePtr &node, const std::string &prefix,
                                         const std::unordered_map<std::string, NodePtr> &all_nodes) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr or it's opdesc is nullptr. check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node not valid");
    return GRAPH_FAILED;
  }

  auto it = all_nodes.find(node->GetName() + prefix);
  if (it == all_nodes.end()) {
    REPORT_INNER_ERROR("E19999", "all_nodes not contain node:%s.", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] not found", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  const auto &new_node = it->second;

  // traversing from the parent node can be completely restored in the original one-to-many order.
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                     REPORT_INNER_ERROR("E19999", "out data anchor is null, node:%s.", node->GetName().c_str());
                     return GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s", node->GetName().c_str());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHK_BOOL_EXEC(peer_in_anchor->GetOwnerNode() != nullptr,
                       REPORT_INNER_ERROR("E19999", "Peer in node:%s is null", node->GetName().c_str());
                       return GRAPH_FAILED, "Peer in node:%s is null", node->GetName().c_str());
      it = all_nodes.find(peer_in_anchor->GetOwnerNode()->GetName() + prefix);
      if (it == all_nodes.end()) {
        REPORT_INNER_ERROR("E19999", "all_nodes not contain node[%s]",
                           peer_in_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] not found", peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return GRAPH_FAILED;
      }
      const auto &new_peer_in_node = it->second;
      const auto ret = GraphUtils::AddEdge(new_node->GetOutAnchor(out_anchor->GetIdx()),
                                           new_peer_in_node->GetInAnchor(peer_in_anchor->GetIdx()));
      GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "add data edge from %s to %s failed",
                                         new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
                       return GRAPH_FAILED, "[Invoke][AddEdge] link data edge failed[%s to %s]",
                       new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
    }
  }

  if (node->GetOutControlAnchor() != nullptr) {
    for (const auto &peer_in_control_anchor : node->GetOutControlAnchor()->GetPeerAnchors()) {
      GE_CHECK_NOTNULL(peer_in_control_anchor);
      GE_CHK_BOOL_EXEC(peer_in_control_anchor->GetOwnerNode() != nullptr,
                       REPORT_INNER_ERROR("E19999", "Peer out node is null");
                       return GRAPH_FAILED, "[Invoke][GetOwnerNode] Peer out node is null");
      it = all_nodes.find(peer_in_control_anchor->GetOwnerNode()->GetName() + prefix);
      if (it == all_nodes.end()) {
        REPORT_INNER_ERROR("E19999", "all_nodes not contain node:%s",
                           peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] not found",
               peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        return GRAPH_FAILED;
      }
      const auto &new_peer_in_node = it->second;
      const auto ret = GraphUtils::AddEdge(new_node->GetOutControlAnchor(),
                                           new_peer_in_node->GetInAnchor(peer_in_control_anchor->GetIdx()));
      GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "add control edge from %s to %s failed.",
                                         new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
                       return GRAPH_FAILED, "[Invoke][AddEdge] link control edge failed[%s to %s]",
                       new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
    }
  }
  return GRAPH_SUCCESS;
}

///
/// Get reference-mapping of all data_anchors in graph
/// @param [in] graph
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::GetRefMapping(const ComputeGraphPtr &graph,
                                      std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                      std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetAllNodes()) {
    // in_data_anchor
    if (HandleInAnchorMapping(graph, node, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Find ref_mapping for in_data_anchors of node %s failed.", node->GetName().c_str());
      GE_LOGE("[Invoke][HandleInAnchorMapping] Find ref_mapping for in_data_anchors of node %s failed.",
              node->GetName().c_str());
      return GRAPH_FAILED;
    }

    // out_data_anchor
    if (HandleOutAnchorMapping(node, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Find ref_mapping for out_data_anchors of node %s failed.", node->GetName().c_str());
      GE_LOGE("[Invoke][HandleInAnchorMapping] Find ref_mapping for out_data_anchors of node %s failed.",
              node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
NodePtr GraphUtils::FindNodeFromAllNodes(ComputeGraphPtr &graph, const std::string &name) {
  const auto root_graph = FindRootGraph(graph);
  if (root_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "param graph is nullptr,check invalid.");
    GE_LOGE("[Check][Param] param graph is nullptr, check invalid");
    return nullptr;
  }

  for (const auto &node : root_graph->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetName() == name) {
      return node;
    }
  }

  return nullptr;
}

///
/// Get reference-mapping for in_data_anchors of node
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleInAnchorMapping(const ComputeGraphPtr &graph, const NodePtr &node,
                                              std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                              std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  if (node->GetOwnerComputeGraph()->GetName() != graph->GetName()) {
    // when curr graph is subgraph , to handle subgraph input/output ref mapping
    if (NodeUtils::IsSubgraphOutput(node)) {
      return HandleSubgraphOutput(node, symbol_to_anchors, anchor_to_symbol);
    }

    if (NodeUtils::IsSubgraphInput(node)) {
      return HandleSubgraphInput(node, symbol_to_anchors, anchor_to_symbol);
    }
  }

  const std::string &type = node->GetType();
  if ((type == MERGE) || (type == STREAMMERGE)) {
    return HandleMergeInput(node, symbol_to_anchors, anchor_to_symbol);
  }

  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const NodeIndexIO cur_node_info(node, in_data_anchor->GetIdx(), kIn);
    const OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      const std::string &symbol = cur_node_info.ToString();
      GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
      symbol_to_anchors[symbol] = { cur_node_info };
      anchor_to_symbol[symbol] = symbol;
    } else {
      const NodeIndexIO exist_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
      if (UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
        GE_LOGE("[Update][SymbolMapping] failed.");
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

///
/// Get reference-mapping for out_data_anchors of node
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleOutAnchorMapping(const NodePtr &node,
                                               std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                               std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    const NodeIndexIO cur_node_info(node, out_data_anchor->GetIdx(), kOut);
    if (anchor_to_symbol.find(cur_node_info.ToString()) != anchor_to_symbol.end()) {
      continue;
    }

    int32_t reuse_in_index = -1;
    const bool reuse_input_flag = IsRefFromInput(out_data_anchor, reuse_in_index);
    if (reuse_input_flag && (node->GetInDataAnchor(reuse_in_index) != nullptr)) {
      const NodeIndexIO exist_node_info(node, reuse_in_index, kIn);
      if (UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
        GE_LOGE("[Update][SymbolMapping] failed.");
        return GRAPH_FAILED;
      }
    } else {
      if (reuse_input_flag) {
        GELOGW("[GetRefMapping][Check] Invalid reuse_input attr on output %d of node %s, please check attr reuse_input "
               "and reuse_input_index", out_data_anchor->GetIdx(), node->GetName().c_str());
      }
      const std::string &symbol = cur_node_info.ToString();
      GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
      (void)symbol_to_anchors.emplace(std::make_pair(symbol, std::list<NodeIndexIO>{ cur_node_info }));
      (void)anchor_to_symbol.emplace(std::make_pair(symbol, symbol));
    }
  }

  return GRAPH_SUCCESS;
}

///
/// Handle input of subgraph
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleSubgraphInput(const NodePtr &node,
                                            std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                            std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());

  // Data in subgraph
  uint32_t index = 0U;
  if (!ge::AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, index)) {
    REPORT_CALL_ERROR("E19999", "Get  Attr ATTR_NAME_PARENT_NODE_INDEX failed, node:%s.", node->GetName().c_str());
    GE_LOGE("[Get][Attr] ATTR_NAME_PARENT_NODE_INDEX failed, node:%s.", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  const NodePtr parent_node = node->GetOwnerComputeGraph()->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  const InDataAnchorPtr parent_in_anchor = parent_node->GetInDataAnchor(static_cast<int32_t>(index));
  GE_CHECK_NOTNULL(parent_in_anchor);
  const OutDataAnchorPtr peer_out_anchor = parent_in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr) {
    // Data has and only has one input
    const NodeIndexIO cur_node_info(node, 0, kIn);
    const NodeIndexIO exist_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
    if (UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
      GE_LOGE("[Update][SymbolMapping] failed.");
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

///
/// Handle input of Merge op
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleMergeInput(const NodePtr &node,
                                         std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                         std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  std::vector<NodeIndexIO> exist_node_infos;
  std::vector<NodeIndexIO> cur_node_infos;
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      std::string next_name;
      if ((AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, next_name)) && (!next_name.empty())) {
        ComputeGraphPtr graph = node->GetOwnerComputeGraph();
        GE_CHECK_NOTNULL(graph);
        const ge::NodePtr next_node = FindNodeFromAllNodes(graph, next_name);
        GE_CHECK_NOTNULL(next_node);
        // NextIteration has and only has one output
        peer_out_anchor = next_node->GetOutDataAnchor(0);
        GE_CHECK_NOTNULL(peer_out_anchor);
        cur_node_infos.emplace_back(NodeIndexIO(node, in_data_anchor->GetIdx(), kIn));
        cur_node_infos.emplace_back(NodeIndexIO(next_node, peer_out_anchor->GetIdx(), kOut));
      }
    } else {
      cur_node_infos.emplace_back(NodeIndexIO(node, in_data_anchor->GetIdx(), kIn));
      exist_node_infos.emplace_back(NodeIndexIO(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut));
    }
  }

  size_t anchor_nums = 0U;
  NodeIndexIO max_node_index_io(nullptr, 0, kOut);
  for (const auto &temp_node_info : exist_node_infos) {
    const auto iter1 = anchor_to_symbol.find(temp_node_info.ToString());
    if (iter1 != anchor_to_symbol.end()) {
      const std::string &temp_symbol = iter1->second;
      const auto iter2 = symbol_to_anchors.find(temp_symbol);
      if (iter2 != symbol_to_anchors.end()) {
        if (iter2->second.size() > anchor_nums) {
          max_node_index_io = temp_node_info;
          anchor_nums = iter2->second.size();
        }
      }
    }
  }

  std::string symbol;
  for (const auto &temp_node_info : exist_node_infos) {
    if ((UnionSymbolMapping(max_node_index_io, temp_node_info, symbol_to_anchors, anchor_to_symbol, symbol) !=
         GRAPH_SUCCESS) ||
        symbol.empty()) {
      GE_LOGE("[Union][SymbolMap] anchor1:%s & anchor2:%s failed.", max_node_index_io.ToString().c_str(),
              temp_node_info.ToString().c_str());
      return GRAPH_FAILED;
    }
  }

  const auto iter = symbol_to_anchors.find(symbol);
  if (iter != symbol_to_anchors.end()) {
    for (const auto &temp_node_info : cur_node_infos) {
      GELOGD("Add anchor %s, symbol %s.", temp_node_info.ToString().c_str(), symbol.c_str());
      iter->second.emplace_back(temp_node_info);
      (void)anchor_to_symbol.emplace(std::make_pair(temp_node_info.ToString(), symbol));
    }
  }

  return GRAPH_SUCCESS;
}

///
/// Handle output of subgraph
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleSubgraphOutput(const NodePtr &node,
                                             std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                             std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  const ComputeGraphPtr owner_graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(owner_graph);
  const NodePtr parent_node = owner_graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);

  const OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    const OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);

    const GeTensorDesc in_tensor = op_desc->GetInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx()));
    uint32_t index = 0U;
    if (!ge::AttrUtils::GetInt(in_tensor, ATTR_NAME_PARENT_NODE_INDEX, index)) {
      continue;
    }
    GE_CHECK_NOTNULL(parent_node->GetOutDataAnchor(static_cast<int32_t>(index)));
    // Union symbol of peer_out_anchor & parent_out_anchor
    const NodeIndexIO peer_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
    const NodeIndexIO parent_node_info(parent_node, index, kOut);
    std::string symbol;
    if ((UnionSymbolMapping(peer_node_info, parent_node_info, symbol_to_anchors, anchor_to_symbol,
                            symbol) != GRAPH_SUCCESS) || symbol.empty()) {
      GE_LOGE("[Union][SymbolMap] anchor1:%s, and anchor2:%s failed.",
              peer_node_info.ToString().c_str(), parent_node_info.ToString().c_str());
      return GRAPH_FAILED;
    }

    NodeIndexIO cur_node_info(node, in_data_anchor->GetIdx(), kIn);
    GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
    symbol_to_anchors[symbol].emplace_back(cur_node_info);
    (void)anchor_to_symbol.emplace(std::make_pair(cur_node_info.ToString(), symbol));
  }

  return GRAPH_SUCCESS;
}

///
/// Union ref-mapping
/// @param [in] exist_node_info1
/// @param [in] exist_node_info2
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @param [out] symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::UnionSymbolMapping(const NodeIndexIO &exist_node_info1, const NodeIndexIO &exist_node_info2,
                                           std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                           std::map<std::string, std::string> &anchor_to_symbol, std::string &symbol) {
  const std::string &symbol1 = anchor_to_symbol[exist_node_info1.ToString()];
  const std::string &symbol2 = anchor_to_symbol[exist_node_info2.ToString()];
  if (symbol1 == symbol2) {
    symbol = symbol1;
    GELOGI("no need to union.");
    return GRAPH_SUCCESS;
  }

  const auto iter1 = symbol_to_anchors.find(symbol1);
  const auto iter2 = symbol_to_anchors.find(symbol2);
  if ((iter1 == symbol_to_anchors.end()) || (iter2 == symbol_to_anchors.end())) {
    REPORT_INNER_ERROR("E19999", "symbol %s or %s not exist.", symbol1.c_str(), symbol2.c_str());
    GE_LOGE("[Check][Param] symbol %s or %s not exist.", symbol1.c_str(), symbol2.c_str());
    return GRAPH_FAILED;
  }

  auto &max_iter = ((iter1->second.size() > iter2->second.size()) ? iter1 : iter2);
  auto &min_iter = ((iter1->second.size() > iter2->second.size()) ? iter2 : iter1);
  symbol = ((iter1->second.size() > iter2->second.size()) ? symbol1 : symbol2);
  const std::string min_symbol = ((iter1->second.size() > iter2->second.size()) ? symbol2 : symbol1);
  for (auto &node_index_io : min_iter->second) {
    GELOGD("Update anchor %s, symbol %s.", node_index_io.ToString().c_str(), symbol.c_str());
    max_iter->second.emplace_back(node_index_io);
    const auto iter = anchor_to_symbol.find(node_index_io.ToString());
    if (iter == anchor_to_symbol.end()) {
      REPORT_INNER_ERROR("E19999", "anchor %s not exist in anchor_to_symbol.", node_index_io.ToString().c_str());
      GE_LOGE("[Check][Param] anchor %s not exist in anchor_to_symbol.", node_index_io.ToString().c_str());
      return GRAPH_FAILED;
    }
    if (iter->second != min_symbol) {
      GELOGW("[GetRefMapping][Check] not expected symbol of anchor %s, expect %s but %s exactly.", iter->first.c_str(),
             min_symbol.c_str(), iter->second.c_str());
    }
    iter->second = symbol;
  }

  GELOGI("Union symbol %s and %s succ.", symbol.c_str(), min_symbol.c_str());
  (void)symbol_to_anchors.erase(min_iter);
  return GRAPH_SUCCESS;
}

///
/// Update symbol mapping with a new reference pair
/// @param [in] cur_node_info
/// @param [in] exist_node_info
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::UpdateRefMapping(const NodeIndexIO &cur_node_info, const NodeIndexIO &exist_node_info,
                                         std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                         std::map<std::string, std::string> &anchor_to_symbol) {
  const auto iter1 = anchor_to_symbol.find(exist_node_info.ToString());
  if (iter1 == anchor_to_symbol.end()) {
    REPORT_INNER_ERROR("E19999", "data_anchor %s is not visible before data_anchor %s, maybe TopoSorting is missing.",
                       exist_node_info.ToString().c_str(), cur_node_info.ToString().c_str());
    GE_LOGE("[Check][Param] data_anchor %s is not visible before data_anchor %s, maybe TopoSorting is missing.",
            exist_node_info.ToString().c_str(), cur_node_info.ToString().c_str());
    return GRAPH_FAILED;
  }

  const std::string &symbol = iter1->second;
  const auto iter2 = symbol_to_anchors.find(symbol);
  if (iter2 == symbol_to_anchors.end()) {
    REPORT_INNER_ERROR("E19999", "symbol %s not exist in symbol_to_anchors.", symbol.c_str());
    GE_LOGE("[Check][Param] symbol %s not found.", symbol.c_str());
    return GRAPH_FAILED;
  }
  GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
  iter2->second.emplace_back(cur_node_info);
  (void)anchor_to_symbol.emplace(std::make_pair(cur_node_info.ToString(), symbol));

  return GRAPH_SUCCESS;
}

graphStatus GraphUtils::GetSubgraphsRecursively(const ComputeGraphPtr &graph, std::vector<ComputeGraphPtr> &subgraphs) {
  const auto root_graph = GraphUtils::FindRootGraph(graph);
  if (root_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Failed to find root graph");
    GELOGE(GRAPH_FAILED, "[Get][Graph] Failed to find root graph");
    return GRAPH_FAILED;
  }
  if (graph == root_graph) {
    subgraphs = graph->GetAllSubgraphs();
    return GRAPH_SUCCESS;
  }
  for (const auto &node : graph->GetAllNodes()) {
    // op_desc of node should not be null
    for (const auto &graph_name : node->GetOpDesc()->GetSubgraphInstanceNames()) {
      const auto &subgraph = root_graph->GetSubgraph(graph_name);
      if (subgraph == nullptr) {
        GELOGW("[Get][Subgraph] subgraph %s of node %s is null", graph_name.c_str(), node->GetName().c_str());
        continue;
      }
      subgraphs.emplace_back(subgraph);
    }
  }
  return GRAPH_SUCCESS;
}

///
/// Check if out_data_anchor is reference of input
/// @param [in] out_data_anchor
/// @param [out] reuse_in_index
/// @return bool
///
bool GraphUtils::IsRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index) {
  if (out_data_anchor == nullptr) {
    GELOGW("[Check][Param] out_data_anchor is null");
    return false;
  }
  const int32_t output_index = out_data_anchor->GetIdx();

  // pass-through op
  const NodePtr node = out_data_anchor->GetOwnerNode();
  const std::string &type = node->GetType();
  const std::set<std::string> pass_through_set = { NETOUTPUT, WHILE, _WHILE, STATELESSWHILE };
  if ((pass_through_set.count(type) > 0U) || (NodeUtils::IsSubgraphInput(node))) {
    reuse_in_index = output_index;
    GELOGI("Pass-Through node name[%s] index[%u].", node->GetName().c_str(), reuse_in_index);
    return true;
  }

  // Merge op 0th output
  const bool is_merge_op = (type == MERGE) && (output_index == 0);
  if (is_merge_op) {
    reuse_in_index = 0;
    GELOGI("Merge name[%s] output_index[0].", node->GetName().c_str());
    return true;
  }

  // ref op
  // op_desc of node should not be null
  const OpDescPtr op_desc = node->GetOpDesc();
  bool is_ref = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (is_ref) {
    const std::string &output_name = op_desc->GetOutputNameByIndex(static_cast<uint32_t>(output_index));
    for (const auto &input_name : op_desc->GetAllInputNames()) {
      if ((!input_name.empty()) && (output_name == input_name)) {
        reuse_in_index = op_desc->GetInputIndexByName(input_name);
        GELOGI("Reference name[%s] output[%s][%d] ref to input[%s][%d].", op_desc->GetName().c_str(),
               output_name.c_str(), output_index, input_name.c_str(), reuse_in_index);
        return true;
      }
    }
  }

  // reuse input
  const auto output_op_desc = op_desc->GetOutputDescPtr(static_cast<uint32_t>(output_index));
  if (output_op_desc != nullptr) {
    bool reuse_input = false;
    if ((TensorUtils::GetReuseInput(*output_op_desc, reuse_input) == GRAPH_SUCCESS) && reuse_input) {
      uint32_t reuse_input_index = 0U;
      if (TensorUtils::GetReuseInputIndex(*output_op_desc, reuse_input_index) == GRAPH_SUCCESS) {
        reuse_in_index = static_cast<int32_t>(reuse_input_index);
        GELOGI("ReuseInput name[%s] output[%d] reuse input[%d].", op_desc->GetName().c_str(),
               output_index, reuse_in_index);
        return true;
      }
    }
  }
  // nopadding reuse
  return IsNoPaddingRefFromInput(out_data_anchor, reuse_in_index);
}

bool GraphUtils::IsNoPaddingRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index) {
  const NodePtr node = out_data_anchor->GetOwnerNode();
  // nopadding means output[0] reuse input[0], but as history reason,
  // other output index also return true for mem assign in block_mem_assigner
  bool attr_reuse = false;
  bool is_input_continuous = false;
  bool is_out_continuous = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_input_continuous);
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_out_continuous);
  const bool get_reuse_flag = ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
  const bool is_no_padding_reuse_input = (is_input_continuous || is_out_continuous) && get_reuse_flag && attr_reuse;
  if (is_no_padding_reuse_input) {
    reuse_in_index = 0;
    GELOGI("Nopadding ReuseInput name[%s] output[%d] reuse input[%d].", node->GetName().c_str(),
           out_data_anchor->GetIdx(), reuse_in_index);
    return true;
  }
  return false;
}

bool GraphUtils::IsNodeInGraphRecursively(const ComputeGraphPtr &graph, const Node &node) {
  auto parent_graph = node.GetOwnerComputeGraph();
  while (parent_graph != nullptr) {
    if (parent_graph == graph) {
      return true;
    }
    parent_graph = parent_graph->GetParentGraph();
  }
  return false;
}

///
/// Determine if the graph is a UNKNOWN_SHAPE graph based on whether the graph and all subgraphs
/// of the graph have UNKNOWN_SHAPE operators or not.
/// Note: This function will only look 'down' from the graph, not 'up'. For example, the following
/// scenario (K for known shape, U for unknown shape), ROOT graph is UNKNOWN_SHAPE while SUB graph is KNOWN_SHAPE
/// ROOT graph:      A -----> B -----> C
///                  K    subgraph     U
///                           |
///                           V
/// SUB graph:          D --> E --> F
///                     K     K     K
/// @param [in] graph
/// @return bool
///
bool GraphUtils::IsUnknownShapeGraph(const ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    GELOGW("[Check][Param] Input graph is nullptr.");
    return false;
  }
  for (const auto &node : graph->GetDirectNode()) {
    bool is_unknown = false;
    const auto ret = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
    if (ret != GRAPH_SUCCESS) {
      GELOGW("[Check][UnknownGraph] Get unknown status failed, node name:%s, type:%s", node->GetName().c_str(),
             node->GetType().c_str());
      continue;
    }
    if (is_unknown) {
      GELOGD("Node %s, type %s is unknown shape in graph %s.",
             node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
      return true;
    }
  }
  GELOGD("Graph %s does not have unknown shape node.", graph->GetName().c_str());
  return false;
}

ComputeGraphPtr GraphUtils::BuildSubgraphWithNodes(const ComputeGraphPtr &graph, const std::set<NodePtr> &nodes,
                                                   const std::string &subgraph_name) {
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Graph is null");
    GELOGE(FAILED, "[Check][Param] graph is null");
    return nullptr;
  }
  return BuildSubgraphWithNodes(*graph, nodes, subgraph_name);
}

ComputeGraphPtr GraphUtils::BuildSubgraphWithNodes(ComputeGraph &graph, const std::set<NodePtr> &nodes,
                                                   const std::string &subgraph_name) {
  if (nodes.empty()) {
    GELOGW("nodes is empty, no need to build subgraph");
    return nullptr;
  }

  GraphInfo graph_info;
  BuildGraphInfoFromNodes(nodes, graph_info);

  const NodePtr graph_node = BuildSubgraphNode(graph, subgraph_name, graph_info);
  if (graph_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Build SubgraphNode failed, subgraph_name:%s.", subgraph_name.c_str());
    GELOGE(FAILED, "[Build][SubgraphNode] failed, subgraph_name:%s.", subgraph_name.c_str());
    return nullptr;
  }

  const ComputeGraphPtr subgraph = BuildSubgraph(graph_node, graph_info, subgraph_name);
  if (subgraph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Build Subgraph %s failed", subgraph_name.c_str());
    GELOGE(FAILED, "[Build][Subgraph] %s failed", subgraph_name.c_str());
    return nullptr;
  }
  const auto &root_graph = GraphUtils::FindRootGraph(graph_node->GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Find root graph failed, graph:%s", graph.GetName().c_str());
    GELOGE(FAILED, "[Find][RootGraph] failed, graph:%s", graph.GetName().c_str());
    return nullptr;
  }
  if (root_graph->AddSubgraph(subgraph) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add subgraph %s failed, root graph:%s", subgraph->GetName().c_str(),
                      root_graph->GetName().c_str());
    GELOGE(FAILED, "[Add][SubGraph] %s failed, root graph:%s", subgraph->GetName().c_str(),
           root_graph->GetName().c_str());
    return nullptr;
  }

  if ((RelinkDataEdges(graph_node, graph_info) != GRAPH_SUCCESS) ||
      (RelinkCtrlEdges(graph_node, graph_info) != GRAPH_SUCCESS)) {
    REPORT_CALL_ERROR("E19999", "ReLink edges for graph %s failed, graph_node:%s", graph.GetName().c_str(),
                      graph_node->GetName().c_str());
    GELOGE(FAILED, "[ReLink][Edges] for graph %s failed, graph_node:%s", graph.GetName().c_str(),
           graph_node->GetName().c_str());
    return nullptr;
  }

  for (const auto &node : nodes) {
    // op_desc of node should not be null
    const auto subgraph_names = node->GetOpDesc()->GetSubgraphInstanceNames();
    for (const auto &subgraph_name : subgraph_names) {
      node->GetOpDesc()->RemoveSubgraphInstanceName(subgraph_name);
    }
    if (RemoveNodeWithoutRelink(node->GetOwnerComputeGraph(), node) != GRAPH_SUCCESS) {
      GELOGW("Remove node %s failed.", node->GetName().c_str());
    }
  }

  return subgraph;
}

void GraphUtils::BuildGraphInfoFromNodes(const std::set<NodePtr> &nodes, GraphInfo &graph_info) {
  std::map<OutDataAnchorPtr, size_t> data_input_index_map;
  for (const auto &node : nodes) {
    // graph nodes
    (void)graph_info.nodes_.emplace(node);
    // in data
    BuildInDataEdgesFromNode(node, nodes, data_input_index_map, graph_info);
    // out data
    std::list<InDataAnchorPtr> peer_data_anchors;
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      peer_data_anchors.clear();
      const auto &peer_in_anchors = out_data_anchor->GetPeerInDataAnchors();
      (void)std::copy_if(peer_in_anchors.begin(), peer_in_anchors.end(), std::back_inserter(peer_data_anchors),
                         [nodes](const InDataAnchorPtr &peer_in_anchor) {
                           return nodes.count(peer_in_anchor->GetOwnerNode()) == 0UL;
                         });
      if (!peer_data_anchors.empty()) {
        const size_t output_index = graph_info.data_outputs_.size();
        graph_info.data_outputs_[output_index] = std::make_pair(out_data_anchor, peer_data_anchors);
      }
    }
    // in ctrl
    for (const auto &in_ctrl_node : node->GetInControlNodes()) {
      if (nodes.count(in_ctrl_node) == 0UL) {
        graph_info.ctrl_inputs_.emplace_back(in_ctrl_node->GetOutControlAnchor(), node->GetInControlAnchor());
      } else {
        graph_info.inner_ctrl_edges_.emplace_back(std::make_pair(in_ctrl_node->GetOutControlAnchor(),
                                                                 node->GetInControlAnchor()));
      }
    }
    // out ctrl
    for (const auto &out_ctrl_node : node->GetOutControlNodes()) {
      if (nodes.count(out_ctrl_node) == 0UL) {
        graph_info.ctrl_outputs_.emplace_back(node->GetOutControlAnchor(), out_ctrl_node->GetInControlAnchor());
      }
    }
  }
}

void GraphUtils::BuildInDataEdgesFromNode(const NodePtr &node, const std::set<NodePtr> &nodes,
                                          std::map<OutDataAnchorPtr, size_t> &data_input_index_map,
                                          GraphInfo &graph_info) {
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    if (nodes.count(peer_out_anchor->GetOwnerNode()) == 0UL) {
      size_t input_index;
      if (data_input_index_map.count(peer_out_anchor) == 0UL) {
        input_index = graph_info.data_inputs_.size();
        data_input_index_map[peer_out_anchor] = input_index;
        graph_info.data_inputs_[input_index].first = peer_out_anchor;
      } else {
        input_index = data_input_index_map[peer_out_anchor];
      }
      graph_info.data_inputs_[input_index].second.emplace_back(in_data_anchor);
    } else {
      graph_info.inner_data_edges_.emplace_back(std::make_pair(peer_out_anchor, in_data_anchor));
    }
  }
}

NodePtr GraphUtils::BuildSubgraphNode(ComputeGraph &graph, const std::string &graph_name,
                                      const GraphInfo &graph_info) {
  OpDescBuilder op_desc_builder(graph_name + "_" + PARTITIONEDCALL, PARTITIONEDCALL);
  int32_t i = 0;
  for (const auto &item : graph_info.data_inputs_) {
    for (const auto &in_data_anchor : item.second.second) {
      const auto input_desc = in_data_anchor->GetOwnerNode()->GetOpDesc();
      if (input_desc == nullptr) {
        REPORT_INNER_ERROR("E19999", "op_desc is null, node:%s", in_data_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(PARAM_INVALID, "[Check][Param] op_desc is null, node:%s",
               in_data_anchor->GetOwnerNode()->GetName().c_str());
        return nullptr;
      }
      (void)op_desc_builder.AddInput("args" + std::to_string(i),
                                     input_desc->GetInputDesc(static_cast<uint32_t>(in_data_anchor->GetIdx())));
      i++;
    }
  }
  for (const auto &item : graph_info.data_outputs_) {
    const auto output_desc = item.second.first->GetOwnerNode()->GetOpDesc();
    if (output_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "op_desc is null, node:%s",
                         item.second.first->GetOwnerNode()->GetName().c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] op_desc is null, node:%s",
             item.second.first->GetOwnerNode()->GetName().c_str());
      return nullptr;
    }
    (void)op_desc_builder.AddOutput("output" + std::to_string(item.first),
                                    output_desc->GetOutputDesc(static_cast<uint32_t>(item.second.first->GetIdx())));
  }

  const OpDescPtr op_desc = op_desc_builder.Build();
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Create op_desc for subgraph node failed, name:%s.", graph_name.c_str());
    GELOGE(FAILED, "[Create][OpDesc] for subgraph node failed, name:%s.", graph_name.c_str());
    return nullptr;
  }

  (void)op_desc->AddSubgraphName("f");
  (void)op_desc->SetSubgraphInstanceName(0U, graph_name);

  return graph.AddNode(op_desc);
}

ComputeGraphPtr GraphUtils::BuildSubgraph(const NodePtr &subgraph_node, const GraphInfo &graph_info,
                                          const std::string &subgraph_name) {
  CompleteGraphBuilder graph_builder(subgraph_name, false);
  // Add parent node
  graph_builder.SetParentNode(subgraph_node);

  // Add node
  for (const auto &node : graph_info.nodes_) {
    graph_builder.AddNode(AttrUtils::CopyOpDesc(node->GetOpDesc()));
  }

  // Set Input
  uint32_t index = 0;
  for (const auto &item : graph_info.data_inputs_) {
    for (const auto &in_data_anchor : item.second.second) {
      graph_builder.SetInput(index, { in_data_anchor->GetOwnerNode()->GetName() },
                             { static_cast<uint32_t>(in_data_anchor->GetIdx()) });
      index++;
    }
  }

  // Add Outputs
  for (const auto &item : graph_info.data_outputs_) {
    graph_builder.AddOutput(item.second.first->GetOwnerNode()->GetName(),
                            item.second.first->GetIdx());
  }

  // Add Data Edges
  for (const auto &data_edge : graph_info.inner_data_edges_) {
    graph_builder.AddDataLink(data_edge.first->GetOwnerNode()->GetName(), data_edge.first->GetIdx(),
                              data_edge.second->GetOwnerNode()->GetName(), data_edge.second->GetIdx());
  }

  // Add Ctrl Edges
  for (const auto &ctrl_edge : graph_info.inner_ctrl_edges_) {
    graph_builder.AddControlLink(ctrl_edge.first->GetOwnerNode()->GetName(),
                                 ctrl_edge.second->GetOwnerNode()->GetName());
  }

  // Add Input-Mapping
  std::map<uint32_t, uint32_t> input_mapping;
  size_t j = 0U;
  for (const auto &item : graph_info.data_inputs_) {
    while (j < item.second.second.size()) {
      input_mapping[j] = j;
      j++;
    }
  }
  graph_builder.SetInputMapping(input_mapping);

  // Add outputMapping
  std::map<uint32_t, uint32_t> output_mapping;
  for (size_t i = 0U; i < graph_info.data_inputs_.size(); i++) {
    output_mapping[i] = i;
  }
  graph_builder.SetOutputMapping(output_mapping);

  graphStatus error_code = GRAPH_SUCCESS;
  std::string error_msg;
  ComputeGraphPtr subgraph = graph_builder.Build(error_code, error_msg);
  if (subgraph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Build subgraph %s failed:%s.", subgraph_node->GetName().c_str(), error_msg.c_str());
    GELOGE(error_code, "[Build][Subgraph] %s failed:%s.", subgraph_node->GetName().c_str(), error_msg.c_str());
    return nullptr;
  }

  return subgraph;
}

graphStatus GraphUtils::RelinkDataEdges(const NodePtr &subgraph_node, const GraphInfo &graph_info) {
  // in data nodes
  int32_t i = 0;
  for (const auto &item : graph_info.data_inputs_) {
    for (const auto &in_data_anchor : item.second.second) {
      GE_CHK_STATUS_RET(item.second.first->Unlink(in_data_anchor), "[Remove][DataEdge] %s:%d->%s:%d failed",
                        item.second.first->GetOwnerNode()->GetName().c_str(), item.second.first->GetIdx(),
                        in_data_anchor->GetOwnerNode()->GetName().c_str(), in_data_anchor->GetIdx());
      GE_CHK_STATUS_RET(item.second.first->LinkTo(subgraph_node->GetInDataAnchor(i)),
                        "[Add][DataEdge] %s:%d->%s:%u failed.",
                        item.second.first->GetOwnerNode()->GetName().c_str(),
                        item.second.first->GetIdx(), subgraph_node->GetName().c_str(), item.first);
      i++;
    }
  }
  // out data nodes
  for (const auto &item : graph_info.data_outputs_) {
    const auto &out_data_anchor = subgraph_node->GetOutDataAnchor(static_cast<int32_t>(item.first));
    GE_CHECK_NOTNULL(out_data_anchor);
    for (const auto &peer_in_anchor : item.second.second) {
      GE_CHK_STATUS_RET(item.second.first->Unlink(peer_in_anchor), "[Remove][DataEdge] %s:%d->%s:%d failed.",
                        item.second.first->GetOwnerNode()->GetName().c_str(), item.second.first->GetIdx(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(), peer_in_anchor->GetIdx());
      GE_CHK_STATUS_RET(out_data_anchor->LinkTo(peer_in_anchor), "[Add][DataEdge] %s:%u->%s:%d failed.",
                        subgraph_node->GetName().c_str(), item.first, peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetIdx());
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus GraphUtils::RelinkCtrlEdges(const NodePtr &subgraph_node, const GraphInfo &graph_info) {
  // in ctrl nodes
  for (const auto &ctrl_input : graph_info.ctrl_inputs_) {
    GE_CHK_STATUS_RET(ctrl_input.first->Unlink(ctrl_input.second), "[Remove][CtrlEdge] %s->%s failed",
                      ctrl_input.first->GetOwnerNode()->GetName().c_str(),
                      ctrl_input.second->GetOwnerNode()->GetName().c_str());
    if (!ctrl_input.first->IsLinkedWith(subgraph_node->GetInControlAnchor())) {
      GE_CHK_STATUS_RET(ctrl_input.first->LinkTo(subgraph_node->GetInControlAnchor()), "[Add][CtrlEdge] %s->%s failed.",
                        ctrl_input.first->GetOwnerNode()->GetName().c_str(), subgraph_node->GetName().c_str());
    }
  }
  // out ctrl nodes
  for (const auto &ctrl_output : graph_info.ctrl_outputs_) {
    GE_CHK_STATUS_RET(ctrl_output.first->Unlink(ctrl_output.second), "[Remove][CtrlEdge] %s->%s failed.",
                      ctrl_output.first->GetOwnerNode()->GetName().c_str(),
                      ctrl_output.second->GetOwnerNode()->GetName().c_str());
    if (!subgraph_node->GetOutControlAnchor()->IsLinkedWith(ctrl_output.second)) {
      GE_CHK_STATUS_RET(subgraph_node->GetOutControlAnchor()->LinkTo(ctrl_output.second),
                        "[Add][CtrlEdge] %s->%s failed.", subgraph_node->GetName().c_str(),
                        ctrl_output.second->GetOwnerNode()->GetName().c_str());
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus GraphUtils::UnfoldSubgraph(const ComputeGraphPtr &graph,
                                       const std::function<bool(const ComputeGraphPtr &)> &filter) {
  GE_CHECK_NOTNULL(graph);
  const auto &parent_graph = graph->GetParentGraph();
  const auto &parent_node = graph->GetParentNode();
  if ((parent_graph == nullptr) && (parent_node == nullptr)) {
    return GRAPH_SUCCESS;
  }

  GE_CHK_STATUS_RET(MergeInputNodes(graph),
                    "[Invoke][MergeInputNodes] Merge data nodes for graph %s failed",
                    graph->GetName().c_str());
  GE_CHK_STATUS_RET(MergeNetOutputNode(graph),
                    "[Invoke][MergeNetOutputNode] Merge net output nodes for graph %s failed",
                    graph->GetName().c_str());
  GELOGD("[%s] Merging graph inputs and outputs successfully", graph->GetName().c_str());

  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == DATA || node->GetType() == NETOUTPUT) {
      continue;
    }

    std::vector<ComputeGraphPtr> subgraphs;
    GE_CHK_STATUS_RET(NodeUtils::GetDirectSubgraphs(node, subgraphs), "[Get][Subgraphs] failed, graph:%s",
                      node->GetName().c_str());
    bool skip_add_node_flag = true;
    for (const auto &subgraph : subgraphs) {
      if ((filter != nullptr) && filter(subgraph)) {
        GE_CHK_STATUS_RET(UnfoldSubgraph(subgraph, filter),
                          "[Invoke][UnfoldSubgraph] Failed to merge graph %s", subgraph->GetName().c_str());
        skip_add_node_flag = false;
      } else {
        subgraph->SetParentGraph(parent_graph);
      }
    }

    if (skip_add_node_flag) {
      (void)parent_graph->AddNode(node);
      GELOGD("[%s::%s] added to parent graph: [%s].", graph->GetName().c_str(), node->GetName().c_str(),
             parent_graph->GetName().c_str());
      (void)node->SetOwnerComputeGraph(parent_graph);
    }
  }

  GELOGD("[%s] Done merging graph. remove it from root graph", graph->GetName().c_str());

  const auto &subgraph_name = graph->GetName();
  const auto &root_graph = GraphUtils::FindRootGraph(parent_graph);
  GE_CHECK_NOTNULL(root_graph);
  root_graph->RemoveSubgraph(graph->GetName());
  parent_node->GetOpDesc()->RemoveSubgraphInstanceName(subgraph_name);
  if (RemoveNodeWithoutRelink(parent_graph, parent_node) != GRAPH_SUCCESS) {
    GELOGW("Remove node %s failed, graph:%s.", parent_node->GetName().c_str(), parent_graph->GetName().c_str());
  }

  return GRAPH_SUCCESS;
}

graphStatus GraphUtils::MergeInputNodes(const ComputeGraphPtr &graph) {
  const auto &parent_node = graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);

  std::set<NodePtr> src_nodes;
  for (const auto &node : graph->GetDirectNode()) {
    if (node->GetType() != DATA) {
      if (node->GetInDataNodes().empty()) {
        (void)src_nodes.emplace(node);
      }
      continue;
    }

    uint32_t parent_index = 0U;
    if (!AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      REPORT_CALL_ERROR("E19999", "Get attr %s failed, node:%s", ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                        node->GetName().c_str());
      GELOGE(FAILED, "[Get][Attr] %s failed, node:%s", ATTR_NAME_PARENT_NODE_INDEX.c_str(), node->GetName().c_str());
      return GRAPH_FAILED;
    }

    const auto parent_node_in_anchor = parent_node->GetInDataAnchor(static_cast<int32_t>(parent_index));
    GE_CHECK_NOTNULL(parent_node_in_anchor);
    const auto src_out_anchor = parent_node_in_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr || src_out_anchor->GetOwnerNode() == nullptr) {
      continue;
    }
    parent_node_in_anchor->UnlinkAll();

    // link src to outputs of DataNode
    for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      for (const auto &peer_in_anchor : out_data_anchor->GetPeerInDataAnchors()) {
        auto dst_node = peer_in_anchor->GetOwnerNode();
        GE_CHECK_NOTNULL(dst_node);
        const auto &in_nodes = dst_node->GetInDataNodes();
        if (std::all_of(in_nodes.begin(), in_nodes.end(), [](const NodePtr &n) { return n->GetType() == DATA; })) {
          (void)src_nodes.emplace(dst_node);
        }
        GE_CHK_STATUS_RET(ReplaceEdgeSrc(out_data_anchor, peer_in_anchor, src_out_anchor),
                          "[Replace][DataEdge] failed");
      }
    }
  }

  // transfer in control edges to all root nodes
  for (const auto &src_node : src_nodes) {
    const auto &in_nodes = src_node->GetInAllNodes();
    const std::set<NodePtr> in_node_set(in_nodes.begin(), in_nodes.end());
    for (const auto &in_control_node : parent_node->GetInControlNodes()) {
      if ((in_node_set.count(in_control_node) == 0UL) && (kMergeInputSkipTypes.count(src_node->GetType()) == 0UL)) {
        GELOGD("[%s] Restore control edge to [%s]", in_control_node->GetName().c_str(), src_node->GetName().c_str());
        (void)AddEdge(in_control_node->GetOutControlAnchor(), src_node->GetInControlAnchor());
      }
    }
  }

  parent_node->GetInControlAnchor()->UnlinkAll();
  return GRAPH_SUCCESS;
}

graphStatus GraphUtils::MergeNetOutputNode(const ComputeGraphPtr &graph) {
  const auto &parent_node = graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);

  const NodePtr &net_output = graph->FindFirstNodeMatchType(NETOUTPUT);
  if (net_output == nullptr) {
    GELOGD("Graph has no NetOutput node, no need to merge");
    return SUCCESS;
  }
  auto all_in_nodes = net_output->GetInAllNodes();
  auto all_out_nodes = parent_node->GetOutAllNodes();
  net_output->GetInControlAnchor()->UnlinkAll();
  parent_node->GetOutControlAnchor()->UnlinkAll();

  for (const auto &in_data_anchor : net_output->GetAllInDataAnchors()) {
    const auto index = in_data_anchor->GetIdx();
    uint32_t parent_index = 0U;
    // op_desc of node should not be null
    if (!AttrUtils::GetInt(net_output->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(index)),
                           ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
      GELOGW("SubGraph: %s NetOutput input tensor %d, attr %s not found.", graph->GetName().c_str(), index,
             ATTR_NAME_PARENT_NODE_INDEX.c_str());
      continue;
    }

    const auto src_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(src_out_anchor);
    GE_CHECK_NOTNULL(src_out_anchor->GetOwnerNode());
    GE_CHK_STATUS_RET(RemoveEdge(src_out_anchor, in_data_anchor), "[Remove][DataEdge] %s:%d->%s:%d failed",
                      src_out_anchor->GetOwnerNode()->GetName().c_str(), src_out_anchor->GetIdx(),
                      net_output->GetName().c_str(), in_data_anchor->GetIdx());

    const OutDataAnchorPtr &parent_out_anchor = parent_node->GetOutDataAnchor(static_cast<int32_t>(parent_index));
    GE_CHECK_NOTNULL(parent_out_anchor);
    for (InDataAnchorPtr &dst_in_anchor : parent_out_anchor->GetPeerInDataAnchors()) {
      GE_CHK_STATUS_RET(ReplaceEdgeSrc(parent_out_anchor, dst_in_anchor, src_out_anchor),
                        "[Replace][DataEdge] failed");
    }
  }

  // transfer out control edges
  const std::set<NodePtr> in_node_set(all_in_nodes.begin(), all_in_nodes.end());
  const std::set<NodePtr> out_node_set(all_out_nodes.begin(), all_out_nodes.end());
  for (auto &src_node : in_node_set) {
    GELOGD("[%s] process in node.", src_node->GetName().c_str());
    auto out_nodes = src_node->GetOutAllNodes();
    const std::set<NodePtr> node_set(out_nodes.begin(), out_nodes.end());
    for (auto &dst_node : out_node_set) {
      if (node_set.count(dst_node) == 0UL) {
        GELOGD("[%s] Restore control edge to [%s]", src_node->GetName().c_str(), dst_node->GetName().c_str());
        (void)src_node->GetOutControlAnchor()->LinkTo(dst_node->GetInControlAnchor());
      }
    }
  }

  return GRAPH_SUCCESS;
}

///
/// @brief Add node to graph
/// @param [in] op_desc
/// @return ComputeGraphBuilder
///
ComputeGraphBuilder& ComputeGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  nodes_.emplace_back(op_desc);
  return *this;
}

///
/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return ComputeGraphBuilder
///
ComputeGraphBuilder& ComputeGraphBuilder::AddDataLink(const std::string &src_name, uint32_t out_anchor_ind,
                                                      const std::string &dst_name, uint32_t in_anchor_ind) {
  data_links_.emplace_back(std::make_pair(std::make_pair(src_name, out_anchor_ind),
                                          std::make_pair(dst_name, in_anchor_ind)));
  return *this;
}

///
/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return ComputeGraphBuilder
///
ComputeGraphBuilder& ComputeGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  ctrl_links_.emplace_back(std::make_pair(src_name, dst_name));
  return *this;
}

///
/// @brief Build nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void ComputeGraphBuilder::BuildNodes(graphStatus &error_code, std::string &error_msg) {
  if (owner_graph_ == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return;
  }

  std::string node_name;
  for (auto &op_desc : nodes_) {
    if (op_desc == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "op_desc is NULL.";
      return;
    }

    node_name = op_desc->GetName();
    const NodePtr node = owner_graph_->AddNode(op_desc);
    if (node == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "Add node " + node_name + " failed.";
      return;
    }

    GELOGD("Add node name:%s, type:%s.", node_name.c_str(), op_desc->GetType().c_str());
    node_names_[node_name] = node;
  }

  GELOGD("BuildNodes succ.");
}

///
/// @brief Build data-links
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void ComputeGraphBuilder::BuildDataLinks(graphStatus &error_code, std::string &error_msg) {
  for (auto &pair : data_links_) {
    const std::string src_name = pair.first.first;
    const auto out_ind = static_cast<int32_t>(pair.first.second);
    const std::string dst_name = pair.second.first;
    const auto in_ind = static_cast<int32_t>(pair.second.second);
    std::string log_msg = "Add data-edge ";
    (void)log_msg.append(src_name).append(":").append(std::to_string(out_ind)).append("->")
                 .append(dst_name).append(":").append(std::to_string(in_ind));

    const auto src_iter = node_names_.find(src_name);
    const auto dst_iter = node_names_.find(dst_name);
    if ((src_iter == node_names_.end()) || (dst_iter == node_names_.end())) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed: node not exist in graph.";
      return;
    }

    const NodePtr src_node = node_names_[src_name];
    const NodePtr dst_node = node_names_[dst_name];
    if ((src_node == nullptr) || (dst_node == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed: node is NULL.";
      return;
    }

    if (GraphUtils::AddEdge(src_node->GetOutDataAnchor(out_ind), dst_node->GetInDataAnchor(in_ind)) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed.";
      return;
    }

    GELOGD("%s succ.", log_msg.c_str());
  }

  GELOGD("BuildDataLinks succ.");
}

///
/// @brief Build ctrl-links
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void ComputeGraphBuilder::BuildCtrlLinks(graphStatus &error_code, std::string &error_msg) {
  for (auto &pair : ctrl_links_) {
    const std::string src_name = pair.first;
    const std::string dst_name = pair.second;
    std::string log_msg = "Add ctrl-edge ";
    (void)log_msg.append(src_name).append("->").append(dst_name);

    const auto src_iter = node_names_.find(src_name);
    const auto dst_iter = node_names_.find(dst_name);
    if ((src_iter == node_names_.end()) || (dst_iter == node_names_.end())) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed: node not exist in graph.";
      return;
    }

    const NodePtr src_node = node_names_[src_name];
    const NodePtr dst_node = node_names_[dst_name];
    if ((src_node == nullptr) || (dst_node == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed: node is NULL.";
      return;
    }

    if (GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed.";
      return;
    }

    GELOGD("%s succ.", log_msg.c_str());
  }

  GELOGD("BuildCtrlLinks succ.");
}

/// @brief Get node with name
/// @param [in] name
/// @return NodePtr
///
NodePtr ComputeGraphBuilder::GetNode(const std::string &name) {
  const auto iter = node_names_.find(name);
  if (iter == node_names_.end()) {
    REPORT_INNER_ERROR("E19999", "node %s not exist.", name.c_str());
    GE_LOGE("[Check][Param] node %s not exist.", name.c_str());
    return nullptr;
  }
  return iter->second;
}

/// @brief Get all nodes
/// @return std::vector<NodePtr>
///
std::vector<NodePtr> ComputeGraphBuilder::GetAllNodes() {
  std::vector<NodePtr> nodes;
  for (const auto &iter : node_names_) {
    nodes.emplace_back(iter.second);
  }
  return nodes;
}

///
/// @brief Add node to graph
/// @param [in] op_desc
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  (void)ComputeGraphBuilder::AddNode(op_desc);
  return *this;
}

///
/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddDataLink(const std::string &src_name, const uint32_t out_anchor_ind,
                                                        const std::string &dst_name, const uint32_t in_anchor_ind) {
  (void)ComputeGraphBuilder::AddDataLink(src_name, out_anchor_ind, dst_name, in_anchor_ind);
  return *this;
}

///
/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  (void)ComputeGraphBuilder::AddControlLink(src_name, dst_name);
  return *this;
}

///
/// @brief Set index_th input anchor for graph
/// @param [in] index
/// @param [in] node_names
/// @param [in] anchor_inds
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetInput(uint32_t index, const std::vector<std::string> &node_names,
                                                     const std::vector<uint32_t> &anchor_inds) {
  graph_inputs_[index] = std::make_pair(node_names, anchor_inds);
  return *this;
}

///
/// @brief Set index_th input of graph as useless
/// @param [in] index
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetUselessInput(uint32_t index) {
  graph_inputs_[index] = std::make_pair(std::vector<std::string>(), std::vector<uint32_t>());
  return *this;
}

///
/// @brief Add output anchor for graph
/// @param [in] owner_node_name
/// @param [in] anchor_ind
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddOutput(const std::string &owner_node_name, uint32_t anchor_ind) {
  graph_outputs_.emplace_back(std::make_pair(owner_node_name, anchor_ind));
  return *this;
}

///
/// @brief Add target for graph
/// @param [in] target_name
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddTarget(const std::string &target_name) {
  graph_targets_.emplace_back(target_name);
  return *this;
}

///
/// @brief Set parent-node of graph
/// @param [in] parent_node
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetParentNode(const NodePtr &parent_node) {
  parent_node_ = parent_node;
  return *this;
}

///
/// @brief Set mapping-relation of parent-node in_anchor_ind & Data-node
/// @param [in] input_mapping: index_of_graph_input -> in_anchor_index_of_parent_node
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetInputMapping(const std::map<uint32_t, uint32_t> &input_mapping) {
  for (auto &item : input_mapping) {
    input_mapping_[item.first] = item.second;
  }
  return *this;
}

///
/// @brief Set mapping-relation of parent-node out_anchor_ind & NetOutput-node out_anchor_ind
/// @param [in] output_mapping: index_of_graph_output -> out_anchor_index_of_parent_node
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) {
  for (auto &item : output_mapping) {
    output_mapping_[item.first] = item.second;
  }
  return *this;
}

///
/// @brief Build graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return ComputeGraphPtr
///
ComputeGraphPtr CompleteGraphBuilder::Build(graphStatus &error_code, std::string &error_msg) {
  owner_graph_ = ComGraphMakeShared<ComputeGraph>(name_);
  if (owner_graph_ == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return nullptr;
  }

  BuildNodes(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildDataLinks(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildCtrlLinks(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  AddDataNodes(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  if (retval_flag_) {
    AddRetValNodes(error_code, error_msg);
    if (error_code != GRAPH_SUCCESS) {
      return nullptr;
    }
    BuildGraphTargets(error_code, error_msg);
    if (error_code != GRAPH_SUCCESS) {
      return nullptr;
    }
  } else {
    AddNetOutputNode(error_code, error_msg);
    if (error_code != GRAPH_SUCCESS) {
      return nullptr;
    }
  }

  PostProcess(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  return owner_graph_;
}

///
/// @brief Add data nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::AddDataNodes(graphStatus &error_code, std::string &error_msg) {
  for (auto &input : graph_inputs_) {
    const NodePtr data_node = AddDataNode(input.first, error_code, error_msg);
    if (data_node == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNodes failed: add node Data:" + std::to_string(input.first) +  + " failed.";
      return;
    }

    if (owner_graph_->AddInputNode(data_node) == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNodes failed: add input node Data:" + std::to_string(input.first) +  + " failed.";
      return;
    }

    // useless input
    const std::vector<std::string> input_names = input.second.first;
    const std::vector<uint32_t> anchor_indes = input.second.second;
    if (input_names.size() != anchor_indes.size()) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNodes failed: num of input_names and indexs not equal.";
      return;
    }
    if (input_names.empty()) {
      continue;
    }

    const size_t input_num = input_names.size();
    for (size_t i = 0U; i < input_num; i++) {
      const std::string input_name = input_names[i];
      const int32_t ind = static_cast<int32_t>(anchor_indes[i]);
      const auto iter = node_names_.find(input_name);
      if (iter == node_names_.end()) {
        error_code = GRAPH_FAILED;
        error_msg = "AddDataNodes failed: node " + input_name + " not exist in graph.";
        return;
      }

      const NodePtr in_node = node_names_[input_name];
      if (in_node == nullptr) {
        error_code = GRAPH_FAILED;
        error_msg = "AddDataNodes failed: node " + input_name + " is NULL.";
        return;
      }

      if (GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), in_node->GetInDataAnchor(ind)) != GRAPH_SUCCESS) {
        error_code = GRAPH_FAILED;
        error_msg = "AddDataNodes failed: add data-edge Data:" + std::to_string(input.first) + ":0->" +
                    input_name + ":" + std::to_string(ind) + " failed.";
        return;
      }
    }

    GELOGD("AddDataNodes : Add %u input succ.", input.first);
  }

  GELOGD("AddDataNodes succ.");
}

///
/// @brief Add data node
/// @param [in] index
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
NodePtr CompleteGraphBuilder::AddDataNode(uint32_t index, graphStatus &error_code, std::string &error_msg) {
  const std::string data_name = "Data_" + std::to_string(index);
  OpDescBuilder op_desc_builder(data_name, "Data");
  const OpDescPtr op_desc = op_desc_builder.AddInput("x")
                                           .AddOutput("y")
                                           .Build();
  if (op_desc == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "AddDataNode failed: create op_desc " + data_name + " failed.";
    return nullptr;
  }

  const auto index_iter = input_mapping_.find(index);
  if (index_iter != input_mapping_.end()) {
    if (!ge::AttrUtils::SetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, static_cast<int64_t>(index_iter->second))) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNode failed: set attr ATTR_NAME_PARENT_NODE_INDEX for " + data_name + " failed.";
      return nullptr;
    }
  }
  if (parent_node_ != nullptr) {
    // op_desc should not be null
    const auto &parent_desc = parent_node_->GetOpDesc()->GetInputDesc(index_iter->second);
    if ((op_desc->UpdateInputDesc(0U, parent_desc) != GRAPH_SUCCESS) ||
        (op_desc->UpdateOutputDesc(0U, parent_desc) != GRAPH_SUCCESS)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNode failed: update tensor_desc for " + data_name + " failed.";
      return nullptr;
    }
  }

  const NodePtr data_node = owner_graph_->AddNode(op_desc);
  if (data_node == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "AddDataNode failed: add node " + data_name + " failed.";
    return nullptr;
  }
  node_names_[data_name] = data_node;

  return data_node;
}

///
/// @brief Add RetVal nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::AddRetValNodes(graphStatus &error_code, std::string &error_msg) {
  const size_t output_num = graph_outputs_.size();
  for (size_t i = 0U; i < output_num; i++) {
    const int32_t index = static_cast<int32_t>(graph_outputs_[i].second);
    const auto out_iter = node_names_.find(graph_outputs_[i].first);
    if (out_iter == node_names_.end()) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode failed: node " + graph_outputs_[i].first + " not exist in graph.";
      return;
    }
    const NodePtr node = out_iter->second;
    if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode failed: node is NULL.";
      return;
    }

    const std::string name = node->GetName() + "_RetVal_"+ std::to_string(index);
    const OpDescPtr ret_val_desc = ComGraphMakeShared<OpDesc>(name, FRAMEWORKOP);
    if (ret_val_desc == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: op_desc is NULL.";
      return;
    }
    const ge::GeTensorDesc tensor = node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(index));
    if ((ret_val_desc->AddInputDesc(tensor) != GRAPH_SUCCESS) ||
        (ret_val_desc->AddOutputDesc(tensor) != GRAPH_SUCCESS)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add input_desc / output_desc failed.";
      return;
    }

    if (!(ge::AttrUtils::SetStr(ret_val_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_RetVal") &&
          ge::AttrUtils::SetInt(ret_val_desc, RETVAL_ATTR_NAME_INDEX, static_cast<int64_t>(i)))) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: set FRAMEWORK_ORIGINAL_TYPE / RETVAL_ATTR_NAME_INDEX failed.";
      return;
    }
    const auto iter = output_mapping_.find(i);
    if (iter != output_mapping_.end()) {
      if (!ge::AttrUtils::SetInt(ret_val_desc, ATTR_NAME_PARENT_NODE_INDEX, static_cast<int64_t>(iter->second))) {
        error_code = GRAPH_FAILED;
        error_msg = "AddRetValNode " + name + " failed: set attr PARENT_NODE_INDEX failed.";
        return;
      }
    }

    const NodePtr ret_val_node = owner_graph_->AddNode(ret_val_desc);
    if (ret_val_node == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add node failed.";
      return;
    }

    if (GraphUtils::AddEdge(node->GetOutDataAnchor(index), ret_val_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add data-edge " +
                  node->GetName() + ":" + std::to_string(index) + "->" + ret_val_node->GetName() + ":0 failed.";
      return;
    }
  }

  GELOGD("AddRetValNodes succ.");
}

///
/// @brief Build target-nodes for graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::BuildGraphTargets(graphStatus &error_code, std::string &error_msg) {
  std::vector<NodePtr> target_nodes;
  for (const std::string &target_name : graph_targets_) {
    const auto target_iter = node_names_.find(target_name);
    if ((target_iter == node_names_.end()) || (target_iter->second == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = "BuildGraphTargets failed: target_node " + target_name + " not exist in graph.";
      return;
    }
    target_nodes.emplace_back(target_iter->second);
  }
  owner_graph_->SetGraphTargetNodesInfo(target_nodes);
}

///
/// @brief Add NetOutput node
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::AddNetOutputNode(graphStatus &error_code, std::string &error_msg) {
  if (graph_outputs_.empty() && graph_targets_.empty()) {
    return;
  }
  const std::string log_msg = "AddNetOutputNode name:" + std::string(NODE_NAME_NET_OUTPUT) + ", type:" + NETOUTPUT;
  const OpDescPtr net_output_desc = ComGraphMakeShared<OpDesc>(NODE_NAME_NET_OUTPUT, NETOUTPUT);
  if (net_output_desc == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = log_msg + " failed: op_desc is NULL.";
    return;
  }

  const size_t output_num = graph_outputs_.size();
  std::vector<OutDataAnchorPtr> peer_out_anchors(output_num);
  for (size_t i = 0U; i < output_num; i++) {
    const uint32_t index = graph_outputs_[i].second;
    const auto out_iter = node_names_.find(graph_outputs_[i].first);
    if (out_iter == node_names_.end()) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: node " + graph_outputs_[i].first + " not exist in graph.";
      return;
    }
    const NodePtr node = out_iter->second;
    if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: node is NULL.";
      return;
    }

    ge::GeTensorDesc tensor = node->GetOpDesc()->GetOutputDesc(index);
    int64_t update_index = static_cast<int64_t>(i);
    const auto iter = output_mapping_.find(i);
    if (iter != output_mapping_.end()) {
      update_index = static_cast<int64_t>(iter->second);
    }
    if (!ge::AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, update_index)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: set attr PARENT_NODE_INDEX failed.";
      return;
    }
    if (net_output_desc->AddInputDesc(tensor) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: add input_desc ailed.";
      return;
    }
    peer_out_anchors[i] = node->GetOutDataAnchor(static_cast<int32_t>(index));
  }

  BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return;
  }

  GELOGD("%s succ.", log_msg.c_str());
}

///
/// @brief Build NetOutput nodes with data & ctrl edges
/// @param [in] net_output_desc
/// @param [in] peer_out_anchors
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::BuildNetOutputNodeWithLink(const OpDescPtr &net_output_desc,
                                                      const std::vector<OutDataAnchorPtr> &peer_out_anchors,
                                                      graphStatus &error_code, std::string &error_msg) {
  const std::string log_msg = "AddNetOutputNode name:" + std::string(NODE_NAME_NET_OUTPUT) + ", type:" + NETOUTPUT;
  const NodePtr net_output = owner_graph_->AddNode(net_output_desc);
  if (net_output == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = log_msg + " failed: add NetOutput node failed.";
    return;
  }

  const size_t output_num = graph_outputs_.size();
  for (size_t i = 0U; i < output_num; i++) {
    if (GraphUtils::AddEdge(peer_out_anchors[i],
                            net_output->GetInDataAnchor(static_cast<int32_t>(i))) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: add data-edge " +
                  peer_out_anchors[i]->GetOwnerNode()->GetName() + ":" + std::to_string(peer_out_anchors[i]->GetIdx()) +
                  "->" + NODE_NAME_NET_OUTPUT + ":" + std::to_string(i) + " failed.";
      return;
    }
  }
  for (const std::string &target_name : graph_targets_) {
    const auto target_iter = node_names_.find(target_name);
    if ((target_iter == node_names_.end()) || (target_iter->second == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = "BuildGraphTargets failed: target_node " + target_name + " not exist in graph.";
      return;
    }
    const auto &target_node = target_iter->second;
    if (GraphUtils::AddEdge(target_node->GetOutControlAnchor(), net_output->GetInControlAnchor()) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: add ctrl-edge " +
                  target_node->GetName() + "->" + NODE_NAME_NET_OUTPUT + " failed.";
      return;
    }
  }
}

///
/// @brief process after build
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::PostProcess(graphStatus &error_code, std::string &error_msg) {
  if (parent_node_ != nullptr) {
    owner_graph_->SetParentNode(parent_node_);
    const auto &parent_graph = parent_node_->GetOwnerComputeGraph();
    if (parent_graph == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "Parent graph is null, parent_node=" + parent_node_->GetName();
      return;
    }
    owner_graph_->SetParentGraph(parent_graph);
    // ATTR_NAME_SESSION_GRAPH_ID
    std::string graph_id;
    if ((!AttrUtils::GetStr(parent_graph, ATTR_NAME_SESSION_GRAPH_ID, graph_id)) ||
        (!AttrUtils::SetStr(owner_graph_, ATTR_NAME_SESSION_GRAPH_ID, graph_id))) {
      error_code = GRAPH_FAILED;
      error_msg = "Copy attr session_graph_id failed.";
      return;
    }
    if (parent_graph->HasAttr(ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED)) {
      bool is_dynamic_shape = false;
      if ((!AttrUtils::GetBool(parent_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape)) ||
          (!AttrUtils::SetBool(owner_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape))) {
        error_code = GRAPH_FAILED;
        error_msg = "Copy attr _dynamic_shape_partitioned failed.";
        return;
      }
    }
    owner_graph_->SetGraphUnknownFlag(parent_graph->GetGraphUnknownFlag());

    // refresh parent node/graph in subgraphs
    for (const NodePtr &node : owner_graph_->GetDirectNode()) {
      std::vector<ComputeGraphPtr> subgraphs;
      if (NodeUtils::GetDirectSubgraphs(node, subgraphs) != GRAPH_SUCCESS) {
        error_code = GRAPH_FAILED;
        error_msg = "Get subgraphs for failed failed, node:" + node->GetName();
        return;
      }
      for (const auto &subgraph : subgraphs) {
        subgraph->SetParentNode(node);
        subgraph->SetParentGraph(subgraph);
      }
    }
  }

  // refresh node name
  for (const NodePtr &node : owner_graph_->GetDirectNode()) {
    if ((node->GetOpDesc() == nullptr) || (node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2)) {
      continue;
    }
    node->GetOpDesc()->SetName(owner_graph_->GetName() + "/" + node->GetName());
  }
}

///
/// @brief Add node to graph
/// @param [in] op_desc
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  (void)ComputeGraphBuilder::AddNode(op_desc);
  return *this;
}

///
/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::AddDataLink(const std::string &src_name, uint32_t out_anchor_ind,
                                                      const std::string &dst_name, uint32_t in_anchor_ind) {
  (void)ComputeGraphBuilder::AddDataLink(src_name, out_anchor_ind, dst_name, in_anchor_ind);
  return *this;
}

///
/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  (void)ComputeGraphBuilder::AddControlLink(src_name, dst_name);
  return *this;
}

///
/// @brief Set owner graph
/// @param [in] graph
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::SetOwnerGraph(const ComputeGraphPtr &graph) {
  owner_graph_ = graph;
  return *this;
}

///
/// @brief Add exist node
/// @param [in] node
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::AddExistNode(const NodePtr &exist_node) {
  exist_nodes_.emplace_back(exist_node);
  return *this;
}

///
/// @brief Build partial graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return ComputeGraphPtr
///
ComputeGraphPtr PartialGraphBuilder::Build(graphStatus &error_code, std::string &error_msg) {
  if (owner_graph_ == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return nullptr;
  }

  BuildNodes(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildExistNodes(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildDataLinks(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildCtrlLinks(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  return owner_graph_;
}

///
/// @brief Build exist nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void PartialGraphBuilder::BuildExistNodes(graphStatus &error_code, std::string &error_msg) {
  std::string node_name;
  for (auto &exist_node : exist_nodes_) {
    if (exist_node == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "Build exist nodes failed: node is NULL.";
      return;
    }

    node_name = exist_node->GetName();
    if (exist_node->GetOwnerComputeGraph() != owner_graph_) {
      error_code = GRAPH_FAILED;
      error_msg = "Build exist nodes failed: node " + node_name + " not belongs to this graph.";
      return;
    }

    GELOGD("Add exist_node name:%s.", node_name.c_str());
    node_names_[node_name] = exist_node;
  }

  GELOGD("Build exist nodes succ.");
}

}  // namespace ge
