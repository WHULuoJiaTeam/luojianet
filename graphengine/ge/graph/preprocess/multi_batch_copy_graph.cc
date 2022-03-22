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
#include "graph/preprocess/multi_batch_copy_graph.h"

#include <queue>
#include <set>
#include <string>

#include "common/formats/utils/formats_trans_utils.h"
#include "common/ge/ge_util.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/omg/omg_inner_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "graph/passes/multi_batch_clone_pass.h"
#include "graph/passes/prune_pass.h"
#include "graph/preprocess/multi_batch_options.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/pass_manager.h"
#include "common/local_context.h"
#include "common/omg_util.h"

using std::set;
using std::string;
using std::vector;
using std::map;
using std::queue;

namespace ge {
namespace multibatch {
namespace {
const char *const kMbatchSwitchnName = "mbatch-switch-name";
const char *const kGetNextName = "IteratorV2";
const int kSwitchNDataIndex = 0;
const int kSwitchNPredIndex = 1;
const int kDataOutIndex = 0;
const int kDataInIndex = 0;
const int kMergeDataOutIndex = 0;
const int kStaticOutput = -1;
const int kDivisionConst = 2;
const int32_t kOneInDataNode = 1;
const int32_t kFindNoMatch = 0;


inline bool IsDataLikeType(const std::string &node_type) { return (node_type == DATA) || (node_type == AIPP); }

inline bool IsEnterType(const string &node_type) { return (node_type == ENTER) || (node_type == REFENTER); }
const set<string> unchange_types({CONSTANT, CONSTANTOP, ENTER, REFENTER});

inline bool IsGetNextType(const NodePtr &node) {
  std::string original_type;
  GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS,
                  GELOGW("Get original type failed"); return false);
  return (original_type == kGetNextName);
}

NodePtr InsertMergeNodeToGraph(const std::string &name, size_t input_num, const ComputeGraphPtr &graph) {
  OpDescPtr desc = MakeShared<OpDesc>();
  if (desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(OUT_OF_MEMORY, "[New][OpDesc] failed, name %s", name.c_str());
    return nullptr;
  }
  desc->SetName(name);
  desc->SetType(MERGE);
  GeTensorDesc tensor_desc;
  for (size_t i = 0; i < input_num; ++i) {
    auto ret = desc->AddInputDesc("x" + std::to_string(i), tensor_desc);
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed, input desc name:%s",
                                      desc->GetName().c_str(), desc->GetType().c_str(),
                                      ("x" + std::to_string(i)).c_str());
                    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed, input desc name:%s",
                           desc->GetName().c_str(), desc->GetType().c_str(), ("x" + std::to_string(i)).c_str());
                    return nullptr);
  }
  auto ret = desc->AddOutputDesc("y", tensor_desc);
  GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed, output desc name:%s",
                                    desc->GetName().c_str(), desc->GetType().c_str(), "y");
                  GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] to op:%s(%s) failed, output desc name:y",
                         desc->GetName().c_str(), desc->GetType().c_str());
                  return nullptr);
  tensor_desc.SetDataType(DT_INT32);
  ret = desc->AddOutputDesc("value_index", tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed, output desc name:%s",
                      desc->GetName().c_str(), desc->GetType().c_str(), "value_index");
    GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] to op:%s(%s) failed, output desc name:value_index",
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }

  if (!AttrUtils::SetBool(desc, ATTR_INSERT_BY_MBATCH, true)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_INSERT_BY_MBATCH.c_str(),
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to op:%s(%s) failed", ATTR_INSERT_BY_MBATCH.c_str(),
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }
  return graph->AddNode(desc);
}

NodePtr InsertCopyNode(const NodePtr &node, size_t n) {
  const std::string &name = node->GetName() + "_ascend_mbatch_batch_" + std::to_string(n);
  auto src_op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(src_op_desc == nullptr,
                  REPORT_INNER_ERROR("E19999", "Param opdesc in node is nullptr, check invalid");
                  GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, src_op_desc is nullptr");
                  return nullptr);

  auto desc = AttrUtils::CopyOpDesc(src_op_desc);
  GE_IF_BOOL_EXEC(desc == nullptr,
                  REPORT_CALL_ERROR("E19999", "Copy OpDesc from op:%s(%s) failed",
                                    src_op_desc->GetName().c_str(), src_op_desc->GetType().c_str());
                  GELOGE(OUT_OF_MEMORY, "[Copy][OpDesc] from op:%s(%s) failed",
                         src_op_desc->GetName().c_str(), src_op_desc->GetType().c_str());
                  return nullptr);

  desc->SetName(name);
  desc->CopyAttrsFrom(*src_op_desc);
  for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
    auto input_desc = desc->MutableInputDesc(i);
    GE_IF_BOOL_EXEC(input_desc == nullptr,
                    REPORT_INNER_ERROR("E19999", "Input desc of op:%s(%s) not exist, index:%u, check invalid",
                                       desc->GetName().c_str(), desc->GetType().c_str(), i);
                    GELOGW("Get null input desc by index %u from node %s when copy from %s", i,
                           desc->GetName().c_str(), node->GetName().c_str());
                    continue);

    input_desc->CopyAttrsFrom(src_op_desc->GetInputDesc(i));
  }
  for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
    auto output_desc = desc->MutableOutputDesc(i);
    GE_IF_BOOL_EXEC(output_desc == nullptr,
                    REPORT_INNER_ERROR("E19999", "Ouput desc of op:%s(%s) not exist, index:%u, check invalid",
                                       desc->GetName().c_str(), desc->GetType().c_str(), i);
                    GELOGE(INTERNAL_ERROR, "[Call][MutableOutputDesc] Ouput desc of op:%s(%s) not exist, index:%u",
                           desc->GetName().c_str(), desc->GetType().c_str(), i);
                    return nullptr);

    output_desc->CopyAttrsFrom(src_op_desc->GetOutputDesc(i));
  }
  const std::string &batch_label = "Batch_" + std::to_string(n);
  if (!AttrUtils::SetStr(desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_BATCH_LABEL.c_str(),
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(FAILED, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_BATCH_LABEL.c_str(),
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }

  (void)AttrUtils::SetListStr(desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {node->GetName()});

  auto graph = node->GetOwnerComputeGraph();
  return graph->AddNode(desc);
}

bool IsAllDimsPositive(const std::vector<int64_t> &dims) {
  for (auto dim : dims) {
    if (dim < 0) {
      return false;
    }
  }
  return true;
}

NodePtr InsertConst(const std::string &name, const ComputeGraphPtr &graph) {
  auto desc = MakeShared<OpDesc>();
  if (desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(OUT_OF_MEMORY, "[Create][ConstOp] %s failed, out of memory", name.c_str());
    return nullptr;
  }
  desc->SetName(name);
  desc->SetType(CONSTANT);
  GeTensor tensor;
  tensor.SetData(std::vector<uint8_t>({0}));
  if (!AttrUtils::SetTensor(desc, ATTR_NAME_WEIGHTS, tensor)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(),
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(OUT_OF_MEMORY, "[Set][Attr] %s to op:%s(%s) failed", ATTR_NAME_WEIGHTS.c_str(),
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }
  if (!AttrUtils::SetBool(desc, ATTR_INSERT_BY_MBATCH, true)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to op:%s(%s) failed", ATTR_INSERT_BY_MBATCH.c_str(),
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(OUT_OF_MEMORY, "[Set][Attr] %s to op:%s(%s) failed", ATTR_INSERT_BY_MBATCH.c_str(),
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }
  if (desc->AddOutputDesc(GeTensorDesc()) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(OUT_OF_MEMORY, "[Add][OutputDesc] to op:%s(%s) failed",
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }
  return graph->AddNode(desc);
}

bool IsOnlyOutputToAipp(const NodePtr &node) {
  for (const auto &out_node : node->GetOutDataNodes()) {
    if (out_node->GetType() != AIPP) {
      return false;
    }
  }
  return true;
}
}  // namespace

Status MultiBatchGraphCopyer::CopyGraph() {
  auto ret = Init();
  if (ret != SUCCESS) {
    return ret;
  }

  if (LabelStatus() != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "[Label][Status] for all nodes failed.");
    return INTERNAL_ERROR;
  }

  ret = CheckAndParseDynamicData();
  if (ret != SUCCESS) {
    return ret;
  }

  ret = CreateNewNodes();
  if (ret != SUCCESS) {
    return ret;
  }

  ret = LinkEdges();
  if (ret != SUCCESS) {
    return ret;
  }

  GELOGI("Begin to remove useless nodes by prune pass after copy process");
  PrunePass prune_pass;
  ret = prune_pass.Run(graph_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Run][PrunePass] failed.");
    return ret;
  }
  return CheckCopyResult(origin_data_nodes_);
}

Status MultiBatchGraphCopyer::Init() {
  auto ret = CheckArguments();
  if (ret != SUCCESS) {
    return ret;
  }

  ret = RelinkConstCtrlEdge();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Relink][ConstCtrlEdge] failed.");
    return FAILED;
  }

  ret = ExtractUnchangedStructureOutofCycle();
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Call][ExtractUnchangedStructureOutofCycle] failed.");
    return FAILED;
  }

  for (auto &node : graph_->GetAllNodes()) {
    origin_all_nodes_.emplace_back(node);
    if (IsDataLikeType(node->GetType())) {
      origin_data_nodes_.emplace_back(node);
    }
    if (!GetLocalOmgContext().dynamic_node_type.empty() && IsGetNextType(node)) {
      origin_data_nodes_.emplace_back(node);
    }
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::RelinkConstCtrlEdge() {
  for (auto &node : graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if ((node->GetType() == CONSTANT) || (node->GetType() == CONSTANTOP)) {
      if (node->GetOutDataNodes().empty()) {
        continue;
      }
      if (!node->GetInControlNodes().empty()) {
        auto in_ctrl_nodes = node->GetInControlNodes();
        auto out_nodes = node->GetOutAllNodes();
        bool has_merge_out = false;
        for (const auto &out_node : out_nodes) {
          GE_CHECK_NOTNULL(out_node);
          if (out_node->GetType() == MERGE || out_node->GetType() == REFMERGE) {
            has_merge_out = true;
            break;
          }
        }
        if (has_merge_out) {
          continue;
        }
        auto in_ctrl_anchor = node->GetInControlAnchor();
        GE_CHECK_NOTNULL(in_ctrl_anchor);
        in_ctrl_anchor->UnlinkAll();
        for (auto &in_ctrl_node : in_ctrl_nodes) {
          auto out_ctrl_anchor_of_in_ctrl_node = in_ctrl_node->GetOutControlAnchor();
          GE_CHECK_NOTNULL(out_ctrl_anchor_of_in_ctrl_node);
          for (auto &out_node : out_nodes) {
            if (IsEnterType(out_node->GetType())) {
              continue;
            }
            if (!out_ctrl_anchor_of_in_ctrl_node->IsLinkedWith(out_node->GetInControlAnchor())) {
              GE_CHK_GRAPH_STATUS_RET(out_ctrl_anchor_of_in_ctrl_node->LinkTo(out_node->GetInControlAnchor()))
            }
          }
        }
      }
      auto out_ctrl_anchor = node->GetOutControlAnchor();
      if (out_ctrl_anchor != nullptr) {
        out_ctrl_anchor->UnlinkAll();
      }
    }
  }

  return SUCCESS;
}

Status MultiBatchGraphCopyer::ExtractUnchangedStructureOutofCycle() {
  map<string, vector<NodePtr>> frame_enter;
  if (GetEnterNodesGroupByFrame(frame_enter) != SUCCESS) {
    GELOGE(FAILED, "[Call][GetEnterNodesGroupByFrame] failed.");
    return FAILED;
  }

  queue<NodePtr> nodes_to_extract;
  if (GetNodeNeedExtract(frame_enter, nodes_to_extract) != SUCCESS) {
    GELOGE(FAILED, "[Call][GetNodeNeedExtract] failed.");
    return FAILED;
  }

  while (!nodes_to_extract.empty()) {
    auto node = nodes_to_extract.front();
    nodes_to_extract.pop();
    OpDescPtr enter_desc = nullptr;
    if (MoveInEntersInDataAnchorDown(node, enter_desc) != SUCCESS) {
      GELOGE(FAILED, "[Call][MoveInEntersInDataAnchorDown] for node:%s failed.", node->GetName().c_str());
      return FAILED;
    }
    set<NodePtr> out_nodes;
    if (InsertEnterAfterNode(node, enter_desc, out_nodes) != SUCCESS) {
      GELOGE(FAILED, "[Insert][EnterNode] after node:%s failed.", node->GetName().c_str());
      return FAILED;
    }

    if (MoveCtrlEdgeToOutNodes(node, out_nodes) != SUCCESS) {
      GELOGE(FAILED, "[Call][MoveCtrlEdgeToOutNodes] for node:%s failed.", node->GetName().c_str());
      return FAILED;
    }

    for (auto &out_node : out_nodes) {
      GE_CHECK_NOTNULL(out_node);
      if (AllInDataNodesUnchangeAndNoMergeOut(out_node)) {
        nodes_to_extract.push(out_node);
      }
    }
  }

  if (DeleteEnterWithoutDataOut() != SUCCESS) {
    GELOGE(FAILED, "[Call][DeleteEnterWithoutDataOut] failed.");
    return FAILED;
  }

  return SUCCESS;
}

Status MultiBatchGraphCopyer::GetEnterNodesGroupByFrame(map<string, vector<NodePtr>> &frame_enter) {
  for (auto &node : graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (IsEnterType(node->GetType())) {
      if (!node->GetInControlNodes().empty() || !node->GetOutControlNodes().empty()) {
        continue;
      }
      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      string frame_name;
      if (!AttrUtils::GetStr(op_desc, ENTER_ATTR_FRAME_NAME, frame_name)) {
        REPORT_CALL_ERROR("E19999", "Get Attr:%s on op:%s(%s) failed",
                          ENTER_ATTR_FRAME_NAME.c_str(),
                          op_desc->GetName().c_str(), op_desc->GetType().c_str());
        GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) failed",
               ENTER_ATTR_FRAME_NAME.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
        return FAILED;
      }
      frame_enter[frame_name].emplace_back(node);
    }
  }

  return SUCCESS;
}

Status MultiBatchGraphCopyer::GetNodeNeedExtract(const map<string, vector<NodePtr>> &frame_enter,
                                                 queue<NodePtr> &nodes_to_extract) {
  for (const auto &one_group : frame_enter) {
    auto enters = one_group.second;
    for (const auto &enter : enters) {
      auto out_data_nodes = enter->GetOutDataNodes();
      for (const auto &out_data_node : out_data_nodes) {
        GE_CHECK_NOTNULL(out_data_node);
        if (AllInDataNodesUnchangeAndNoMergeOut(out_data_node)) {
          nodes_to_extract.push(out_data_node);
        }
      }
    }
  }

  return SUCCESS;
}

bool MultiBatchGraphCopyer::AllInDataNodesUnchangeAndNoMergeOut(const NodePtr &node) {
  auto out_data_nodes = node->GetOutDataNodes();
  for (const auto &out_data_node : out_data_nodes) {
    if (out_data_node == nullptr) {
      return false;
    }

    if (out_data_node->GetType() == MERGE || out_data_node->GetType() == REFMERGE) {
      return false;
    }
  }

  auto in_data_nodes = node->GetInDataNodes();
  if (in_data_nodes.size() == kOneInDataNode) {
    return true;
  }

  for (const auto &in_data_node : in_data_nodes) {
    if (in_data_node == nullptr) {
      return false;
    }
    if (unchange_types.count(in_data_node->GetType()) == kFindNoMatch) {
      return false;
    }
  }

  return true;
}

Status MultiBatchGraphCopyer::MoveInEntersInDataAnchorDown(NodePtr &node, OpDescPtr &enter_desc) {
  auto in_data_anchors = node->GetAllInDataAnchors();
  for (auto &in_data_anchor : in_data_anchors) {
    auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_data_anchor);
    auto peer_in_data_node = peer_out_data_anchor->GetOwnerNode();
    if (IsEnterType(peer_in_data_node->GetType())) {
      GE_CHK_GRAPH_STATUS_RET(peer_out_data_anchor->Unlink(in_data_anchor))
      GELOGD("Unlink data edge from %s to %s.", peer_in_data_node->GetName().c_str(), node->GetName().c_str());
      auto enter_in_data_anchors = peer_in_data_node->GetAllInDataAnchors();
      for (auto &enter_in_data_anchor : enter_in_data_anchors) {
        auto peer_out_data_anchor_of_enter = enter_in_data_anchor->GetPeerOutAnchor();
        GE_CHECK_NOTNULL(peer_out_data_anchor_of_enter);
        if (peer_out_data_anchor_of_enter->IsLinkedWith(in_data_anchor)) {
          continue;
        }
        GE_CHK_GRAPH_STATUS_RET(peer_out_data_anchor_of_enter->LinkTo(in_data_anchor))
        GELOGD("Relink data edge from %s to %s.", peer_out_data_anchor_of_enter->GetOwnerNode()->GetName().c_str(),
               node->GetName().c_str());
      }
      enter_desc = peer_in_data_node->GetOpDesc();
      GE_CHECK_NOTNULL(enter_desc);
    }
  }

  return SUCCESS;
}

Status MultiBatchGraphCopyer::InsertEnterAfterNode(NodePtr &node, const OpDescPtr &copy_desc, set<NodePtr> &out_nodes) {
  if (copy_desc == nullptr) {
    return SUCCESS;
  }
  map<OutDataAnchorPtr, vector<std::pair<InDataAnchorPtr, NodePtr>>> outanchors_inanchors_nodes;
  auto out_data_anchors = node->GetAllOutDataAnchors();
  for (auto &out_data_anchor : out_data_anchors) {
    auto peer_in_data_anchors = out_data_anchor->GetPeerInDataAnchors();
    for (auto peer_in_data_anchor : peer_in_data_anchors) {
      GE_CHECK_NOTNULL(peer_in_data_anchor);
      auto peer_in_data_node = peer_in_data_anchor->GetOwnerNode();
      out_nodes.emplace(peer_in_data_node);
      outanchors_inanchors_nodes[out_data_anchor].emplace_back(std::make_pair(peer_in_data_anchor, peer_in_data_node));
    }
  }

  int32_t i = 0;
  auto node_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(node_desc);
  // Insert one enter node after node's per out data anchor
  for (auto &outanchor_inanchors_nodes : outanchors_inanchors_nodes) {
    string name = node->GetName() + "_" + ENTER + "_" + std::to_string(i++);
    GELOGD("Create Enter op %s after %s.", name.c_str(), node->GetName().c_str());
    auto enter_desc = AttrUtils::CopyOpDesc(copy_desc);
    enter_desc->SetName(name);
    GE_CHK_GRAPH_STATUS_RET(
        enter_desc->UpdateInputDesc("x", node_desc->GetOutputDesc(outanchor_inanchors_nodes.first->GetIdx())))
    GE_CHK_GRAPH_STATUS_RET(
        enter_desc->UpdateOutputDesc("y", node_desc->GetOutputDesc(outanchor_inanchors_nodes.first->GetIdx())))
    auto enter_node = graph_->AddNode(enter_desc);
    GE_CHECK_NOTNULL(enter_node);
    GE_CHK_GRAPH_STATUS_RET(outanchor_inanchors_nodes.first->LinkTo(enter_node->GetInDataAnchor(kDataInIndex)))
    GE_CHECK_NOTNULL(enter_node->GetOutDataAnchor(kDataInIndex));
    for (auto &inanchor_node : outanchor_inanchors_nodes.second) {
      GE_CHK_GRAPH_STATUS_RET(outanchor_inanchors_nodes.first->Unlink(inanchor_node.first))
      GE_CHK_GRAPH_STATUS_RET(enter_node->GetOutDataAnchor(kDataInIndex)->LinkTo(inanchor_node.first))
      GELOGD("Unlink from %s to %s, link from %s to %s then to %s.", node->GetName().c_str(),
             inanchor_node.second->GetName().c_str(), node->GetName().c_str(), enter_node->GetName().c_str(),
             inanchor_node.second->GetName().c_str());
    }
  }

  return SUCCESS;
}

// Move node's in control edges to out data nodes
Status MultiBatchGraphCopyer::MoveCtrlEdgeToOutNodes(NodePtr &node, set<NodePtr> &out_nodes) {
  auto in_ctrl_anchor = node->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  auto peer_out_ctrl_anchors = in_ctrl_anchor->GetPeerOutControlAnchors();
  for (auto &peer_out_ctrl_anchor : peer_out_ctrl_anchors) {
    GE_CHK_GRAPH_STATUS_RET(peer_out_ctrl_anchor->Unlink(in_ctrl_anchor))
    GELOGD("Unlink control edge from %s to %s.", peer_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
           node->GetName().c_str());
    for (auto &out_node : out_nodes) {
      auto in_ctrl_anchor_of_out_node = out_node->GetInControlAnchor();
      GE_CHECK_NOTNULL(in_ctrl_anchor_of_out_node);
      if (!peer_out_ctrl_anchor->IsLinkedWith(in_ctrl_anchor_of_out_node)) {
        GE_CHK_GRAPH_STATUS_RET(peer_out_ctrl_anchor->LinkTo(in_ctrl_anchor_of_out_node))
        GELOGD("Link control edge from %s to %s.", peer_out_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
               out_node->GetName().c_str());
      }
    }
  }

  return SUCCESS;
}

Status MultiBatchGraphCopyer::DeleteEnterWithoutDataOut() {
  for (auto &node : graph_->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    if (IsEnterType(node->GetType())) {
      auto out_nodes = node->GetOutAllNodes();
      if (out_nodes.empty()) {
        GELOGD("Delete enter node: %s which has no output.", node->GetName().c_str());
        GE_CHK_GRAPH_STATUS_RET(GraphUtils::IsolateNode(node, {}))
        GE_CHK_GRAPH_STATUS_RET(GraphUtils::RemoveNodeWithoutRelink(graph_, node))
      }
    }
  }

  return SUCCESS;
}

void MultiBatchGraphCopyer::LabelStatusForData(const NodePtr &data) {
  auto data_shape = NodeUtils::GetOutputDesc(*data, kDataOutIndex).GetShape();
  GELOGI("Label status for %s, shape_dims is %s.", data->GetName().c_str(),
         formats::JoinToString(data_shape.GetDims()).c_str());
  if (!IsAllDimsPositive(data_shape.GetDims())) {
    origin_nodes_status_[data.get()] = kNodeInBatchBranch;
  }
}

void MultiBatchGraphCopyer::LabelStatusForGetNextSink(const NodePtr &data) {
  auto op_desc = data->GetOpDesc();
  GELOGI("Out count of %s is %zu.", data->GetName().c_str(), op_desc->GetOutputsSize());
  size_t data_count = op_desc->GetOutputsSize() / kDivisionConst;
  for (size_t i = 0; i < data_count; ++i) {
    GeTensorDesc output_desc = op_desc->GetOutputDesc(i);
    GELOGD("The %zu data shape from getnext sink is %s.", i,
           formats::JoinToString(output_desc.GetShape().GetDims()).c_str());
    const auto &out_data_anchor = data->GetOutDataAnchor(i);
    if (out_data_anchor == nullptr) {
      continue;
    }
    size_t reference_times = out_data_anchor->GetPeerInDataAnchors().size();
    GELOGD("The %zu data has %zu referenced times.", i, reference_times);
    getnext_sink_dynamic_out_mapping_.emplace_back(std::make_pair(i, reference_times));
    if (!IsAllDimsPositive(output_desc.GetShape().GetDims())) {
      getnext_sink_dynamic_dims_ = true;
    }
  }

  if (getnext_sink_dynamic_dims_) {
    origin_nodes_status_[data.get()] = kNodeInBatchBranch;
  }
}

Status MultiBatchGraphCopyer::LabelInBatchBranchStatus() {
  GELOGD("Start label in batch branch status.");
  for (const auto &data : origin_data_nodes_) {
    auto op_desc = data->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr,
        REPORT_INNER_ERROR("E19999", "op_desc in origin_data_nodes_ is nullptr, check invalid");
        GELOGE(PARAM_INVALID, "[Get][OpDesc] failed, op_desc is nullptr.");
        return PARAM_INVALID);
    LabelStatusForData(data);
    if (!GetLocalOmgContext().dynamic_node_type.empty()) {
      LabelStatusForGetNextSink(data);
    }
  }

  map<string, vector<NodePtr>> frame_enters;
  InitStatus(frame_enters);
  bool changed = true;
  // If anyone of in node is kNodeInBatchBranch, it is also kNodeInBatchBranch
  while (changed) {
    changed = false;
    for (const auto &node : origin_all_nodes_) {
      auto iter = origin_nodes_status_.find(node.get());
      if (iter != origin_nodes_status_.end()) {
        continue;
      }
      for (auto &in_node : node->GetInDataNodes()) {
        if (origin_nodes_status_.find(in_node.get()) != origin_nodes_status_.end()) {
          if (origin_nodes_status_.find(node.get()) == origin_nodes_status_.end()) {
            origin_nodes_status_[node.get()] = kNodeInBatchBranch;
            ResetEnterStatus(frame_enters, node);
            changed = true;
          }
          break;
        }
      }
    }
  }
  return SUCCESS;
}

void MultiBatchGraphCopyer::InitStatus(map<string, vector<NodePtr>> &frame_enters) {
  for (const auto &node : origin_all_nodes_) {
    if (!IsEnterType(node->GetType())) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    string frame_name;
    if (AttrUtils::GetStr(op_desc, ENTER_ATTR_FRAME_NAME, frame_name)) {
      frame_enters[frame_name].emplace_back(node);
    }
  }

  for (const auto &data : origin_data_nodes_) {
    auto data_shape = NodeUtils::GetOutputDesc(*data, kDataOutIndex).GetShape();
    if (!IsAllDimsPositive(data_shape.GetDims())) {
      origin_nodes_status_[data.get()] = kNodeInBatchBranch;
    }
  }
}

void MultiBatchGraphCopyer::ResetEnterStatus(map<string, vector<NodePtr>> &frame_enters, const NodePtr &node) {
  if (!IsEnterType(node->GetType())) {
    return;
  }

  for (const auto &frame_enter : frame_enters) {
    auto &enters = frame_enter.second;
    if (std::find(enters.begin(), enters.end(), node) != enters.end()) {
      for (const auto &enter : enters) {
        origin_nodes_status_[enter.get()] = kNodeInBatchBranch;
      }
      break;
    }
  }
}

Status MultiBatchGraphCopyer::LabelStatus() {
  if (LabelInBatchBranchStatus() != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Call][LabelInBatchBranchStatus] failed.");
    return PARAM_INVALID;
  }

  for (const auto &node : origin_all_nodes_) {
    if (!(node->GetOpDesc()->GetSubgraphInstanceNames().empty())) {
      origin_nodes_status_[node.get()] = kNodeNotSupportNode;
      continue;
    }
    if (node->GetType() == NETOUTPUT) {
      origin_nodes_status_[node.get()] = kNodeOutBatchBranch;
      continue;
    }
    if (GetLocalOmgContext().dynamic_node_type.empty()) {
      if (IsDataLikeType(node->GetType())) {
        if (IsOnlyOutputToAipp(node)) {
          origin_nodes_status_[node.get()] = kNodeOutBatchBranch;
        } else {
          origin_nodes_status_[node.get()] = kNodeStartNode;
        }
        continue;
      }
    } else {
      if (IsDataLikeType(node->GetType())) {
        origin_nodes_status_[node.get()] = kNodeStartNode;
        continue;
      }
      if (IsGetNextType(node)) {
        origin_nodes_status_[node.get()] = kNodeStartNode;
        continue;
      }
    }
    if (origin_nodes_status_.find(node.get()) == origin_nodes_status_.end()) {
      origin_nodes_status_[node.get()] = kNodeOutBatchBranch;
    }
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::CheckAndParseDynamicData(){
  size_t unknown_shape_count = 0;
  auto data_name_and_shape = GetLocalOmgContext().user_input_dims;
  GELOGD("raw data_name_and_shape size: %zu", data_name_and_shape.size());
  if (!getnext_sink_dynamic_dims_) {
    for (const auto &node : origin_all_nodes_) {
      auto data_desc = NodeUtils::GetOutputDesc(*node, kDataOutIndex);
      auto data_shape = data_desc.GetShape();
      auto data_format = data_desc.GetFormat() == Format::FORMAT_NCHW ? "NCHW" :
                         data_desc.GetFormat() == Format::FORMAT_NHWC ? "NHWC" : "Others";
      auto data_name = node->GetName();
      auto branch_status = GetNodeStatus(node);
      if (branch_status != kNodeStartNode) {
        continue;
      }
      GELOGI("CheckAndParseDynamicData shape_dims is %s.", formats::JoinToString(data_shape.GetDims()).c_str());
      if (IsAllDimsPositive(data_shape.GetDims())) {
        continue;
      }

      std::vector<int64_t> data_shape_dims = data_shape.GetDims();
      ++unknown_shape_count;
      auto iter = find(data_name_order_.begin(), data_name_order_.end(), data_name);
      if (iter == data_name_order_.end()) {
        if (dynamic_type_ == DynamicType::kDynamicBatch) {
          auto ret = CheckDynamicBatchShape(data_shape_dims, data_name);
          GE_IF_BOOL_EXEC(ret == false, GELOGE(PARAM_INVALID, "[Check][DynamicBatchShape] of %s failed.",
                                               data_name.c_str()); return PARAM_INVALID);
        } else if (dynamic_type_ == DynamicType::kDynamicImageSize) {
          auto ret = CheckDynamicImageSizeShape(data_shape_dims, data_name, data_format);
          GE_IF_BOOL_EXEC(ret == false, GELOGE(PARAM_INVALID, "[Check][DynamicImageSizeShape] of %s failed.",
                                               data_name.c_str()); return PARAM_INVALID);
        } else if (dynamic_type_ == DynamicType::kDynamicDims) {
          ErrorManager::GetInstance().ATCReportErrMessage("E10001",
                                                          {"parameter", "value" "reason"},
                                                          {"--dynamic_dims", data_name,
                                                           "all dynamic data must be set in --input_shape"});
          GELOGE(INTERNAL_ERROR, "[Check][Param] data:%s shape:%s must be set int --input_shape",
                 node->GetName().c_str(), data_shape.ToString().c_str());
          return INTERNAL_ERROR;
        }
        GELOGI("Data shape of %s is %s", data_name.c_str(), formats::JoinToString(data_shape_dims).c_str());
        data_name_and_shape.emplace_back(data_name, data_shape_dims);
      }
    }
  }
  auto ret = ParserDataToDynamicInfo(shapes_, data_name_and_shape, data_to_dynamic_info_);
  GE_CHK_STATUS_RET(ret, "[Call][ParserDataToDynamicInfo] failed.");
  if (!getnext_sink_dynamic_dims_ && unknown_shape_count == 0) {
    ErrorManager::GetInstance().ATCReportErrMessage("E10040");
    GELOGE(PARAM_INVALID, "[Check][Param] Need unknow shape data "
           "when user set --dynamic_batch_size, --dynamic_image_size or --dynamic_dims");
    return PARAM_INVALID;
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::CreateNewNodes() {
  if (!getnext_sink_dynamic_dims_) {
    shape_data_ = InsertShapeDataNode();
  } else {
    shape_data_ = InsertGetDynamicDimsNode();
  }
  GE_IF_BOOL_EXEC(shape_data_ == nullptr, GELOGE(INTERNAL_ERROR, "[Create][TheShapeNode] for multi batch failed");
                  return INTERNAL_ERROR);
  GE_CHECK_NOTNULL(shape_data_->GetOpDesc());

  for (const auto &node : origin_all_nodes_) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    auto node_type = node->GetType();
    Status ret = INTERNAL_ERROR;
    auto branch_status = GetNodeStatus(node);
    GELOGD("Process node %s, status %d", node->GetName().c_str(), static_cast<int>(branch_status));
    switch (branch_status) {
      case kNodeStartNode:
        GELOGD("Name: %s, type: %s, status: kNodeStartNode.", node->GetName().c_str(), node->GetType().c_str());
        ret = InsertSwitchNAndUpdateMaxShape(node);
        break;
      case kNodeInBatchBranch:
        GELOGD("Name: %s, type: %s, status: kNodeInBatchBranch.", node->GetName().c_str(), node->GetType().c_str());
        ret = CopyNodeInBatchBranch(node);
        break;
      case kNodeOutBatchBranch:
        GELOGD("Name: %s, type: %s, status: kNodeOutBatchBranch.", node->GetName().c_str(), node->GetType().c_str());
        ret = InsertMergeForEdgeNode(node);
        if (ret == SUCCESS) {
          ret = LinkGetDynamicDimsToNetOutput(node);
        }
        break;
      case kNodeNotSupportNode:
        GELOGD("Name: %s, type: %s, status: kNodeNotSupportNode.", node->GetName().c_str(), node->GetType().c_str());
        break;
      default:
        GELOGE(INTERNAL_ERROR, "[Get][NodeStatus] Unexpected status %d on node %s", static_cast<int>(branch_status),
               node->GetName().c_str());
        break;
    }
    if (ret != SUCCESS) {
      GELOGE(ret, "[DealWith][Node] %s in multi-batch process failed", node->GetName().c_str());
      return ret;
    }
  }
  return SUCCESS;
}

NodePtr MultiBatchGraphCopyer::InsertMergeNode(const NodePtr &node, int index) {
  if (index < 0) {
    // the merge node must has data inputs, if origin connection is a control
    // edge, we use data edge instead
    index = 0;
  }

  auto &merge_nodes = nodes_to_merge_nodes_[node.get()];
  if (merge_nodes.empty()) {
    auto count = node->GetAllOutDataAnchorsSize();
    if (count == 0) {
      count = 1;
    }
    merge_nodes.resize(count, nullptr);
  }

  if (merge_nodes.at(index) != nullptr) {
    return merge_nodes[index];
  }

  auto merge_node_name = node->GetName() + "_ascend_mbatch_merge_" + std::to_string(index);
  auto merge_node = InsertMergeNodeToGraph(merge_node_name, shapes_.size(), node->GetOwnerComputeGraph());
  GE_IF_BOOL_EXEC(merge_node == nullptr,
                  GELOGE(INTERNAL_ERROR, "[Create][MergeNode] for node %s failed, out index %d",
                         node->GetName().c_str(), index);
                  return nullptr);
  merge_nodes[index] = merge_node;
  GELOGI("Create merge node %s for node %s index %d", merge_node_name.c_str(), node->GetName().c_str(), index);
  return merge_node;
}

NodePtr MultiBatchGraphCopyer::FindSwitchnNodeForDataEdge(const OutDataAnchorPtr &data_out_anchor,
                                                          const NodePtr &origin_node) {
  auto data_node = data_out_anchor->GetOwnerNode();
  GELOGD("Start find switchn node insert between %s and %s", data_node->GetName().c_str(),
         origin_node->GetName().c_str());
  NodePtr switchn = nullptr;
  if (!getnext_sink_dynamic_dims_ && data_nodes_to_switchn_.count(data_node.get()) > 0) {
    switchn = data_nodes_to_switchn_[data_node.get()];
    return switchn;
  }
  bool is_getnext_sink_data = false;
  for (size_t i = 0; i < getnext_nodes_to_switchn_.size(); ++i) {
    for (size_t j = 0; j < getnext_nodes_to_switchn_.at(i).size(); ++j) {
      if (getnext_nodes_to_switchn_.at(i).at(j).first == data_node.get()) {
        is_getnext_sink_data = true;
        break;
      }
    }
  }
  // get output_idx of origin_node(getnext)
  if (is_getnext_sink_data) {
    auto output_idx = data_out_anchor->GetIdx();
    size_t referenced_index = 0;
    GELOGI("The output idx %d has %zu referenced nums.", output_idx, data_out_anchor->GetPeerInDataAnchors().size());
    for (const auto &peer_in_anchor : data_out_anchor->GetPeerInDataAnchors()) {
      if (peer_in_anchor->GetOwnerNode()->GetOpDesc() == nullptr) {
        REPORT_INNER_ERROR("E19999", "peer op_desc of op:%s(%s)'s out_index:%d anchor exist nullptr, "
                           "check invalid",
                           data_node->GetName().c_str(), data_node->GetType().c_str(), output_idx);
        GELOGE(INTERNAL_ERROR, "[Get][OpDesc] failed, peer op_desc of op:%s(%s)'s out_index:%d anchor exist nullptr",
               data_node->GetName().c_str(), data_node->GetType().c_str(), output_idx);
        return nullptr;
      }
      if (getnext_nodes_to_switchn_.at(output_idx).empty()) {
        GELOGI("Output idx %d of %s is static output.", output_idx, data_node->GetName().c_str());
        return nullptr;
      }
      if (output_idx >= static_cast<int>(getnext_nodes_to_switchn_.size()) ||
         referenced_index >= getnext_nodes_to_switchn_.at(output_idx).size()) {
        REPORT_INNER_ERROR("E19999", "output_index:%d of op:%s(%s) > getnext_nodes_to_switchn_.size():%zu or "
                           "referenced_index:%zu >= getnext_nodes_to_switchn_.at(output_idx).size():%zu, "
                           "check invalid", output_idx,
                           data_node->GetName().c_str(), data_node->GetType().c_str(), getnext_nodes_to_switchn_.size(),
                           referenced_index, getnext_nodes_to_switchn_.at(output_idx).size());
        GELOGE(INTERNAL_ERROR, "[Check][Param] output_index:%d of op:%s(%s) >= getnext_nodes_to_switchn_.size():%zu or "
               "referenced_index:%zu >= getnext_nodes_to_switchn_.at(output_idx).size():%zu", output_idx,
               data_node->GetName().c_str(), data_node->GetType().c_str(), getnext_nodes_to_switchn_.size(),
               referenced_index, getnext_nodes_to_switchn_.at(output_idx).size());
        return nullptr;
      }
      if (peer_in_anchor->GetOwnerNode()->GetOpDesc()->GetName() == origin_node->GetName()) {
        switchn = getnext_nodes_to_switchn_.at(output_idx).at(referenced_index).second;
        GELOGI("Name of switchn is %s.", switchn->GetName().c_str());
        return switchn;
      }
      referenced_index++;
    }
  }
  return switchn;
}

Status MultiBatchGraphCopyer::CopyInDataEdges(const NodePtr &origin_node, int batch_num, const NodePtr &copyed_node) {
  GELOGI("Start copy data edges for %s and %s.", origin_node->GetName().c_str(), copyed_node->GetName().c_str());
  for (auto &in_anchor : origin_node->GetAllInDataAnchors()) {
    auto origin_src_anchor = in_anchor->GetPeerOutAnchor();
    if (origin_src_anchor == nullptr) {
      GELOGD("The node %s does not have input on index %d", origin_node->GetName().c_str(), in_anchor->GetIdx());
      continue;
    }
    auto origin_src_node = origin_src_anchor->GetOwnerNode();
    auto dst_anchor = copyed_node->GetInDataAnchor(in_anchor->GetIdx());
    GE_CHECK_NOTNULL(dst_anchor);
    auto switchn = FindSwitchnNodeForDataEdge(origin_src_anchor, origin_node);
    if (switchn != nullptr) {
      auto ret = GraphUtils::AddEdge(switchn->GetOutDataAnchor(batch_num), dst_anchor);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(out_index:%d) and op:%s(%s)(in_index:%d) failed",
                          switchn->GetName().c_str(), switchn->GetType().c_str(),
                          batch_num, copyed_node->GetName().c_str(), copyed_node->GetType().c_str(),
                          in_anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Add][DataEdge] between %s(%d) and %s(%d) failed, error-code %u",
               switchn->GetName().c_str(), batch_num, copyed_node->GetName().c_str(), in_anchor->GetIdx(),
               ret);
        return INTERNAL_ERROR;
      }
      GELOGD("Add data edge from %s(%d) to %s(%d)", switchn->GetName().c_str(), batch_num,
             copyed_node->GetName().c_str(), in_anchor->GetIdx());
      continue;
    }

    auto batch_branch_iter = nodes_to_batch_nodes_.find(origin_src_node.get());
    if (batch_branch_iter != nodes_to_batch_nodes_.end()) {
      auto src_batch_node = batch_branch_iter->second.at(batch_num);
      auto ret = GraphUtils::AddEdge(src_batch_node->GetOutDataAnchor(origin_src_anchor->GetIdx()), dst_anchor);
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(out_index:%d) and op:%s(%s)(in_index:%d) failed",
                          src_batch_node->GetName().c_str(),
                          src_batch_node->GetType().c_str(), origin_src_anchor->GetIdx(),
                          copyed_node->GetName().c_str(), copyed_node->GetType().c_str(),
                          in_anchor->GetIdx());
        GELOGE(INTERNAL_ERROR, "[Add][DataEdge] between %s(%d) and %s(%d) failed, error-code %u",
               src_batch_node->GetName().c_str(), batch_num, copyed_node->GetName().c_str(), in_anchor->GetIdx(), ret);
        return INTERNAL_ERROR;
      }
      GELOGD("Add data edge from %s(%d) to %s(%d)", src_batch_node->GetName().c_str(), batch_num,
             copyed_node->GetName().c_str(), in_anchor->GetIdx());
      continue;
    }

    auto ret = GraphUtils::AddEdge(origin_src_anchor, dst_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(out_index:%d) and op:%s(%s)(in_index:%d) failed",
                        origin_src_node->GetName().c_str(),
                        origin_src_node->GetType().c_str(), origin_src_anchor->GetIdx(),
                        copyed_node->GetName().c_str(), copyed_node->GetType().c_str(),
                        in_anchor->GetIdx());
      GELOGE(INTERNAL_ERROR, "[Add][DataEdge] between origin node %s(%d) and copyed %s(%d) failed",
             origin_src_node->GetName().c_str(), origin_src_anchor->GetIdx(), copyed_node->GetName().c_str(),
             dst_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    GELOGD("Add data edge between branch-out %s(%d) to branch-in %s(%d)", origin_src_node->GetName().c_str(),
           origin_src_anchor->GetIdx(), copyed_node->GetName().c_str(), dst_anchor->GetIdx());
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::CopyInControlEdges(const NodePtr &node, int batch_num, const NodePtr &copyed_node) {
  GELOGI("Start copy control edge for %s and %s.", node->GetName().c_str(), copyed_node->GetName().c_str());
  for (auto &origin_src_node : node->GetInControlNodes()) {
    auto switchn_iter = data_nodes_to_switchn_.find(origin_src_node.get());
    if (switchn_iter != data_nodes_to_switchn_.end()) {
      // reconnect data node
      auto ret = GraphUtils::AddEdge(switchn_iter->second->GetOutControlAnchor(), copyed_node->GetInControlAnchor());
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add ctrl edge between op:%s(%s) and op:%s(%s) failed",
                          switchn_iter->second->GetName().c_str(), switchn_iter->second->GetType().c_str(),
                          copyed_node->GetName().c_str(), copyed_node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] between %s and %s failed, error-code %u",
               switchn_iter->second->GetName().c_str(), copyed_node->GetName().c_str(), ret);
        return INTERNAL_ERROR;
      }
      GELOGD("Add control edge from %s to %s", switchn_iter->second->GetName().c_str(), copyed_node->GetName().c_str());
      continue;
    }

    auto batch_branch_iter = nodes_to_batch_nodes_.find(origin_src_node.get());
    if (batch_branch_iter != nodes_to_batch_nodes_.end()) {
      // reconnect node in batch branch
      auto src_batch_node = batch_branch_iter->second.at(batch_num);
      auto ret = GraphUtils::AddEdge(src_batch_node->GetOutControlAnchor(), copyed_node->GetInControlAnchor());
      if (ret != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Add ctrl edge between op:%s(%s) and op:%s(%s) failed",
                          src_batch_node->GetName().c_str(), src_batch_node->GetType().c_str(),
                          copyed_node->GetName().c_str(), copyed_node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] between %s and %s failed, error-code %u",
               src_batch_node->GetName().c_str(), copyed_node->GetName().c_str(), ret);
        return INTERNAL_ERROR;
      }
      GELOGD("Add control edge from %s to %s", src_batch_node->GetName().c_str(), copyed_node->GetName().c_str());
      continue;
    }

    auto ret = GraphUtils::AddEdge(origin_src_node->GetOutControlAnchor(), copyed_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add ctrl edge between op:%s(%s) and op:%s(%s) failed",
                        origin_src_node->GetName().c_str(), origin_src_node->GetType().c_str(),
                        copyed_node->GetName().c_str(), copyed_node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] from origin %s to copyed %s failed",
             origin_src_node->GetName().c_str(), copyed_node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    GELOGD("Add control edge between branch-out %s to branch-in %s", origin_src_node->GetName().c_str(),
           copyed_node->GetName().c_str());
  }
  return SUCCESS;
}

NodePtr MultiBatchGraphCopyer::InsertShapeDataNode() {
  auto desc = MakeShared<OpDesc>();
  if (desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(OUT_OF_MEMORY, "[New][OpDesc] failed, out of memory");
    return nullptr;
  }
  string node_name = "ascend_mbatch_shape_data";
  // Only flush subgraph name
  if (graph_->GetParentGraph() != nullptr) {
    node_name = graph_->GetName() + "_" + node_name;
  }
  desc->SetName(node_name);
  desc->SetType(DATA);
  // input and output of DATA is gear_info
  GeTensorDesc tensor_desc(GeShape({static_cast<int64_t>(shapes_.at(0).size())}), FORMAT_ND, DT_INT64);
  auto ret = desc->AddInputDesc(tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed",
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }
  ret = desc->AddOutputDesc(tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add output desc into op:%s(%s) failed",
                      desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] into op:%s(%s) failed",
           desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }

  if (!AttrUtils::SetBool(desc, ATTR_INSERT_BY_MBATCH, true)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                      ATTR_INSERT_BY_MBATCH.c_str(), desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed",
           ATTR_INSERT_BY_MBATCH.c_str(), desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }

  auto data_node = graph_->AddNode(desc);
  if (data_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      desc->GetName().c_str(), desc->GetType().c_str(), graph_->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] %s(%s) to graph:%s failed",
           desc->GetName().c_str(), desc->GetType().c_str(), graph_->GetName().c_str());
    return nullptr;
  }
  ret = GraphUtils::AppendInputNode(graph_, data_node);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Append input node:%s(%s) to graph:%s failed",
                      data_node->GetName().c_str(), data_node->GetType().c_str(),
                      graph_->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Append][InputNode] %s to graph:%s failed",
           data_node->GetName().c_str(), graph_->GetName().c_str());
    return nullptr;
  }

  return data_node;
}

NodePtr MultiBatchGraphCopyer::InsertGetDynamicDimsNode() {
  GELOGD("Start insert getdynamicdims node to get shape info.");
  auto desc = MakeShared<OpDesc>();
  if (desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "New OpDesc failed");
    GELOGE(OUT_OF_MEMORY, "[New][OpDesc] failed, out of memory");
    return nullptr;
  }
  string node_name = "ascend_mbatch_get_dynamic_dims_node";

  // Only flush subgraph name
  if (graph_->GetParentGraph() != nullptr) {
    node_name = graph_->GetName() + "_" + node_name;
  }

  desc->SetName(node_name);
  desc->SetType(GETDYNAMICDIMS);

  // input of GetDynamicDims is shape_of_each_data, output is gear_info
  for (size_t i = 0; i < GetLocalOmgContext().user_input_dims.size(); ++i) {
    size_t input_shape_dims = GetLocalOmgContext().user_input_dims.at(i).second.size();
    if (input_shape_dims == 1 && GetLocalOmgContext().user_input_dims.at(i).second.at(0) == 0) {
      GeTensorDesc tensor_desc;
      tensor_desc.SetFormat(FORMAT_ND);
      tensor_desc.SetDataType(DT_INT64);
      auto ret = desc->AddInputDesc(tensor_desc);
      GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                      REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                        desc->GetName().c_str(), desc->GetType().c_str());
                      GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed",
                             desc->GetName().c_str(), desc->GetType().c_str());
                      return nullptr);
      continue;
    }
    GeTensorDesc tensor_desc(GeShape({static_cast<int64_t>(input_shape_dims)}), FORMAT_ND, DT_INT64);
    auto ret = desc->AddInputDesc(tensor_desc);
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed",
                                      desc->GetName().c_str(), desc->GetType().c_str());
                    GELOGE(INTERNAL_ERROR, "[Add][InputDesc] to op:%s(%s) failed",
                           desc->GetName().c_str(), desc->GetType().c_str());
                    return nullptr);
  }

  GeTensorDesc tensor_desc(GeShape({static_cast<int64_t>(shapes_.at(0).size())}), FORMAT_ND, DT_INT64);
  auto ret = desc->AddOutputDesc(tensor_desc);
  GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed",
                                    desc->GetName().c_str(), desc->GetType().c_str());
                  GELOGE(INTERNAL_ERROR, "[Add][OutputDesc] to op:%s(%s) failed",
                         desc->GetName().c_str(), desc->GetType().c_str());
                  return nullptr);

  if (!AttrUtils::SetBool(desc, ATTR_INSERT_BY_MBATCH, true)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                      ATTR_INSERT_BY_MBATCH.c_str(), desc->GetName().c_str(), desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed",
           ATTR_INSERT_BY_MBATCH.c_str(), desc->GetName().c_str(), desc->GetType().c_str());
    return nullptr;
  }

  auto data_node = graph_->AddNode(desc);
  if (data_node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                      desc->GetName().c_str(), desc->GetType().c_str(), graph_->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Add][Node] %s(%s) to graph:%s failed",
           desc->GetName().c_str(), desc->GetType().c_str(), graph_->GetName().c_str());
    return nullptr;
  }
  ret = GraphUtils::AppendInputNode(graph_, data_node);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Append input node:%s(%s) to graph:%s failed",
                      data_node->GetName().c_str(), data_node->GetType().c_str(),
                      graph_->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Append][InputNode] %s(%s) to graph:%s failed",
           data_node->GetName().c_str(), data_node->GetType().c_str(), graph_->GetName().c_str());
    return nullptr;
  }

  return data_node;
}

Status MultiBatchGraphCopyer::CheckArguments() {
  if (graph_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "graph_ is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] graph_ is nullptr");
    return PARAM_INVALID;
  }

  return CheckDynamicParams(shapes_);
}

Status MultiBatchGraphCopyer::CheckCopyResult(const std::vector<NodePtr> &start_nodes) {
  for (auto &node : start_nodes) {
    if (IsOnlyOutputToAipp(node)) {
      continue;
    }
    auto dims = NodeUtils::GetOutputDesc(*node, kDataOutIndex).GetShape().GetDims();
    if (!IsAllDimsPositive(dims)) {
      REPORT_CALL_ERROR("E19999", "Failed to copy multi batch graph, the node %s still has unknown shape %s",
                        node->GetName().c_str(), formats::ShapeToString(dims).c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to copy multi batch graph, the node %s still has unknown shape %s",
             node->GetName().c_str(), formats::ShapeToString(dims).c_str());
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

bool MultiBatchGraphCopyer::IsInBatchBranch(const NodePtr &node) {
  if (!getnext_sink_dynamic_dims_) {
    return (nodes_to_batch_nodes_.count(node.get()) > 0) || (data_nodes_to_switchn_.count(node.get()) > 0);
  } else {
    for (size_t i = 0; i < getnext_nodes_to_switchn_.size(); ++i) {
      for (size_t j = 0; j < getnext_nodes_to_switchn_.at(i).size(); ++j) {
        if (getnext_nodes_to_switchn_.at(i).at(j).first == node.get()) {
          return true;
        }
      }
    }
    return nodes_to_batch_nodes_.count(node.get()) > 0;
  }
}

Status MultiBatchGraphCopyer::LinkDataToMerge(const NodePtr &data, const NodePtr &merge, const NodePtr &switchn) {
  // The caller should make sure that the there is a SwitchN node in the map
  GELOGI("Link edge between data %s to merge %s throw switchn %s", data->GetName().c_str(), merge->GetName().c_str(),
         switchn->GetName().c_str());
  for (size_t i = 0; i < shapes_.size(); ++i) {
    auto ret = GraphUtils::AddEdge(switchn->GetOutDataAnchor(i), merge->GetInDataAnchor(i));
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%zu) and op:%s(%s)(index:%zu) failed",
                                      switchn->GetName().c_str(), switchn->GetType().c_str(), i,
                                      merge->GetName().c_str(), merge->GetType().c_str(), i);
                    GELOGE(INTERNAL_ERROR, "[Add][Edge] between switchn %s(%zu) and merge %s(%zu) failed, ret:%u",
                           switchn->GetName().c_str(), i, merge->GetName().c_str(), i, ret);
                    return INTERNAL_ERROR);
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::LinkNodeToMerge(const NodePtr &node, int out_index, const NodePtr &merge) {
  auto &copyed_nodes = nodes_to_batch_nodes_[node.get()];
  if (copyed_nodes.size() != shapes_.size()) {
    REPORT_INNER_ERROR("E19999", "Create merge node for node %s failed, "
                       "the copyed nodes for it count %zu different with shape %zu, check invalid",
                       node->GetName().c_str(), copyed_nodes.size(), shapes_.size());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to create merge node for node %s, "
           "the copyed nodes for it count %zu different with shape %zu",
           node->GetName().c_str(), copyed_nodes.size(), shapes_.size());
    return INTERNAL_ERROR;
  }
  for (size_t i = 0; i < copyed_nodes.size(); ++i) {
    auto src_node = copyed_nodes[i];
    if (src_node->GetAllOutDataAnchorsSize() == 0) {
      // if the node does not has any data output, we should create an const for it, like this:
      //       c          d
      // node ---> const ---> merge
      auto const_name = src_node->GetName() + "_merge_const";
      GELOGI("The node %s on the batch branch edge does not have any data output, create a const %s for it",
             src_node->GetName().c_str(), const_name.c_str());
      auto const_node = InsertConst(const_name, graph_);
      GE_IF_BOOL_EXEC(const_node == nullptr,
                      GELOGE(OUT_OF_MEMORY, "[Create][Const] for node:%s failed, which to connect to a merge node",
                             src_node->GetName().c_str());
                      return OUT_OF_MEMORY);

      auto ret = GraphUtils::AddEdge(src_node->GetOutControlAnchor(), const_node->GetInControlAnchor());
      GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                      REPORT_CALL_ERROR("E19999", "Add ctrl edge between op:%s(%s) and op:%s(%s) failed",
                                        src_node->GetName().c_str(), src_node->GetType().c_str(),
                                        const_node->GetName().c_str(), const_node->GetType().c_str());
                      GELOGE(INTERNAL_ERROR, "[Add][ControlEdge] from %s to %s failed",
                             src_node->GetName().c_str(), const_node->GetName().c_str());
                      return INTERNAL_ERROR);

      src_node = const_node;
    }
    auto ret = GraphUtils::AddEdge(src_node->GetOutDataAnchor(out_index), merge->GetInDataAnchor(i));
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%zu) failed",
                        src_node->GetName().c_str(), src_node->GetType().c_str(), out_index,
                        merge->GetName().c_str(), merge->GetType().c_str(), i);
      GELOGE(INTERNAL_ERROR,
             "[Add][Edge] between copyed node %s(%d) and inserted merge node %s(%zu) failed, error-code %u",
             copyed_nodes[i]->GetName().c_str(), out_index, merge->GetName().c_str(), i, ret);
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::InsertSwitchNAndUpdateMaxShape(const NodePtr &node) {
  std::vector<std::pair<Node *, NodePtr>> dynamic_out_to_switchn;
  if (!getnext_sink_dynamic_dims_) {
    if (InsertSwitchNForData(node, kDataOutIndex, kDataOutIndex, dynamic_out_to_switchn) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Insert][SwitchN] for node:%s failed.", node->GetName().c_str());
      return PARAM_INVALID;
    }
    if (UpdateMaxShapeToData(node, kDataOutIndex) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Update][MaxShape] of node:%s failed.", node->GetName().c_str());
      return PARAM_INVALID;
    }
  } else {
    if (!IsGetNextType(node)) {
      GELOGI("No need to insert switchn and update max shape for %s when get sink dynamic.", node->GetName().c_str());
      return SUCCESS;
    }
    for (size_t i = 0; i < getnext_sink_dynamic_out_mapping_.size(); ++i) {
      dynamic_out_to_switchn.clear();
      for (size_t j = 0; j < getnext_sink_dynamic_out_mapping_.at(i).second; ++j) {
        GELOGI("The %zu data_index has %zu referenced nums.", getnext_sink_dynamic_out_mapping_.at(i).first,
               getnext_sink_dynamic_out_mapping_.at(i).second);
        if (InsertSwitchNForData(node, getnext_sink_dynamic_out_mapping_.at(i).first, j, dynamic_out_to_switchn) !=
            SUCCESS) {
          GELOGE(PARAM_INVALID, "[Insert][SwitchN] for %s of %zu out anchor failed, when referenced index is %zu",
                 node->GetName().c_str(), getnext_sink_dynamic_out_mapping_.at(i).first, j);
          return PARAM_INVALID;
        }
      }
      getnext_nodes_to_switchn_.emplace_back(dynamic_out_to_switchn);
    }

    for (size_t i = 0; i < getnext_sink_dynamic_out_mapping_.size(); ++i) {
      if(UpdateMaxShapeToData(node, i) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Call][UpdateMaxShapeToData]Failed to update %s max shape of %zu out anchor",
               node->GetName().c_str(), i);
        return PARAM_INVALID;
      }
    }
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::UpdateShapeOfShapeNode(const NodePtr &node, size_t out_anchor_index) {
  auto data_shape = NodeUtils::GetOutputDesc(*node, out_anchor_index).GetShape();
  size_t shape_index = out_anchor_index + (node->GetAllOutDataAnchors().size() / kDivisionConst);
  GeTensorDesc output_desc = node->GetOpDesc()->GetOutputDesc(shape_index);
  std::vector<int64_t> output_dims = {static_cast<int64_t>(data_shape.GetDims().size())};
  GeShape output_shape(output_dims);
  output_desc.SetShape(output_shape);
  if (node->GetOpDesc()->UpdateOutputDesc(shape_index, output_desc) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Update output desc to op:%s(%s) failed, index:%zu",
                      node->GetName().c_str(), node->GetType().c_str(), shape_index);
    GELOGE(FAILED, "[Update][OutputDesc] to op:%s(%s) failed, index:%zu",
           node->GetName().c_str(), node->GetType().c_str(), shape_index);
    return FAILED;
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::UpdateMaxShapeToData(const NodePtr &node, size_t out_anchor_index) {
  GELOGD("Start update max shape of %s, %zu output.", node->GetName().c_str(), out_anchor_index);
  auto data_shape = NodeUtils::GetOutputDesc(*node, out_anchor_index).GetShape();
  string data_name = node->GetName();
  if (getnext_sink_dynamic_dims_) {
    data_name.append("_").append(std::to_string(out_anchor_index));
  }
  GELOGD("Update max shape of %s, shape dims is %s.", data_name.c_str(),
         formats::JoinToString(data_shape.GetDims()).c_str());
  if (!getnext_sink_dynamic_dims_) {
    if (IsAllDimsPositive(data_shape.GetDims())) {
      GELOGD("No need to do anything for static data.");
      return SUCCESS;
    }
  } else {
    if (IsAllDimsPositive(data_shape.GetDims())) {
      // need to update shape of Shape_node
      GE_CHK_STATUS_RET(UpdateShapeOfShapeNode(node, out_anchor_index),
                        "[Update][ShapeOfShapeNode] %s failed, index:%zu", node->GetName().c_str(), out_anchor_index);
      return SUCCESS;
    }
  }

  size_t max_shape_index = 0;
  int64_t max_size = 0;
  for (size_t i = 0; i < shapes_.size(); ++i) {
    int64_t size = 1;
    for (auto dim : data_to_dynamic_info_.at(data_name).at(i)) {
      if (INT64_MAX / dim < size) {
        REPORT_CALL_ERROR("E19999", "Op:%s(%s)'s shape:%s size will overflow after multi, check invalid",
                          node->GetName().c_str(), node->GetType().c_str(),
                          formats::ShapeToString(data_to_dynamic_info_[data_name].at(i)).c_str());
        GELOGE(PARAM_INVALID, "[Check][Param] Op:%s(%s)'s shape:%s size will overflow after multi",
               node->GetName().c_str(), node->GetType().c_str(),
               formats::ShapeToString(data_to_dynamic_info_[data_name].at(i)).c_str());
        return PARAM_INVALID;
      }
      size *= dim;
    }
    if (size > max_size) {
      max_size = size;
      max_shape_index = i;
    }
  }
  // must not be error, the calc result has been checked in function InsertSwitchNForData
  (void)CalcShape(data_to_dynamic_info_.at(data_name).at(max_shape_index), data_shape);
  auto ret = NodeUtils::UpdateOutputShape(*node, out_anchor_index, data_shape);
  GE_CHK_GRAPH_STATUS_RET(ret, "[Update][OutputShape] for data %s failed", node->GetName().c_str());
  // getnext_sink not has input
  if (!getnext_sink_dynamic_dims_) {
    ret = NodeUtils::UpdateInputShape(*node, kDataInIndex, data_shape);
    GE_CHK_GRAPH_STATUS_RET(ret, "[Update][InputShape] for data %s failed", node->GetName().c_str());
  } else {
    // need to update shape of Shape_node when getnext_sink_dynamic
    GE_CHK_STATUS_RET(UpdateShapeOfShapeNode(node, out_anchor_index),
                      "[Update][ShapeOfShapeNode] %s failed, index:%zu", node->GetName().c_str(), out_anchor_index);
  }
  GELOGI("Update the data %s input/output shape to the max %s", node->GetName().c_str(),
         formats::ShapeToString(data_shape).c_str());
  return SUCCESS;
}

Status MultiBatchGraphCopyer::InsertSwitchNForData(const NodePtr &node, const size_t &out_anchor_index,
                                                   const size_t &peer_in_anchor_index,
                                                   std::vector<std::pair<Node *, NodePtr>> &dynamic_out_to_switchn) {
  auto data_shape = NodeUtils::GetOutputDesc(*node, out_anchor_index).GetShape();
  string data_name = node->GetName();
  if (getnext_sink_dynamic_dims_) {
    data_name.append("_").append(std::to_string(out_anchor_index));
  }
  (void)AttrUtils::SetListInt(node->GetOpDesc(), ATTR_MBATCH_ORIGIN_INPUT_DIMS, data_shape.GetDims());
  GELOGI("Insert switchn node of %s, shape dims is %s.", data_name.c_str(),
         formats::JoinToString(data_shape.GetDims()).c_str());
  if (IsAllDimsPositive(data_shape.GetDims())) {
    GELOGI("The shape of data %s are positive(%s), skip the multi batch process", node->GetName().c_str(),
           data_shape.ToString().c_str());
    return SUCCESS;
  }

  auto switchn_desc = MakeShared<OpDesc>();
  GE_IF_BOOL_EXEC(switchn_desc == nullptr,
                  REPORT_CALL_ERROR("E19999", "New OpDesc failed");
                  GELOGE(OUT_OF_MEMORY, "[New][OpDesc] failed");
                  return OUT_OF_MEMORY);
  string switchn_name = node->GetName() + "_ascend_mbatch_switchn";
  if (getnext_sink_dynamic_dims_) {
    switchn_name.append("_").append(std::to_string(out_anchor_index))
                .append("_").append(std::to_string(peer_in_anchor_index));
  }
  GELOGI("name of switchn is %s.", switchn_name.c_str());
  switchn_desc->SetName(switchn_name);
  switchn_desc->SetType(SWITCHN);

  GeTensorDesc tensor(NodeUtils::GetOutputDesc(*node, out_anchor_index));
  GE_IF_BOOL_EXEC(switchn_desc->AddInputDesc("data", tensor) != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed, input desc name:%s",
                                    switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str(),
                                    "data");
                  GELOGE(OUT_OF_MEMORY, "[Add][InputDesc] to op:%s(%s) failed, input desc name:data",
                         switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str());
                  return OUT_OF_MEMORY);
  GeTensorDesc pred_tensor;
  GE_IF_BOOL_EXEC(switchn_desc->AddInputDesc("pred_value", pred_tensor) != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Add input desc to op:%s(%s) failed, input desc name:%s",
                                    switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str(),
                                    "pred_value");
                  GELOGE(OUT_OF_MEMORY, "[Add][InputDesc] to op:%s(%s) failed, input desc name:pred_value",
                         switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str());
                  return OUT_OF_MEMORY);
  std::vector<std::string> input_dims_str;
  for (size_t i = 0; i < shapes_.size(); ++i) {
    GELOGI("Start clac shape for data %s, batch shape is %s.", data_name.c_str(),
           formats::JoinToString(data_to_dynamic_info_.at(data_name).at(i)).c_str());
    auto shape = data_shape;
    auto ret = CalcShape(data_to_dynamic_info_.at(data_name).at(i), shape);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Calc][Shape] Failed to calculate the batched shape for data node %s, the shapes may not match",
             node->GetName().c_str());
      return ret;
    }
    tensor.SetShape(shape);
    string input_str;
    int64_t tensor_size = 0;
    (void)TensorUtils::GetTensorSizeInBytes(tensor, tensor_size);
    input_str = TypeUtils::FormatToSerialString(tensor.GetFormat()) + ":" +
                TypeUtils::DataTypeToSerialString(tensor.GetDataType()) + ":" + node->GetName() + ":" +
                std::to_string(tensor_size) + ":" + std::to_string(tensor.GetShape().GetDimNum()) + ":" +
                formats::JoinToString(tensor.GetShape().GetDims());
    input_dims_str.emplace_back(input_str);
    if (!AttrUtils::SetListInt(tensor, ATTR_NAME_SWITCHN_PRED_VALUE, shapes_.at(i))) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to output tensor of node:%s(%s) failed, index:%zu",
                        ATTR_NAME_SWITCHN_PRED_VALUE.c_str(),
                        node->GetName().c_str(), node->GetType().c_str(), out_anchor_index);
      GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to output tensor of node:%s(%s) failed, index:%zu",
             ATTR_NAME_SWITCHN_PRED_VALUE.c_str(), node->GetName().c_str(), node->GetType().c_str(), out_anchor_index);
      return INTERNAL_ERROR;
    }
    (void) AttrUtils::SetListInt(tensor, ATTR_NAME_COMBINED_DYNAMIC_DIMS, shape.GetDims());
    if (switchn_desc->AddOutputDesc("output" + std::to_string(i), tensor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add output desc to op:%s(%s) failed, output desc name:%s",
                        switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str(),
                        ("output" + std::to_string(i)).c_str());
      GELOGE(GRAPH_FAILED, "[Add][OutputDesc] to op:%s(%s) failed, output desc name:%s",
             switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str(),
             ("output" + std::to_string(i)).c_str());
      return GRAPH_FAILED;
    }
    GELOGD("The switchn %s output index %zu, shape %s", switchn_desc->GetName().c_str(), i, shape.ToString().c_str());
  }
  (void)AttrUtils::SetListStr(node->GetOpDesc(), "_all_origin_gears_inputs", input_dims_str);
  if (!AttrUtils::SetListStr(switchn_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, data_name_order_)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                      ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str(),
                      switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed", ATTR_USER_DESIGNEATE_SHAPE_ORDER.c_str(),
           switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (!AttrUtils::SetBool(switchn_desc, ATTR_INSERT_BY_MBATCH, true)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                      ATTR_INSERT_BY_MBATCH.c_str(), switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed",
           ATTR_INSERT_BY_MBATCH.c_str(), switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (!AttrUtils::SetStr(node->GetOpDesc(), kMbatchSwitchnName, switchn_desc->GetName())) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                      kMbatchSwitchnName, node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed",
           kMbatchSwitchnName, node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (StampDynamicType(switchn_desc) != SUCCESS) {
    GELOGE(INTERNAL_ERROR, "Failed to add dynamic type attr on switchn node %s", switchn_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }

  auto switchn = graph_->AddNode(switchn_desc);
  GE_IF_BOOL_EXEC(switchn == nullptr,
                  REPORT_CALL_ERROR("E19999", "Add node:%s(%s) to graph:%s failed",
                                    switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str(),
                                    graph_->GetName().c_str());
                  GELOGE(OUT_OF_MEMORY, "[Add][Node] %s(%s) to graph:%s failed",
                         switchn_desc->GetName().c_str(), switchn_desc->GetType().c_str(), graph_->GetName().c_str());
                  return OUT_OF_MEMORY);
  if (!getnext_sink_dynamic_dims_) {
    data_nodes_to_switchn_[node.get()] = switchn;
  } else {
    dynamic_out_to_switchn.emplace_back(std::make_pair(node.get(), switchn));
    GELOGD("Insert %s for %s.", switchn->GetName().c_str(), node->GetName().c_str());
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::InsertMergeForEdgeNode(const NodePtr &node) {
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto src_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr) {
      GELOGD("The node %s does not has input at index %d", node->GetName().c_str(), in_data_anchor->GetIdx());
      continue;
    }
    auto in_node = src_out_anchor->GetOwnerNode();
    if (!IsInBatchBranch(in_node)) {
      continue;
    }
    auto merge_node = InsertMergeNode(in_node, src_out_anchor->GetIdx());
    if (merge_node == nullptr) {
      return INTERNAL_ERROR;
    }
  }

  for (auto &in_node : node->GetInControlNodes()) {
    if (!IsInBatchBranch(in_node)) {
      continue;
    }
    auto merge_node = InsertMergeNode(in_node, -1);
    if (merge_node == nullptr) {
      return INTERNAL_ERROR;
    }
  }

  return SUCCESS;
}

Status MultiBatchGraphCopyer::LinkGetDynamicDimsToNetOutput(const NodePtr &node) {
  if (node->GetType() == NETOUTPUT) {
    if (!GetLocalOmgContext().dynamic_node_type.empty()) {
      if (!AttrUtils::SetStr(node->GetOpDesc(), ATTR_ALL_GEARS_INFO, GetLocalOmgContext().dynamic_dims)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                          ATTR_ALL_GEARS_INFO.c_str(), node->GetName().c_str(), node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed",
               ATTR_ALL_GEARS_INFO.c_str(), node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
    if (getnext_sink_dynamic_dims_) {
      size_t input_index = node->GetAllInDataAnchors().size();
      if (NodeUtils::AppendInputAnchor(node, input_index + 1) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Append %zu input anchors to node:%s(%s) failed",
                          input_index + 1, node->GetName().c_str(), node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Append][InputAnchor] of node:%s failed, input_index:%zu.",
               node->GetName().c_str(), input_index + 1);
        return INTERNAL_ERROR;
      }
      auto ret =
          ge::GraphUtils::AddEdge(shape_data_->GetOutDataAnchor(kDataOutIndex), node->GetInDataAnchor(input_index));
      GE_IF_BOOL_EXEC(
          ret != GRAPH_SUCCESS,
          REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%zu) failed",
                            shape_data_->GetName().c_str(), shape_data_->GetType().c_str(), kDataOutIndex,
                            node->GetName().c_str(), node->GetType().c_str(), input_index);
          GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%zu) failed",
                 shape_data_->GetName().c_str(), shape_data_->GetType().c_str(), kDataOutIndex,
                 node->GetName().c_str(), node->GetType().c_str(), input_index);
          return INTERNAL_ERROR);
      if (!AttrUtils::SetBool(node->GetOpDesc(), ATTR_GETNEXT_SINK_DYNMAIC, true)) {
        REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                          ATTR_GETNEXT_SINK_DYNMAIC.c_str(), node->GetName().c_str(), node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed",
               ATTR_GETNEXT_SINK_DYNMAIC.c_str(), node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR;
      }
    }
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::CopyNodeInBatchBranch(const NodePtr &node) {
  auto &copyed_nodes = nodes_to_batch_nodes_[node.get()];
  for (size_t i = 0; i < shapes_.size(); ++i) {
    auto copyed_node = InsertCopyNode(node, i);
    if (copyed_node == nullptr) {
      GELOGE(INTERNAL_ERROR, "[Add][Node] to graph failed, when copy node %s", node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    copyed_nodes.emplace_back(copyed_node);
    GELOGI("Copy node %s type %s for shape %s, new node name %s", node->GetName().c_str(), node->GetType().c_str(),
           formats::JoinToString(shapes_.at(i)).c_str(), copyed_node->GetName().c_str());
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::AddAttrForGetDynamicDims(const NodePtr &node) {
  GELOGD("Add attr for :%s, type is %s:", shape_data_->GetName().c_str(), shape_data_->GetType().c_str());
  size_t data_count = node->GetAllOutDataAnchors().size() / kDivisionConst;
  if (!AttrUtils::SetInt(shape_data_->GetOpDesc(), ATTR_GETNEXT_SINK_DATA_COUNT, data_count)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                      ATTR_GETNEXT_SINK_DATA_COUNT.c_str(),
                      shape_data_->GetName().c_str(), shape_data_->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed", ATTR_GETNEXT_SINK_DATA_COUNT.c_str(),
           shape_data_->GetName().c_str(), shape_data_->GetType().c_str());
    return INTERNAL_ERROR;
  }
  vector<int64_t> shape_info;
  for (size_t i = 0; i < GetLocalOmgContext().user_input_dims.size(); ++i) {
    if (GetLocalOmgContext().user_input_dims.at(i).second.size() == 1 &&
        GetLocalOmgContext().user_input_dims.at(i).second.at(0) == 0) {
      shape_info.emplace_back(0);
      continue;
    }
    shape_info.emplace_back(GetLocalOmgContext().user_input_dims.at(i).second.size());
    for (size_t j = 0; j < GetLocalOmgContext().user_input_dims.at(i).second.size(); ++j) {
      shape_info.emplace_back(GetLocalOmgContext().user_input_dims.at(i).second.at(j));
    }
  }
  if (!AttrUtils::SetListInt(shape_data_->GetOpDesc(), ATTR_GETNEXT_SINK_SHAPE_INFO, shape_info)) {
    REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                      ATTR_GETNEXT_SINK_SHAPE_INFO.c_str(),
                      shape_data_->GetName().c_str(), shape_data_->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Set][Attr] %s to node:%s(%s) failed", ATTR_GETNEXT_SINK_SHAPE_INFO.c_str(),
           shape_data_->GetName().c_str(), shape_data_->GetType().c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::AddLinkForGetDynamicDims(const NodePtr &node) {
  GELOGD("Start relink out anchor from shape node to getdynamicdims, and delete link between shape node and identity.");
  size_t input_index = 0;
  GELOGD("Out count of %s is %zu.", node->GetName().c_str(), node->GetAllOutDataAnchors().size());
  size_t data_count = node->GetAllOutDataAnchors().size() / kDivisionConst;
  for (size_t out_index = data_count; out_index < node->GetAllOutDataAnchors().size(); ++out_index, ++input_index) {
    GELOGI("Start add %s of %zu out_anchor to %s of %zu in_anchor.", node->GetName().c_str(), out_index,
        shape_data_->GetName().c_str(), input_index);
    auto out_data_anchor =  node->GetOutDataAnchor(out_index);
    auto ret = GraphUtils::AddEdge(out_data_anchor, shape_data_->GetInDataAnchor(input_index));
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%zu) and op:%s(%s)(index:%zu) failed",
                                      node->GetName().c_str(), node->GetType().c_str(), out_index,
                                      shape_data_->GetName().c_str(), shape_data_->GetType().c_str(), input_index);
                    GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%zu) and op:%s(%s)(index:%zu) failed",
                           node->GetName().c_str(), node->GetType().c_str(), out_index,
                           shape_data_->GetName().c_str(), shape_data_->GetType().c_str(), input_index);
                    return INTERNAL_ERROR);
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::LinkEdges() {
  Status ret;
  for (const auto &node : origin_all_nodes_) {
    GE_CHECK_NOTNULL(node->GetOpDesc());
    if (!getnext_sink_dynamic_dims_) {
      if (data_nodes_to_switchn_.count(node.get()) > 0) {
        auto switchn = data_nodes_to_switchn_[node.get()];
        GE_IF_BOOL_EXEC(switchn == nullptr,
                        REPORT_INNER_ERROR("E19999",
                                           "swithn in data_nodes_to_switchn_ for op:%s(%s) is nullptr, check invalid",
                                           node->GetName().c_str(), node->GetType().c_str());
                        GELOGE(PARAM_INVALID, "[Check][Param]Switchn should not be nullptr for %s.",
                               node->GetName().c_str());
                        return OUT_OF_MEMORY);
        ret = LinkDataToSwitchN(node, switchn, kDataOutIndex);
        GE_CHK_STATUS_RET(ret, "Link data to switchn failed.");
      }
    } else {
      if (IsGetNextType(node)) {
        GELOGD("Start add attr and link edge for %s.", node->GetName().c_str());
        GE_CHK_STATUS_RET(AddAttrForGetDynamicDims(node), "[Add][Attr] for %s failed.", node->GetName().c_str());
        GE_CHK_STATUS_RET(AddLinkForGetDynamicDims(node), "[Add][Link] for %s failed.", node->GetName().c_str());
      }
      for (size_t i = 0; i < getnext_nodes_to_switchn_.size(); ++i) {
        for (size_t j = 0; j < getnext_nodes_to_switchn_.at(i).size(); ++j) {
          if (getnext_nodes_to_switchn_.at(i).at(j).first == node.get()) {
            auto switchn = getnext_nodes_to_switchn_.at(i).at(j).second;
            GE_CHK_STATUS_RET(LinkDataToSwitchN(node, switchn, i),
                              "[Link][Data] %s to %s failed.", node->GetName().c_str(), switchn->GetName().c_str());
          }
        }
      }
    }
    if (nodes_to_merge_nodes_.count(node.get()) > 0) {
      GE_CHK_STATUS_RET(LinkToMerge(node), "[Link][Node] %s to merge failed.", node->GetName().c_str());
    }
    if (nodes_to_batch_nodes_.count(node.get()) > 0) {
      ret = LinkToNodeInBranch(node);
    } else {
      ret = LinkToNodeOutBranch(node);
    }
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::LinkDataToSwitchN(const NodePtr &data, const NodePtr &switchn, const int &out_index) {
  auto ret =
      GraphUtils::AddEdge(shape_data_->GetOutDataAnchor(kDataOutIndex), switchn->GetInDataAnchor(kSwitchNPredIndex));
  GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                    shape_data_->GetName().c_str(), shape_data_->GetType().c_str(), kDataOutIndex,
                                    switchn->GetName().c_str(), switchn->GetType().c_str(), kSwitchNPredIndex);
                  GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                         shape_data_->GetName().c_str(), shape_data_->GetType().c_str(), kDataOutIndex,
                         switchn->GetName().c_str(), switchn->GetType().c_str(), kSwitchNPredIndex);
                  return INTERNAL_ERROR);

  ret = GraphUtils::AddEdge(data->GetOutDataAnchor(out_index), switchn->GetInDataAnchor(kSwitchNDataIndex));
  GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                  REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                                    data->GetName().c_str(), data->GetType().c_str(), out_index,
                                    switchn->GetName().c_str(), switchn->GetType().c_str(), kSwitchNDataIndex);
                  GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                         data->GetName().c_str(), data->GetType().c_str(), out_index,
                         switchn->GetName().c_str(), switchn->GetType().c_str(), kSwitchNDataIndex);
                  return INTERNAL_ERROR);
  return SUCCESS;
}

Status MultiBatchGraphCopyer::LinkToMerge(const NodePtr &node) {
  auto &merge_nodes = nodes_to_merge_nodes_[node.get()];
  for (size_t i = 0; i < merge_nodes.size(); ++i) {
    auto merge_node = merge_nodes[i];
    if (merge_node == nullptr) {
      continue;
    }
    if (nodes_to_batch_nodes_.count(node.get()) > 0) {
      auto ret = LinkNodeToMerge(node, i, merge_node);
      if (ret != SUCCESS) {
        return ret;
      }
      continue;
    }

    if (!getnext_sink_dynamic_dims_) {
      if (data_nodes_to_switchn_.count(node.get()) > 0) {
        auto &switchn = data_nodes_to_switchn_[node.get()];
        auto ret = LinkDataToMerge(node, merge_node, switchn);
        if (ret != SUCCESS) {
          return ret;
        }
        continue;
      }
    } else {
      for (size_t j = 0; j < getnext_nodes_to_switchn_.size(); ++j) {
        for (size_t k = 0; k < getnext_nodes_to_switchn_.at(j).size(); ++k) {
          if (getnext_nodes_to_switchn_.at(j).at(k).first == node.get()) {
            auto &switchn = getnext_nodes_to_switchn_.at(j).at(k).second;
            auto ret = LinkDataToMerge(node, merge_node, switchn);
            if (ret != SUCCESS) {
              return ret;
            }
          }
        }
      }
      continue;
    }
    REPORT_INNER_ERROR("E19999", "The merge node %s is created, index %zu, but can not find the src node, "
                       "check invalid", merge_node->GetName().c_str(), i);
    GELOGE(INTERNAL_ERROR, "[Check][Param] The merge node %s is created, index %zu, but can not find the src node",
           merge_node->GetName().c_str(), i);
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::LinkToNodeInBranch(const NodePtr &node) {
  GELOGI("Start LinkToNodeInBranch for %s.", node->GetName().c_str());
  auto &branch_nodes = nodes_to_batch_nodes_[node.get()];
  for (size_t i = 0; i < branch_nodes.size(); ++i) {
    auto ret = CopyInDataEdges(node, i, branch_nodes[i]);
    if (ret != SUCCESS) {
      return ret;
    }
    ret = CopyInControlEdges(node, i, branch_nodes[i]);
    if (ret != SUCCESS) {
      return ret;
    }
  }
  return SUCCESS;
}

Status MultiBatchGraphCopyer::LinkToNodeOutBranch(const NodePtr &node) {
  for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto src_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (src_out_anchor == nullptr) {
      GELOGD("The node %s does not has input at index %d", node->GetName().c_str(), in_data_anchor->GetIdx());
      continue;
    }
    auto in_node = src_out_anchor->GetOwnerNode();
    if (!IsInBatchBranch(in_node)) {
      continue;
    }
    auto iter = nodes_to_merge_nodes_.find(in_node.get());
    if (iter == nodes_to_merge_nodes_.end()) {
      REPORT_INNER_ERROR("E19999", "Failed to link data edge from %s(%s)(index:%d) to %s(%s)(index:%d), "
                         "cause no merge node found, check invalid",
                         in_node->GetName().c_str(), in_node->GetType().c_str(), src_out_anchor->GetIdx(),
                         node->GetName().c_str(), node->GetType().c_str(), in_data_anchor->GetIdx());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to link IO data edge from %s(%d) to %s(%d), no merge node found",
             in_node->GetName().c_str(), src_out_anchor->GetIdx(), node->GetName().c_str(), in_data_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    auto merge_node = iter->second[src_out_anchor->GetIdx()];
    if (merge_node == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to link data edge from %s(%s)(index:%d) to %s(%s)(index:%d), "
                         "cause no merge node found, check invalid",
                         in_node->GetName().c_str(), in_node->GetType().c_str(), src_out_anchor->GetIdx(),
                         node->GetName().c_str(), node->GetType().c_str(), in_data_anchor->GetIdx());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to link IO data edge from %s(%d) to %s(%d), no merge node found",
             in_node->GetName().c_str(), src_out_anchor->GetIdx(), node->GetName().c_str(), in_data_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    auto ret = src_out_anchor->Unlink(in_data_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_INNER_ERROR("E19999", "Unlink edge from %s(%s)(index:%d) to %s(%s)(index:%d) failed",
                         in_node->GetName().c_str(), in_node->GetType().c_str(), src_out_anchor->GetIdx(),
                         node->GetName().c_str(), node->GetType().c_str(), in_data_anchor->GetIdx());
      GELOGE(INTERNAL_ERROR, "[Unlink][Edge] from %s(%s)(index:%d) to %s(%s)(index:%d) failed",
             in_node->GetName().c_str(), in_node->GetType().c_str(), src_out_anchor->GetIdx(),
             node->GetName().c_str(), node->GetType().c_str(), in_data_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    ret = GraphUtils::AddEdge(merge_node->GetOutDataAnchor(kMergeDataOutIndex), in_data_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                        merge_node->GetName().c_str(), merge_node->GetType().c_str(), kMergeDataOutIndex,
                        node->GetName().c_str(), node->GetType().c_str(), in_data_anchor->GetIdx());
      GELOGE(INTERNAL_ERROR, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
             merge_node->GetName().c_str(), merge_node->GetType().c_str(), kMergeDataOutIndex,
             node->GetName().c_str(), node->GetType().c_str(), in_data_anchor->GetIdx());
      return INTERNAL_ERROR;
    }
    GELOGI("Link data edge from merge %s(from %s(%d)) to %s(%d)", merge_node->GetName().c_str(),
           in_node->GetName().c_str(), src_out_anchor->GetIdx(), node->GetName().c_str(), in_data_anchor->GetIdx());
  }

  for (auto &in_node : node->GetInControlNodes()) {
    if (!IsInBatchBranch(in_node)) {
      continue;
    }
    auto iter = nodes_to_merge_nodes_.find(in_node.get());
    if (iter == nodes_to_merge_nodes_.end()) {
      REPORT_INNER_ERROR("E19999", "Failed to link IO control edge from %s(%s) to %s(%s), no merge node found,"
                         "check invalid",
                         in_node->GetName().c_str(), in_node->GetType().c_str(),
                         node->GetName().c_str(), node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to link IO control edge from %s to %s, no merge node found",
             in_node->GetName().c_str(), node->GetName().c_str());
      return INTERNAL_ERROR;
    }
    auto merge_node = iter->second[0];
    if (merge_node == nullptr) {
      REPORT_INNER_ERROR("E19999",
                         "Failed to link IO control edge from %s(%s) to %s(%s), no merge node found, check invalid",
                         in_node->GetName().c_str(), in_node->GetType().c_str(),
                         node->GetName().c_str(), node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Check][Param] Failed to link IO control edge from %s to %s, no merge node found",
             in_node->GetName().c_str(), node->GetName().c_str());
      return INTERNAL_ERROR;
    }

    GE_IF_BOOL_EXEC(in_node->GetOutControlAnchor() == nullptr,
                    REPORT_INNER_ERROR("E19999", "Out control anchor of op:%s(%s) is nullptr, check invalid",
                                       in_node->GetName().c_str(), in_node->GetType().c_str());
                    GELOGE(INTERNAL_ERROR, "[Get][OutControlAnchor]Out control anchor of op:%s(%s) is nullptr",
                           in_node->GetName().c_str(), in_node->GetType().c_str());
                    return INTERNAL_ERROR);
    auto ret = in_node->GetOutControlAnchor()->Unlink(node->GetInControlAnchor());
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                    REPORT_INNER_ERROR("E19999", "Unlink ctrl edge from %s(%s) to %s(%s) failed",
                                       in_node->GetName().c_str(), in_node->GetType().c_str(),
                                       node->GetName().c_str(), node->GetType().c_str());
                    GELOGE(INTERNAL_ERROR, "[Unlink][CtrlEdge] from %s(%s) to %s(%s) failed",
                           in_node->GetName().c_str(), in_node->GetType().c_str(),
                           node->GetName().c_str(), node->GetType().c_str());
                    return INTERNAL_ERROR);
    ret = GraphUtils::AddEdge(merge_node->GetOutControlAnchor(), node->GetInControlAnchor());
    GE_IF_BOOL_EXEC(
        ret != GRAPH_SUCCESS,
        REPORT_CALL_ERROR("E19999", "Add ctrl edge between op:%s(%s) and op:%s(%s) failed",
                          merge_node->GetName().c_str(), merge_node->GetType().c_str(),
                          node->GetName().c_str(), node->GetType().c_str());
        GELOGE(INTERNAL_ERROR, "[Add][CtrlEdge] between op:%s(%s) and op:%s(%s) failed",
               merge_node->GetName().c_str(), merge_node->GetType().c_str(),
               node->GetName().c_str(), node->GetType().c_str());
        return INTERNAL_ERROR);
    GELOGI("Link control edge from merge %s(from %s) to %s", merge_node->GetName().c_str(), in_node->GetName().c_str(),
           node->GetName().c_str());
  }

  return SUCCESS;
}

Status ProcessMultiBatch(ComputeGraphPtr &graph) {
  const char *multi_batch_with_switchn = std::getenv("MULTI_BATCH_WITH_SWITCHN");
  if (multi_batch_with_switchn == nullptr) {
    PassManager pass_manager;
    GE_CHK_STATUS_RET(pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass));
    return pass_manager.Run(graph);
  }
  if (!GetLocalOmgContext().need_multi_batch) {
    GELOGI("No need to process_multi for no_train graph.");
    return SUCCESS;
  }
  std::vector<NodePtr> data_nodes;
  std::vector<NodePtr> getnext_nosink_nodes;
  std::vector<NodePtr> getnext_sink_nodes;
  if (CheckSequenceOfOptions(graph, data_nodes, getnext_nosink_nodes, getnext_sink_nodes) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Train_Dynamic][Check][SequenceOfOptions] failed.");
    return PARAM_INVALID;
  }
  if (UpdateNameOfInputShape(graph, data_nodes, getnext_nosink_nodes, getnext_sink_nodes) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Train_Dynamic][Update][NameOfInputShape] failed.");
    return PARAM_INVALID;
  }
  if (DeleteIdentityInsertByAdapter(graph) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Call][DeleteIdentityInsertByAdapter] failed.");
    return PARAM_INVALID;
  }

  std::vector<std::vector<int64_t>> shapes;
  if (!InitDynamicParams(shapes)) {
    GELOGD("There is no multi-batch options, no need to process multi-batch copy");
    return SUCCESS;
  }
  if (CheckNegativeCountOfOptions(shapes) != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Check][Param] Input_shape and dynamic_dims should set correct params.");
    return PARAM_INVALID;
  }

  DynamicType dynamic_type = DynamicType::kDynamicUnknown;
  if (!GetLocalOmgContext().dynamic_batch_size.empty()) {
    dynamic_type = DynamicType::kDynamicBatch;
  } else if (!GetLocalOmgContext().dynamic_image_size.empty()) {
    dynamic_type = DynamicType::kDynamicImageSize;
  } else if (!GetLocalOmgContext().dynamic_dims.empty()) {
    dynamic_type = DynamicType::kDynamicDims;
  }
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_designate_shape;
  user_designate_shape = GetLocalOmgContext().user_input_dims;

  GELOGI("Begin to copy graph for multi-batch");
  multibatch::MultiBatchGraphCopyer copyer(graph);
  for (auto &shape : shapes) {
    copyer.AddShape(shape);
  }
  copyer.SetDynamicType(dynamic_type);
  copyer.SetUserDesignateShape(user_designate_shape);
  return copyer.CopyGraph();
}

//              +-----------+
//              |   Data    |                      +-----------+       +-----------+       +-----------+
//              +-----------+                      |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
//                       \                      /. +-----------+       +-----------+       +-----------+
//                        \                    /.
// +-----------+       +-----------+          /.   +-----------+       +-----------+       +-----------+
// |   Data    | ----> |    Case   |         S---  |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
// +-----------+       +-----------+          \.   +-----------+       +-----------+       +-----------+
//                               \             \.
//                                \             \. +-----------+       +-----------+       +-----------+
//                           +-----------+         |    Data   | ----> | SoftmaxV2 | ----> | NetOutput |
//                           | NetOutput |         +-----------+       +-----------+       +-----------+
//                           +-----------+
// +-----------+                  /
// |   Data    | --------------->/
// +-----------+
void GetDynamicShapeByGraph(const ComputeGraphPtr &graph, const NodePtr &node,
                            set<size_t> &dynamic_output_index, vector<string> &dynamic_output_dims) {
  GELOGD("Try get dynamic shape info, Graph: %s, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
  const auto &func_desc = node->GetOpDesc();
  if (!func_desc->HasAttr(ATTR_NAME_BATCH_NUM)) {
    GELOGD("Graph: %s Not multi-batch, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
    return;
  }

  const auto &dynamic_branch_names = func_desc->GetSubgraphInstanceNames();
  for (size_t i = 0; i < func_desc->GetOutputsSize(); ++i) {
    for (size_t j = 0; j < dynamic_branch_names.size(); ++j) {
      const auto &subgraph = graph->GetSubgraph(dynamic_branch_names[j]);
      if (subgraph == nullptr) {
        REPORT_INNER_ERROR("E19999", "Get subgraph:%s from graph:%s failed",
                           dynamic_branch_names[j].c_str(), graph->GetName().c_str());
        GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "[Get][SubGraph] %s from graph:%s failed",
               dynamic_branch_names[j].c_str(), graph->GetName().c_str());
        dynamic_output_dims.clear();
        return;
      }

      const auto &out_node = subgraph->FindFirstNodeMatchType(NETOUTPUT);
      if (out_node == nullptr) {
        REPORT_INNER_ERROR("E19999", "No netoutput node exist in subgraph:%s, check invalid",
                           subgraph->GetName().c_str());
        GELOGE(GE_GRAPH_GRAPH_NODE_NULL, "[Check][Param] No netoutput node exist in subgraph:%s",
               subgraph->GetName().c_str());
        dynamic_output_dims.clear();
        return;
      }

      GELOGI("Find the subgraph Output node %s and the index is %zu", out_node->GetName().c_str(), i);
      const auto &out_desc = out_node->GetOpDesc();
      if (out_desc == nullptr || out_desc->GetInputsSize() <= i) {
        REPORT_INNER_ERROR("E19999",
                           "op_desc of node in subgraph:%s is nullptr or input desc size:%zu <= %zu, check invalid",
                           subgraph->GetName().c_str(), out_desc->GetInputsSize(), i);
        GELOGE(GE_GRAPH_GRAPH_NODE_NULL,
               "[Check][Param] op_desc of node in subgraph:%s is nullptr or input desc size:%zu <= %zu",
               subgraph->GetName().c_str(), out_desc->GetInputsSize(), i);
        dynamic_output_dims.clear();
        return;
      }

      const auto &input_tensor = out_desc->GetInputDesc(i);
      const auto &shape_msg = input_tensor.GetShape().ToString();
      string output_shape = std::to_string(j) + "," + std::to_string(i) + "," + shape_msg;
      GELOGI("The shape msg in dynamic batch is %s", output_shape.c_str());
      dynamic_output_dims.emplace_back(output_shape);

      uint32_t parent_index = 0;
      (void)AttrUtils::GetInt(input_tensor, ATTR_NAME_PARENT_NODE_INDEX, parent_index);
      dynamic_output_index.insert(parent_index);
    }
  }
}

//                                         +-----------+       +-----------+ i = 0
//                                  +----> | SoftmaxV2 | ----> |MemcpyAsync| ----> \.
//                                 /       +-----------+       +-----------+        \.
//                                /                                                  \.
// +-----------+       +-----------+       +-----------+       +-----------+ i = 1 +-----------+
// |   Data    | ----> |  SwitchN  | ----> | SoftmaxV2 | ----> |MemcpyAsync| ----> |   Merge   |
// +-----------+       +-----------+       +-----------+       +-----------+       +-----------+
//                                \                                                  /       \.  j = 0
//                                 \       +-----------+       +-----------+ i = 2  /         \.
//                                  +----> | SoftmaxV2 | ----> |MemcpyAsync| ----> /       +-----------+
//                                         +-----------+       +-----------+               | NetOutput |
//                                                                                         +-----------+
// +-----------+                                                                              /.
// |   Data    | --------------------------------------------------------------------------->/.  j = 1
// +-----------+
void GetDynamicShapeByMerge(const ComputeGraphPtr &graph, const NodePtr &node,
                            set<size_t> &dynamic_output_index, vector<string> &dynamic_output_dims) {
  GELOGD("Try get dynamic shape info, Graph: %s, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
  const auto &netoutput_desc = node->GetOpDesc();
  const auto &inputnode_to_netoutput = node->GetInAllNodes();
  GELOGI("Train_Dynamic Find the merge node size is %zu.", inputnode_to_netoutput.size());
  for (size_t i = 0; i < inputnode_to_netoutput.size(); ++i) {
    bool insert_by_mbatch = false;
    (void)AttrUtils::GetBool(inputnode_to_netoutput.at(i)->GetOpDesc(), ATTR_INSERT_BY_MBATCH, insert_by_mbatch);
    GELOGI("Train_Dynamic type is %s", inputnode_to_netoutput.at(i)->GetType().c_str());
    if (inputnode_to_netoutput.at(i)->GetType() == MERGE && insert_by_mbatch) {
      GELOGI("Find the merge node %s with mbatch attr and the index is %zu",
             inputnode_to_netoutput.at(i)->GetName().c_str(), i);
      dynamic_output_index.insert(i);
      for (size_t j = 0; j < inputnode_to_netoutput.at(i)->GetInNodes().size(); ++j) {
        auto input_desc = inputnode_to_netoutput.at(i)->GetOpDesc();
        auto input_tensor_desc = input_desc->GetInputDesc(j);
        auto shape_msg = input_tensor_desc.GetShape().ToString();
        string output_shape = std::to_string(j) + "," + std::to_string(i) + "," + shape_msg;
        GELOGI("The shape msg in dynamic batch is %s", output_shape.c_str());
        dynamic_output_dims.emplace_back(output_shape);
      }
    }
  }
}

// Connect NetOutput directly
void GetDirectOutputShape(const ComputeGraphPtr &graph, const NodePtr &node,
                          const set<size_t> &dynamic_output_index, vector<string> &dynamic_output_dims) {
  if (!GetLocalOmgContext().dynamic_node_type.empty()) {
    GELOGD("No need to get directly shape info of %s when train.", node->GetName().c_str());
    return;
  }
  GELOGD("Try get directly shape info, Graph: %s, Node: %s", graph->GetName().c_str(), node->GetName().c_str());
  const auto &netoutput_desc = node->GetOpDesc();
  const auto &inputnode_to_netoutput = node->GetInAllNodes();
  for (size_t i = 0; i < inputnode_to_netoutput.size(); ++i) {
    if (dynamic_output_index.count(i) > 0) {
      continue;
    }

    auto tensor_desc = netoutput_desc->GetInputDesc(i);
    auto shape = tensor_desc.GetShape().ToString();
    string static_output_shape = std::to_string(kStaticOutput) + "," + std::to_string(i) + "," + shape;
    GELOGI("The static output shape msg is %s", static_output_shape.c_str());
    dynamic_output_dims.emplace_back(static_output_shape);
  }
}

Status GetDynamicOutputShape(ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  GELOGI("Start to get output dynamic batch shape message");

  NodePtr net_output;
  set<size_t> dynamic_output_index;
  vector<string> dynamic_output_dims;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == NETOUTPUT) {
      net_output = node;
      GetDynamicShapeByMerge(graph, node, dynamic_output_index, dynamic_output_dims);
    } else if (node->GetType() == CASE) {
      GetDynamicShapeByGraph(graph, node, dynamic_output_index, dynamic_output_dims);
    }
  }

  if ((net_output != nullptr) && !dynamic_output_dims.empty()) {
    GetDirectOutputShape(graph, net_output, dynamic_output_index, dynamic_output_dims);
    if (!AttrUtils::SetListStr(net_output->GetOpDesc(), ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_output_dims)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s to node:%s(%s) failed",
                        ATTR_NAME_DYNAMIC_OUTPUT_DIMS.c_str(),
                        net_output->GetName().c_str(), net_output->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s to node:%s(%s) failed", ATTR_NAME_DYNAMIC_OUTPUT_DIMS.c_str(),
             net_output->GetName().c_str(), net_output->GetType().c_str());
      return FAILED;
    }
  }

  return SUCCESS;
}
}  // namespace multibatch
}  // namespace ge
