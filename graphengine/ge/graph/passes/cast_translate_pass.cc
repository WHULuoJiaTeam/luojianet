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

#include "graph/passes/cast_translate_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/pass_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"

namespace ge {
bool CastTranslatePass::CheckInAndOutDataAnchor(NodePtr &node) const {
  if (node == nullptr) {
    GELOGE(FAILED, "[Check][Param] parameter node is nullptr.");
    return false;
  }
  if (node->GetOpDesc() == nullptr) {
    GELOGW("Param [node] op desc is null.");
    return false;
  }

  auto in_anchors = node->GetAllInDataAnchors();
  auto out_anchors = node->GetAllOutDataAnchors();
  // Cast|Translate has one input one output data anchor
  if (in_anchors.size() != 1 || out_anchors.size() != 1) {
    return false;
  }
  return true;
}

bool CastTranslatePass::IsCastNode(NodePtr &node) const {
  std::string original_type;
  GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS,
          GELOGW("get original type failed"); return false);
  return (original_type == CAST);
}

bool CastTranslatePass::IsTranslateNode(NodePtr &node) const {
  std::string original_type;
  GE_IF_BOOL_EXEC(GetOriginalType(node, original_type) != SUCCESS,
                    GELOGW("get original type failed"); return false);
  return (original_type == TRANSLATE);
}

bool CastTranslatePass::IsSameCastOrTranslate(NodePtr &node, NodePtr &base_node) const {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGW("node is null."); return false);
  GE_IF_BOOL_EXEC(base_node == nullptr, GELOGW("base_node is null."); return false);
  auto op_desc = node->GetOpDesc();
  GE_IF_BOOL_EXEC(op_desc == nullptr, return false);
  auto base_op_desc = base_node->GetOpDesc();
  GE_IF_BOOL_EXEC(base_op_desc == nullptr, return false);
  auto in_desc = op_desc->MutableInputDesc(0);
  auto out_desc = op_desc->MutableOutputDesc(0);
  auto base_in_desc = base_op_desc->MutableInputDesc(0);
  auto base_out_desc = base_op_desc->MutableOutputDesc(0);
  GE_IF_BOOL_EXEC(in_desc == nullptr, GELOGW("in_desc is null."); return false);
  GE_IF_BOOL_EXEC(out_desc == nullptr, GELOGW("out_desc is null."); return false);
  GE_IF_BOOL_EXEC(base_in_desc == nullptr, GELOGW("base_in_desc is null."); return false);
  GE_IF_BOOL_EXEC(base_out_desc == nullptr, GELOGW("base_out_desc is null."); return false);
  if (in_desc->GetDataType() == base_in_desc->GetDataType() &&
      out_desc->GetDataType() == base_out_desc->GetDataType() && in_desc->GetFormat() == base_in_desc->GetFormat() &&
      out_desc->GetFormat() == base_out_desc->GetFormat()) {
    return true;
  }
  GELOGD("Output node [%s] isn't the same Cast or Translate.", node->GetName().c_str());
  return false;
}

bool CastTranslatePass::IsNodeNeedOptimize(NodePtr &node) const {
  if (CheckInAndOutDataAnchor(node) && (IsCastNode(node) || IsTranslateNode(node))) {
    return true;
  }
  return false;
}

bool CastTranslatePass::CheckDstNode(NodePtr &out_node, bool &is_src_cast) const {
  return (CheckInAndOutDataAnchor(out_node) &&
          ((!is_src_cast && IsCastNode(out_node)) || (is_src_cast && IsTranslateNode(out_node))));
}

bool CastTranslatePass::IsNextNodeNeedOptimize(NodePtr &node, bool &is_src_cast) const {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGW("cast_node is null."); return false);
  const std::string &node_name = node->GetName();
  auto out_data_nodes = node->GetOutDataNodes();
  if (out_data_nodes.empty()) {
    return false;
  }
  auto &out_node = out_data_nodes.at(0);
  bool is_first = true;
  // Cast-->all Translate; Translate-->all Cast
  for (auto &out_data_node : out_data_nodes) {
    if (out_data_node == nullptr) {
      continue;
    }
    if (CheckDstNode(out_data_node, is_src_cast) && (is_first || IsSameCastOrTranslate(out_data_node, out_node))) {
      is_first = false;
      continue;
    }
    GELOGD("[%s] Output node is %s, can't optimize.", node_name.c_str(), out_data_node->GetType().c_str());
    return false;
  }

  GELOGD("[%s] %zu dst nodes have the same input and output.", node_name.c_str(), out_data_nodes.size());
  return true;
}

bool CastTranslatePass::IsOpSupportedOptimize(NodePtr &cast_node, NodePtr &trans_node, bool &is_src_cast) {
  GE_IF_BOOL_EXEC(cast_node == nullptr, GELOGW("cast_node is null."); return false);
  GE_IF_BOOL_EXEC(trans_node == nullptr, GELOGW("trans_node is null."); return false);
  OpDescPtr trans_op_desc = trans_node->GetOpDesc();
  GE_IF_BOOL_EXEC(trans_op_desc == nullptr, GELOGW("trans_op_desc is null."); return false);
  // backup datatype
  const auto &trans_op_indesc = trans_op_desc->MutableInputDesc(0);
  const auto &trans_op_outdesc = trans_op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL_EXEC(trans_op_indesc, return false);
  GE_CHECK_NOTNULL_EXEC(trans_op_outdesc, return false);
  DataType trans_in_datatype = trans_op_indesc->GetDataType();
  DataType trans_out_datatype = trans_op_outdesc->GetDataType();

  auto cast_op_desc = cast_node->GetOpDesc();
  GE_IF_BOOL_EXEC(cast_op_desc == nullptr, GELOGW("cast_op_desc is null."); return false);
  const auto &cast_op_indesc = cast_op_desc->MutableInputDesc(0);
  const auto &cast_op_outdesc = cast_op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL_EXEC(cast_op_indesc, return false);
  GE_CHECK_NOTNULL_EXEC(cast_op_outdesc, return false);
  DataType cast_in_datatype = cast_op_indesc->GetDataType();
  DataType cast_out_datatype = cast_op_outdesc->GetDataType();
  GELOGI("CastTranslatePass, cast in %s out %s, translate in %s out %s.",
         TypeUtils::DataTypeToSerialString(cast_in_datatype).c_str(),
         TypeUtils::DataTypeToSerialString(cast_out_datatype).c_str(),
         TypeUtils::DataTypeToSerialString(trans_in_datatype).c_str(),
         TypeUtils::DataTypeToSerialString(trans_out_datatype).c_str());

  if (is_src_cast) {
    // A-->Cast-->Translate
    // change Translate input datatype to be the input of Cast
    // then delete Cast
    // [MutableInputDesc guarantees non empty throughout the process]
    trans_op_indesc->SetDataType(cast_in_datatype);
  } else {
    // Translate-->Cast-->A
    // change Translate output datatype to be the output of Cast
    // then delete Cast
    // [MutableInputDesc guarantees non empty throughout the process]
    trans_op_outdesc->SetDataType(cast_out_datatype);
  }

  if (!TranslateCheckAccuracySupported(trans_node)) {
    if (is_src_cast) {
      trans_op_desc->MutableInputDesc(0)->SetDataType(trans_in_datatype);
    } else {
      trans_op_desc->MutableOutputDesc(0)->SetDataType(trans_out_datatype);
    }
    GELOGW("CheckAccuracySupported fail, don't delete Cast[%s].", cast_node->GetName().c_str());
    return false;
  }

  if (is_src_cast) {
    GE_IF_BOOL_EXEC(
            !AttrUtils::SetInt(trans_op_desc, ATTR_NAME_INPUT_DATATYPE, static_cast<int64_t>(cast_in_datatype)),
            GELOGW("set ATTR_NAME_INPUT_DATATYPE failed"); return false);
  } else {
    GE_IF_BOOL_EXEC(
            !AttrUtils::SetInt(trans_op_desc, ATTR_NAME_OUTPUT_DATATYPE, static_cast<int64_t>(cast_out_datatype)),
            GELOGW("set ATTR_NAME_INPUT_DATATYPE failed"); return false);
  }
  GELOGI("CastTranslatePass, translate in %d out %d.", trans_op_indesc->GetDataType(), trans_op_outdesc->GetDataType());
  return true;
}

bool CastTranslatePass::CheckOpSupportOptimize(NodePtr &node, bool &is_src_cast) {
  GE_IF_BOOL_EXEC(node == nullptr, GELOGE(FAILED, "[Check][Param] node is nullptr."); return false);
  auto out_node = node->GetOutDataNodes().at(0);
  // N dst nodes have the same datatype and format, check the first node
  if (is_src_cast) {
    return IsOpSupportedOptimize(node, out_node, is_src_cast);
  } else {
    return IsOpSupportedOptimize(out_node, node, is_src_cast);
  }
}

Status CastTranslatePass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);

  bool is_src_cast = IsCastNode(node);
  if (!IsNodeNeedOptimize(node) || !IsNextNodeNeedOptimize(node, is_src_cast)) {
    return SUCCESS;
  }

  GELOGI("CastTranslatePass, optimize %s.", node->GetName().c_str());
  if (CheckOpSupportOptimize(node, is_src_cast)) {
    if (is_src_cast) {
      if (FuseDstNTranslates(node) != SUCCESS) {
        return FAILED;
      }
      return IsolateAndDeleteNode(node, {0});
    } else {
      auto out_data_nodes = node->GetOutDataNodes();
      for (auto &out_data_node : out_data_nodes) {
        if (out_data_node == nullptr) {
          continue;
        }
        if (IsolateAndDeleteNode(out_data_node, {0}) != SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Isolate and delete node:%s(%s) failed",
                            out_data_node->GetName().c_str(), out_data_node->GetType().c_str());
          return FAILED;
        }
      }
    }
  }

  return SUCCESS;
}

Status CastTranslatePass::FuseDstNTranslates(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto out_data_nodes = node->GetOutDataNodes();
  size_t nums = out_data_nodes.size();
  if (nums == 1) {
    return SUCCESS;
  }

  auto &base_node = out_data_nodes.at(0);
  GE_CHECK_NOTNULL(base_node);
  for (size_t i = 1; i < nums; i++) {
    auto &out_data_node = out_data_nodes.at(i);
    GE_CHECK_NOTNULL(out_data_node);
    AddRePassNodesWithInOut(out_data_node);
    // Has checked nodes only has one in data anchor one out data anchor
    GE_CHK_GRAPH_STATUS_RET(NodeUtils::MoveOutputEdges(out_data_node, base_node),
                            "[Move][OutputEdges] failed, out data node:%s, index:0",
                            base_node->GetName().c_str());

    // Relink in control anchor, delete in data anchor
    auto in_ctr_anchor = out_data_node->GetInControlAnchor();
    GE_CHECK_NOTNULL(in_ctr_anchor);
    for (const auto &peer_anchor : in_ctr_anchor->GetPeerOutControlAnchors()) {
      GE_CHECK_NOTNULL(base_node->GetInControlAnchor());
      GE_CHK_GRAPH_STATUS_RET(base_node->GetInControlAnchor()->LinkFrom(peer_anchor),
                              "[Add][Edge] between %s and %s failed",
                              base_node->GetInControlAnchor()->GetOwnerNode()->GetName().c_str(),
                              peer_anchor->GetOwnerNode()->GetName().c_str());
    }
    in_ctr_anchor->UnlinkAll();
    out_data_node->GetAllInDataAnchors().at(0)->UnlinkAll();

    ComputeGraphPtr graph = out_data_node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(graph);
    if (GraphUtils::RemoveNodeWithoutRelink(graph, out_data_node) != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                        out_data_node->GetName().c_str(), out_data_node->GetType().c_str(), graph->GetName().c_str());
      GELOGE(FAILED, "[Remove][Node] %s(%s) without relink in graph:%s failed",
             out_data_node->GetName().c_str(), out_data_node->GetType().c_str(), graph->GetName().c_str());
      return FAILED;
    }
    AddNodeDeleted(out_data_node);
  }

  return SUCCESS;
}

bool CastTranslatePass::TranslateCheckAccuracySupported(NodePtr &node) {
  const OpDescPtr &op_desc = node->GetOpDesc();
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGW("GE is not initialized or is finalized.");
    return false;
  }

  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGE(FAILED, "Opdesc is nullptr"); return false);
  vector<OpInfo> op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    GELOGI("Can not get op info by op type %s", op_desc->GetType().c_str());
    return false;
  }

  std::string unsupported_reason;
  for (auto &it : op_infos) {
    auto kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
    auto &kernel_name = it.opKernelLib;
    auto kernel_info_store = kernel_map.find(kernel_name);
    if (kernel_info_store != kernel_map.end()) {
      if (kernel_info_store->second != nullptr &&
          kernel_info_store->second->CheckAccuracySupported(node, unsupported_reason)) {
        return true;
      }
    }
  }
  GELOGI("CastTranslatePass CheckAccuracySupported[%s] fail.", op_desc->GetName().c_str());
  return false;
}
}  // namespace ge
