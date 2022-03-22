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

#include "graph/passes/cast_remove_pass.h"
#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/transop_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/type_utils.h"

namespace ge {
Status CastRemovePass::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] Param [node] must not be null.");
    return PARAM_INVALID;
  }
  OpDescPtr op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param op_desc of node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] OpDesc of param [node] must not be null.");
    return PARAM_INVALID;
  }

  // begin with not trans op, and only has one out data anchor
  if (TransOpUtil::IsTransOp(node) || node->GetAllOutDataAnchorsSize() != 1) {
    return SUCCESS;
  }

  std::vector<NodePtr> nodes_to_fuse;
  NodePtr end_node = GetTheEndNode(node, nodes_to_fuse);
  if (nodes_to_fuse.empty()) {
    return SUCCESS;
  }
  OpDescPtr end_op_desc = end_node->GetOpDesc();
  if (end_op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "op_desc of end_node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] OpDesc of end node must not be null.");
    return PARAM_INVALID;
  }

  if (!CheckPrecisionLoss(nodes_to_fuse)) {
    return SUCCESS;
  }

  DataType type = DT_UNDEFINED;
  if (!HasSameDataType(op_desc, end_op_desc, type)) {
    return SUCCESS;
  }
  if (RemoveCast(type, nodes_to_fuse) != SUCCESS) {
    return FAILED;
  }
  return SUCCESS;
}

bool CastRemovePass::CheckPrecisionLoss(const std::vector<NodePtr> &nodes_to_fuse) {
  for (const NodePtr &node : nodes_to_fuse) {
    if (!TransOpUtil::CheckPrecisionLoss(node)) {
      return false;
    }
  }
  return true;
}

bool CastRemovePass::HasSameDataType(OpDescPtr &begin_op_desc, OpDescPtr &end_op_desc, DataType &type) const {
  if (begin_op_desc->GetName() == end_op_desc->GetName()) {
    return false;
  }
  auto end_out_desc = end_op_desc->MutableOutputDesc(0);
  DataType end_out_datatype = end_out_desc->GetDataType();

  auto begin_out_desc = begin_op_desc->MutableOutputDesc(0);
  DataType begin_out_datatype = begin_out_desc->GetDataType();
  if (begin_out_datatype == end_out_datatype && (begin_out_datatype == DT_FLOAT16 || begin_out_datatype == DT_FLOAT)) {
    type = begin_out_datatype;
    return true;
  }
  return false;
}

// op1->TransData->Cast->TransposeD->Cast->TransData->op2
// change to be
// op1->TransData->TransposeD->TransData->op2
Status CastRemovePass::RemoveCast(DataType &type, std::vector<NodePtr> &nodes_to_fuse) {
  string cast_name;
  for (NodePtr &node : nodes_to_fuse) {
    if (node->GetType() == CAST) {
      GELOGI("CastRemovePass, remove Cast %s.", node->GetName().c_str());
      cast_name = node->GetName();
      if (IsolateAndDeleteNode(node, {0}) != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Isolate and delete node:%s(%s) failed",
                          node->GetName().c_str(), node->GetType().c_str());
        GELOGE(FAILED, "[IsolateAndDelete][Node] %s failed.", node->GetName().c_str());
        return FAILED;
      }
    }
  }

  if (cast_name.empty()) {
    return SUCCESS;
  }
  for (auto &node : nodes_to_fuse) {
    if (node->GetType() == CAST) {
      continue;
    }
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "Find nullptr op_desc in node, check invalid");
      GELOGE(FAILED, "[Get][OpDesc] OpDesc must not be null.");
      return FAILED;
    }

    // change node name for recompile cache, will be abandoned in April
    string new_node_name = cast_name + op_desc->GetName();
    op_desc->SetName(new_node_name);
    // add attr to changed TransData, then will be rebuild
    if (!AttrUtils::SetBool(op_desc, ATTR_NEED_COMPILE, true)) {
      REPORT_CALL_ERROR("E19999", "Set Attr:%s of op:%s(%s) failed",
                        ATTR_NEED_COMPILE.c_str(),
                        op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Set][Attr] %s of op:%s(%s) failed", ATTR_NEED_COMPILE.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str());
      return FAILED;
    }
    auto in_desc = op_desc->MutableInputDesc(0);
    auto out_desc = op_desc->MutableOutputDesc(0);
    in_desc->SetDataType(type);
    out_desc->SetDataType(type);
    GELOGI("CastRemovePass, change %s %s datatype to be %s.", node->GetType().c_str(), node->GetName().c_str(),
           TypeUtils::DataTypeToSerialString(type).c_str());
  }
  return SUCCESS;
}

NodePtr CastRemovePass::GetTheEndNode(NodePtr begin_node, std::vector<NodePtr> &nodes_to_fuse) {
  while (begin_node->GetOutDataNodes().size() == 1) {
    auto out_node = begin_node->GetOutDataNodes().at(0);
    if (!TransOpUtil::IsTransOp(out_node)) {
      return begin_node;  // when seen not trans op
    }
    begin_node = out_node;
    nodes_to_fuse.emplace_back(begin_node);
  }
  return begin_node;  // when seen branch
}
}  // namespace ge
