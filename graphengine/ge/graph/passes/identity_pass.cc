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

#include "graph/passes/identity_pass.h"

#include <string>
#include <vector>
#include "framework/common/debug/ge_log.h"
#include "common/omg_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
namespace {
///
/// 1. A `Identity` node may after a `Switch` node and has control-dependency-out nodes.
/// Or a `Identity` node may before a `Merge` node and has control-dependency-in nodes.
/// The identity nodes are used to represent control dependencies in condition branch, and can not be deleted.
/// 2. Check identity is near subgraph.
///    Eg. As output of Data node in subgraph
///        or as input of Netoutput of subgraph
///        or as input of one node with subgraph
///        or as output of one node with subgraph
/// 3. identity with attr no_need_constant_folding should not be deleted too
Status CheckIdentityUsable(const NodePtr &node, bool &usable) {
  std::string node_type;
  if (node->GetOpDesc()->HasAttr(ge::ATTR_NO_NEED_CONSTANT_FOLDING)) {
    usable = true;
    return SUCCESS;
  }

  for (auto &in_node : node->GetInDataNodes()) {
    auto in_node_opdesc = in_node->GetOpDesc();
    GE_CHECK_NOTNULL(in_node_opdesc);
    // near entrance of subgraph || near subgraph
    if ((in_node->GetType() == DATA && NodeUtils::IsSubgraphInput(in_node))
        || !in_node_opdesc->GetSubgraphInstanceNames().empty()) {
      usable = true;
      return SUCCESS;
    }

    GE_CHK_STATUS_RET(GetOriginalType(in_node, node_type),
                      "[Get][OriginalType] of node:%s failed", in_node->GetName().c_str());
    bool need_skip = (node_type != SWITCH) && (node_type != REFSWITCH) && (node_type != SWITCHN);
    if (need_skip) {
      GELOGD("skip identity %s connected to switch", node->GetName().c_str());
      break;
    }
    GE_CHECK_NOTNULL(node->GetOutControlAnchor());
    if (!node->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
      usable = true;
      return SUCCESS;
    }
  }
  for (auto &out_node : node->GetOutDataNodes()) {
    auto out_node_opdesc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(out_node_opdesc);
    // near output of subgraph || near subgraph
    if (NodeUtils::IsSubgraphOutput(out_node)
        || !out_node_opdesc->GetSubgraphInstanceNames().empty()) {
      usable = true;
      return SUCCESS;
    }
    GE_CHK_STATUS_RET(GetOriginalType(out_node, node_type),
                      "[Get][OriginalType] of node:%s failed", out_node->GetName().c_str());
    if ((node_type != MERGE) && (node_type != REFMERGE)) {
      GELOGD("skip identity %s connected to merge", node->GetName().c_str());
      break;
    }
    GE_CHECK_NOTNULL(node->GetInControlAnchor());
    if (!node->GetInControlAnchor()->GetPeerOutControlAnchors().empty()) {
      usable = true;
      return SUCCESS;
    }
  }
  usable = false;
  return SUCCESS;
}
}  // namespace

Status IdentityPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  string type;
  Status status_ret = GetOriginalType(node, type);
  if (status_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get original type of node:%s failed", node->GetName().c_str());
    GELOGE(status_ret, "[Get][OriginalType] of node:%s failed.", node->GetName().c_str());
    return status_ret;
  }
  if ((type != IDENTITY) && (type != IDENTITYN) && (type != READVARIABLEOP)) {
    return SUCCESS;
  }

  if (!force_) {
    bool usable = false;
    auto ret = CheckIdentityUsable(node, usable);
    if (ret != SUCCESS) {
      return ret;
    }
    if (usable) {
      return SUCCESS;
    }
  }
  size_t n = node->GetOpDesc()->GetOutputsSize();
  if (node->GetOpDesc()->GetInputsSize() != n) {
    REPORT_CALL_ERROR("E19999", "Num:%zu of input desc node:%s(%s) not equal to it's output desc num:%zu, "
                      "check invalid", node->GetOpDesc()->GetInputsSize(),
                      node->GetName().c_str(), node->GetType().c_str(), n);
    GELOGE(PARAM_INVALID, "[Check][Param] Num:%zu of input desc node:%s(%s) not equal to it's output desc num:%zu",
           node->GetOpDesc()->GetInputsSize(), node->GetName().c_str(), node->GetType().c_str(), n);
    return PARAM_INVALID;
  }
  std::vector<int> io_map;
  for (size_t i = 0; i < n; i++) {
    io_map.push_back(i);
  }
  return IsolateAndDeleteNode(node, io_map);
}
}  // namespace ge
