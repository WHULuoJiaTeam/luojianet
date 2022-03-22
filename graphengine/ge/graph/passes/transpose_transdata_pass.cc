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

#include "graph/passes/transpose_transdata_pass.h"

#include <memory>
#include <string>
#include <vector>
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "init/gelib.h"
#include "opskernel_manager/ops_kernel_manager.h"

namespace {
const char *const kAttrNameSrcFormat = "src_format";
}  // namespace

namespace ge {
Status TransposeTransDataPass::Run(NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Check][Param] param [node] must not be null.");
    return PARAM_INVALID;
  }
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param node's op_desc is nullptr, check invalid");
    GELOGE(PARAM_INVALID, "[Get][OpDesc] failed, OpDesc of param [node] must not be null.");
    return PARAM_INVALID;
  }

  if (op_desc->GetType() != TRANSPOSED) {
    return SUCCESS;
  }
  auto input_format = op_desc->GetInputDescPtr(0)->GetFormat();
  auto output_format = op_desc->GetOutputDescPtr(0)->GetFormat();
  if (input_format ==  output_format) {
    GELOGW("Node %s input format is %s, output format is %s, should not happend. Ignore pass.",
           op_desc->GetName().c_str(),
           TypeUtils::FormatToSerialString(input_format).c_str(),
           TypeUtils::FormatToSerialString(output_format).c_str());
    return SUCCESS;
  }
  if (CheckOneInAndOneOutDataAnchor(node) != SUCCESS) {
    return FAILED;
  }
  bool is_unknown = false;
  auto ret = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("Get node unknown status failed, node name:%s, type:%s.", node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  if (is_unknown) {
    GELOGI("Current node %s, type %s is unknown shape which should be skip.", node->GetName().c_str(),
           node->GetType().c_str());
    return SUCCESS;
  }
  GELOGD("[%s] TransposeTransDataPass in.", node->GetName().c_str());

  auto out_nodes = node->GetOutDataNodes();
  bool is_add_flag = false;
  for (auto &out_node : out_nodes) {
    GE_CHECK_NOTNULL(out_node);
    OpDescPtr out_op_desc = out_node->GetOpDesc();
    if (out_op_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "OpDesc in node is nullptr, check invalid");
      GELOGE(FAILED, "[Get][OpDesc] failed, OpDesc of out data node must not be null.");
      return FAILED;
    }
    if (out_op_desc->GetType() != TRANSDATA) {
      continue;
    }
    if (CheckOneInAndOneOutDataAnchor(out_node)) {
      return FAILED;
    }
    if (!FusionIfNeed(op_desc, out_node)) {
      continue;
    }
    CopyInputEdges(node, out_node);
    is_add_flag = true;
  }

  if (is_add_flag) {
    AddRePassNode(node->GetInDataNodes().at(0));
  }
  if (node->GetOutDataNodesSize() == 0) {
    // all output nodes of transpose has fused, delete transpose
    return RemoveTranspose(node);
  }
  return SUCCESS;
}

Status TransposeTransDataPass::CheckOneInAndOneOutDataAnchor(NodePtr &node) const {
  GE_CHECK_NOTNULL(node);
  // Trans op has one input one output data anchor
  uint32_t in_data_anchor_nums = node->GetAllInDataAnchorsSize();
  uint32_t out_data_anchor_nums = node->GetAllOutDataAnchorsSize();
  // Trans op has one input data node, maybe has N output data nodes
  uint32_t in_data_node_nums = node->GetInDataNodes().size();
  if (in_data_anchor_nums != 1 || out_data_anchor_nums != 1 || in_data_node_nums != 1) {
    REPORT_INNER_ERROR("E19999", "In data anchor num:%u, out data anchor num:%u, in data node num:%u of node:%s(%s) "
                       "must be all equal to 1, check invalid", in_data_anchor_nums,
                       out_data_anchor_nums, in_data_node_nums, node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Check][Param] In data anchor num:%u, out data anchor num:%u, in data node num:%u of node:%s(%s) "
           "must be all equal to 1.", in_data_anchor_nums, out_data_anchor_nums, in_data_node_nums,
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }
  return SUCCESS;
}

Status TransposeTransDataPass::RemoveTranspose(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  ComputeGraphPtr graph = node->GetOwnerComputeGraph();
  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "Owner graph of node:%s(%s) is nullptr, check invalid",
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(FAILED, "[Get][OwnerComputeGraph] failed, The owner graph of node:%s(%s) must not be null.",
           node->GetName().c_str(), node->GetType().c_str());
    return FAILED;
  }

  // If delete Transpos/TransposeD, change its peer in ctrl anchor to its input node
  // If not delete, need do nothing
  auto origin_node_in = node->GetInDataNodes().at(0);
  GE_CHECK_NOTNULL(node->GetOutControlAnchor());
  for (auto &peer_anchor : node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
    GE_CHECK_NOTNULL(origin_node_in);
    GE_CHECK_NOTNULL(origin_node_in->GetOutControlAnchor());
    GE_CHK_STATUS_RET(origin_node_in->GetOutControlAnchor()->LinkTo(peer_anchor),
                      "[Link][Anchor] between %s and %s failed",
                      origin_node_in->GetName().c_str(), peer_anchor->GetOwnerNode()->GetName().c_str());
  }

  for (const auto &anchor : node->GetAllInAnchors()) {
    GE_CHECK_NOTNULL(anchor);
    anchor->UnlinkAll();
  }
  for (const auto &anchor : node->GetAllOutAnchors()) {
    GE_CHECK_NOTNULL(anchor);
    anchor->UnlinkAll();
  }
  AddNodeDeleted(node);
  if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Remove node:%s(%s) without relink in graph:%s failed",
                      node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    GELOGE(FAILED, "[Remove][Node] %s(%s) without relink in graph:%s failed",
           node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

bool TransposeTransDataPass::FusionIfNeed(OpDescPtr &op_desc, NodePtr &node) {
  auto transdata_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  GE_CHECK_NOTNULL(transdata_op_desc);
  auto out_input_desc = transdata_op_desc->MutableInputDesc(0);
  GE_CHECK_NOTNULL(out_input_desc);
  auto out_input_format = out_input_desc->GetFormat();
  auto out_input_shape = out_input_desc->GetShape();

  auto input_desc = op_desc->MutableInputDesc(0);
  auto out_desc = op_desc->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(input_desc);
  GE_CHECK_NOTNULL(out_desc);
  auto src_format = input_desc->GetFormat();
  auto dst_format = out_desc->GetFormat();
  auto &dst_shape = out_desc->MutableShape();
  if (dst_format != out_input_format || !formats::IsShapeEqual(dst_shape, out_input_shape) || src_format == FORMAT_ND) {
    GELOGD("Output of transpose isn't the same as input of transdata, or transpose input format must not be ND.");
    GELOGD("Transpose input format %s, output format %s shape %s. transdata in %s %s.",
           TypeUtils::FormatToSerialString(src_format).c_str(), TypeUtils::FormatToSerialString(dst_format).c_str(),
           formats::ShapeToString(dst_shape.GetDims()).c_str(),
           TypeUtils::FormatToSerialString(out_input_format).c_str(),
           formats::ShapeToString(out_input_shape.GetDims()).c_str());
    return false;
  }

  auto &src_shape = input_desc->MutableShape();
  GELOGI("Begin to fuse transpose transdata, transpose in format %s shape %s, transdata in %s %s",
         TypeUtils::FormatToSerialString(src_format).c_str(), formats::ShapeToString(src_shape.GetDims()).c_str(),
         TypeUtils::FormatToSerialString(out_input_format).c_str(),
         formats::ShapeToString(out_input_shape.GetDims()).c_str());

  // Transpose can change format and shape
  out_input_desc->SetFormat(src_format);
  out_input_desc->SetShape(src_shape);

  if (!TransDataCheckAccuracySupported(node)) {
    out_input_desc->SetFormat(out_input_format);
    out_input_desc->SetShape(out_input_shape);
    return false;
  }

  // add attr to fused TransData, then will be rebuild
  string new_node_name = op_desc->GetName() + transdata_op_desc->GetName();
  transdata_op_desc->SetName(new_node_name);
  GE_IF_BOOL_EXEC(!AttrUtils::SetBool(transdata_op_desc, ATTR_NEED_COMPILE, true), GELOGW("set ext attr failed");
                  return false);

  string format_val = TypeUtils::FormatToSerialString(src_format);
  GE_IF_BOOL_EXEC(!AttrUtils::SetStr(transdata_op_desc, kAttrNameSrcFormat, format_val),
                  GELOGW("set kAttrNameSrcFormat failed");
                  return false);
  GELOGI("TransposeTransDataPass, fuse to be node %s.", transdata_op_desc->GetName().c_str());
  return true;
}

void TransposeTransDataPass::CopyInputEdges(NodePtr &origin_node, NodePtr &new_node) {
  if (origin_node == nullptr || new_node == nullptr) {
    return;
  }
  InDataAnchorPtr new_in_data_anchor = new_node->GetInDataAnchor(0);
  if (new_in_data_anchor == nullptr || origin_node->GetInDataAnchor(0) == nullptr) {
    return;
  }
  OutDataAnchorPtr out_anchor = origin_node->GetInDataAnchor(0)->GetPeerOutAnchor();
  new_in_data_anchor->UnlinkAll();
  GE_IF_BOOL_EXEC(new_in_data_anchor->LinkFrom(out_anchor) != GRAPH_SUCCESS, GELOGW("Link failed"); return);

  // control anchor only link to control anchor
  GE_IF_BOOL_EXEC(
    GraphUtils::CopyInCtrlEdges(origin_node, new_node) != GRAPH_SUCCESS, GELOGW("Copy in ctrl edges failed"); return);
}

bool TransposeTransDataPass::TransDataCheckAccuracySupported(NodePtr &node) {
  const OpDescPtr &op_desc = node->GetOpDesc();
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGW("GELib not initialized");
    return false;
  }

  OpsKernelManager &ops_kernel_manager = instance_ptr->OpsKernelManagerObj();
  vector<OpInfo> op_infos = ops_kernel_manager.GetOpsKernelInfo(op_desc->GetType());
  if (op_infos.empty()) {
    GELOGW("Can not get op info by op type %s", op_desc->GetType().c_str());
    return false;
  }

  std::string unsupported_reason;
  for (auto &it : op_infos) {
    auto kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
    auto &kernel_name = it.opKernelLib;
    auto kernel_info_store = kernel_map.find(kernel_name);
    if (kernel_info_store != kernel_map.end()) {
      if (kernel_info_store->second->CheckAccuracySupported(node, unsupported_reason, true)) {
        return true;
      }
    }
  }
  GELOGI("TransposeTransDataPass CheckAccuracySupported[%s] all not support, reason:%s.", op_desc->GetName().c_str(),
         unsupported_reason.c_str());
  return false;
}
}  // namespace ge
