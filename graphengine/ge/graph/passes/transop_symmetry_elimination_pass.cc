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

#include "graph/passes/transop_symmetry_elimination_pass.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "common/transop_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "framework/common/types.h"

namespace {
const std::set<std::string> white_list_op{ge::TRANSPOSED, ge::RESHAPE, ge::REFORMAT, ge::CAST, ge::TRANSDATA};
}  // namespace
namespace ge {
Status TransOpSymmetryEliminationPass::Run(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  if (white_list_op.find(node->GetType()) == white_list_op.end()) { return SUCCESS; }
  GELOGD("Symmetry Elimination Pass in.");
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_CHECK_NOTNULL(out_anchor);
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNode());
      GE_CHECK_NOTNULL(peer_in_anchor->GetOwnerNode()->GetOpDesc());
      if (!CheckCanBeEliminated(node, peer_in_anchor)) { continue; }
      auto dst_node = peer_in_anchor->GetOwnerNode();
      Status ret = EliminateTransOp(node, out_anchor, dst_node, peer_in_anchor);
      if (ret != SUCCESS) {
        // if eliminate failed ,it should't break precess, so give a warning here
        GELOGW("Eliminate %s and %s failed, ignore current pass.", node->GetName().c_str(),
               dst_node->GetName().c_str());
        return ret;
      }
    }
  }
  GELOGD("Symmetry Elimination Pass end.");
  return SUCCESS;
}

bool TransOpSymmetryEliminationPass::CheckCanBeEliminated(const ge::NodePtr &src_node,
                                                          const InDataAnchorPtr &dst_in_anchor) {
  auto dst_node = dst_in_anchor->GetOwnerNode();
  if (src_node->GetType() != dst_node->GetType()) {
    GELOGD("Pre node %s type %s is not equal with node %s type %s. Ignore pass.", src_node->GetName().c_str(),
           src_node->GetType().c_str(), dst_node->GetName().c_str(), dst_node->GetType().c_str());
    return false;
  }
  if (dst_in_anchor->GetIdx() != TransOpUtil::GetTransOpDataIndex(src_node)) {
    GELOGD("Next node %s type %s input %d is not for transform. Ignore pass.", dst_node->GetName().c_str(),
           dst_node->GetType().c_str(), dst_in_anchor->GetIdx());
    return false;
  }
  if (src_node->GetType() == ge::RESHAPE) {
    GE_CHECK_NOTNULL(src_node->GetOpDesc());
    auto unknown_dims_num = GetUnknownDimsNum(src_node->GetOpDesc()->GetInputDesc(0));
    if (unknown_dims_num != 0 && (unknown_dims_num == UNKNOWN_DIM_NUM || unknown_dims_num > 1)) {
      GELOGD("Pre node %s is reshape op which input is dynamic shape and has more than one unknown dimension. "
             "Ignore pass.",
             src_node->GetName().c_str());
      return false;
    }
  } else if (src_node->GetType() == ge::TRANSPOSED) {
    if (!JudgeTransposeDBack2Raw(src_node, dst_node)) {
      GELOGD("Two Transpose op src node %s dst node %s will change the raw data. Ignore pass.",
             src_node->GetName().c_str(), dst_node->GetName().c_str());
      return false;
    }
  } else if (src_node->GetType() == ge::TRANSDATA) {
    auto unknown_dims_num = GetUnknownDimsNum(src_node->GetOpDesc()->GetInputDesc(0));
    if (unknown_dims_num == UNKNOWN_DIM_NUM) {
      GELOGD("Pre node %s is transdata op which input is dynamic shape and all dimension are unknown(-2). Ignore pass.",
             src_node->GetName().c_str());
      return false;
    }
  }
  return TransOpUtil::CheckPrecisionLoss(src_node) && DescAreSymmetry(src_node, dst_node);
}

bool TransOpSymmetryEliminationPass::DescAreSymmetry(const NodePtr &src_node, const NodePtr &dst_node) {
  const auto &src_input_desc = src_node->GetOpDesc()->MutableInputDesc(0);
  const auto &dst_output_desc = dst_node->GetOpDesc()->MutableOutputDesc(0);
  GE_CHECK_NOTNULL(src_input_desc);
  GE_CHECK_NOTNULL(dst_output_desc);
  const auto &src_input_dtype = src_input_desc->GetDataType();
  const auto &src_input_format = src_input_desc->GetFormat();
  const auto &src_input_shape = src_input_desc->GetShape().GetDims();
  const auto &dst_output_dtype = dst_output_desc->GetDataType();
  const auto &dst_output_format = dst_output_desc->GetFormat();
  const auto &dst_output_shape = dst_output_desc->GetShape().GetDims();

  bool is_symmetry = true;
  if (src_node->GetType() == CAST && dst_node->GetType() == CAST) {
    bool is_format_symmetry =
        (src_input_format == dst_output_format) || (dst_output_format == FORMAT_ND) || (src_input_format == FORMAT_ND);
    is_symmetry = (src_input_dtype == dst_output_dtype) && is_format_symmetry;
  } else {
    is_symmetry = (src_input_dtype == dst_output_dtype) && (src_input_shape == dst_output_shape)
        && (src_input_format == dst_output_format);
  }
  if (!is_symmetry) {
    GELOGD("Not satisfied symmetry. ignore pass.\n"
           "Src node %s input type: %s format: %s shape: %s, "
           "dst node %s output type: %s format: %s shape: %s. ",
           src_node->GetName().c_str(), TypeUtils::DataTypeToSerialString(src_input_dtype).c_str(),
           TypeUtils::FormatToSerialString(src_input_format).c_str(), formats::ShapeToString(src_input_shape).c_str(),
           dst_node->GetName().c_str(), TypeUtils::DataTypeToSerialString(dst_output_dtype).c_str(),
           TypeUtils::FormatToSerialString(dst_output_format).c_str(),
           formats::ShapeToString(dst_output_shape).c_str());
  }
  return is_symmetry;
}

int TransOpSymmetryEliminationPass::GetUnknownDimsNum(const GeTensorDesc& node_desc){
  //
  //  unknown_dims_num != 0 , is dynamic shape
  //  unknown_dims_num = UNKNOWN_DIM_NUM , all dims are unknown
  //  unknown_dims_num = n , n > 0 , has n dims unknown
  //
  int unknown_dims_num = 0;
  auto ge_shape = node_desc.GetShape();
  for (const auto dim : ge_shape.GetDims()) {
    if (dim == UNKNOWN_DIM_NUM) { return UNKNOWN_DIM_NUM; }
    if (dim == UNKNOWN_DIM) { ++unknown_dims_num; }
  }
  return unknown_dims_num;
}

bool TransOpSymmetryEliminationPass::JudgeTransposeDBack2Raw(const NodePtr &src_node, const NodePtr &dst_node) {
  //
  //  A transpose to C : A---->(perm_1)---->B---->(perm_2)---->C
  //  we want to judge A is equal with C or not
  //  suppose A = C then:
  //  1. B[i] = A[perm_1[i]]
  //  2. C[i] = B[perm_2[i]]
  //  3. combine 1 and 2 then: C[i] = A[perm_1[perm_2[i]]]
  //  which we get through 3: i = perm_1[perm_2[i]]
  //
  vector<int64_t> src_node_perm;
  (void)AttrUtils::GetListInt(src_node->GetOpDesc(), ge::PERMUTE_ATTR_PERM, src_node_perm);
  vector<int64_t> dst_node_perm;
  (void)AttrUtils::GetListInt(dst_node->GetOpDesc(), ge::PERMUTE_ATTR_PERM, dst_node_perm);

  if (src_node_perm.size() != dst_node_perm.size()) { return false; }
  for (size_t src_index = 0; src_index < src_node_perm.size(); ++src_index) {
    if (dst_node_perm[src_index] >= static_cast<int64_t>(src_node_perm.size())) { return false; }
    if (static_cast<int64_t>(src_index) != src_node_perm[dst_node_perm[src_index]]) { return false; }
  }
  return true;
}

Status TransOpSymmetryEliminationPass::EliminateTransOp(NodePtr &src_node, const OutDataAnchorPtr &src_out_anchor,
                                                        NodePtr &dst_node, const InDataAnchorPtr &dst_in_anchor) {
  // Two transform nodes can be offset like A->T1->T2->B
  // 1.Unlink T1->T2
  auto ret = src_out_anchor->Unlink(dst_in_anchor);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999",
                      "Op:%s(%s) out index:%d unlink from op:%s(%s) in index:%d failed",
                      src_out_anchor->GetOwnerNode()->GetName().c_str(),
                      src_out_anchor->GetOwnerNode()->GetType().c_str(), src_out_anchor->GetIdx(),
                      dst_in_anchor->GetOwnerNode()->GetName().c_str(),
                      dst_in_anchor->GetOwnerNode()->GetType().c_str(), dst_in_anchor->GetIdx());
    GELOGE(FAILED, "[Unlink][DataAnchor] from %s(%s)(index:%d) to %s(%s)(index:%d) failed.",
           src_out_anchor->GetOwnerNode()->GetName().c_str(),
           src_out_anchor->GetOwnerNode()->GetType().c_str(), src_out_anchor->GetIdx(),
           dst_in_anchor->GetOwnerNode()->GetName().c_str(),
           dst_in_anchor->GetOwnerNode()->GetType().c_str(), dst_in_anchor->GetIdx());
    return ret;
  }
  // 2.Link A->T2
  auto data_idx = TransOpUtil::GetTransOpDataIndex(src_node);
  auto in_anchor = src_node->GetInDataAnchor(data_idx);
  GE_CHECK_NOTNULL(in_anchor);
  GE_CHECK_NOTNULL(in_anchor->GetPeerOutAnchor());
  auto pre_normal_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
  ret = GraphUtils::AddEdge(in_anchor->GetPeerOutAnchor(), dst_in_anchor);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Add edge between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
                      pre_normal_node->GetName().c_str(), pre_normal_node->GetType().c_str(),
                      in_anchor->GetPeerOutAnchor()->GetIdx(),
                      dst_in_anchor->GetOwnerNode()->GetName().c_str(),
                      dst_in_anchor->GetOwnerNode()->GetType().c_str(), dst_in_anchor->GetIdx());
    GELOGE(FAILED, "[Add][Edge] between op:%s(%s)(index:%d) and op:%s(%s)(index:%d) failed",
           pre_normal_node->GetName().c_str(), pre_normal_node->GetType().c_str(),
           in_anchor->GetPeerOutAnchor()->GetIdx(),
           dst_in_anchor->GetOwnerNode()->GetName().c_str(),
           dst_in_anchor->GetOwnerNode()->GetType().c_str(), dst_in_anchor->GetIdx());
    return ret;
  }
  // 3.Copy in-control/data-in-control from T1->T2
  ret = GraphUtils::CopyInCtrlEdges(src_node, dst_node);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Copy in control edge from node:%s(%s) to node:%s(%s) failed",
                      src_node->GetName().c_str(), src_node->GetType().c_str(),
                      dst_node->GetName().c_str(), dst_node->GetType().c_str());
    GELOGE(FAILED, "[Copy][InCtrlEdges] from node:%s(%s) to node:%s(%s) failed",
           src_node->GetName().c_str(), src_node->GetType().c_str(),
           dst_node->GetName().c_str(), dst_node->GetType().c_str());
    return ret;
  }
  // 4.Add control edge from T1 other input to T2, like reshape second input
  for (const auto &in_node : src_node->GetInDataNodes()) {
    if (in_node->GetName() == pre_normal_node->GetName()) { continue; }
    ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add control edge between op:%s(%s) and op:%s(%s) failed",
                        in_node->GetName().c_str(), in_node->GetType().c_str(),
                        dst_node->GetName().c_str(), dst_node->GetType().c_str());
      GELOGE(FAILED, "[Add][ControlEdge] between op:%s(%s) and op:%s(%s) failed",
             in_node->GetName().c_str(), in_node->GetType().c_str(),
             dst_node->GetName().c_str(), dst_node->GetType().c_str());
      return ret;
    }
  }
  // 5.IsolateAndDelete T2, A will link to B automatically, and all control edge will also relink.
  ret = IsolateAndDeleteNode(dst_node, {0});
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Isolate and delete node:%s(%s) failed",
                      dst_node->GetName().c_str(), dst_node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[IsolateAndDelete][Node] failed, node name:%s, node type:%s ",
           dst_node->GetName().c_str(), dst_node->GetType().c_str());
    return ret;
  }
  GELOGI("Trans op symmetry eliminate successfully. Node %s has been removed.", dst_node->GetName().c_str());
  // 6.If T1 has no data out, isolate and deleted it.
  ret = RemoveTransOpWithoutOutput(pre_normal_node, src_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(ret, "[Call][RemoveTransOpWithoutOutput] for node:%s(%s) failed",
           src_node->GetName().c_str(), src_node->GetType().c_str());
    return ret;
  }
  return SUCCESS;
}
Status TransOpSymmetryEliminationPass::RemoveTransOpWithoutOutput(NodePtr &pre_node, NodePtr &trans_node) {
  if (trans_node->GetOutDataNodesSize() == 0) {
    // 6.1 Copy out control to pre normal node
    Status ret = GraphUtils::CopyOutCtrlEdges(trans_node, pre_node);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Copy out control edge from node:%s(%s) to node:%s(%s) failed",
                        trans_node->GetName().c_str(), trans_node->GetType().c_str(),
                        pre_node->GetName().c_str(), pre_node->GetType().c_str());
      GELOGE(FAILED, "[Copy][OutCtrlEdges] from %s to %s failed.", trans_node->GetName().c_str(),
             pre_node->GetName().c_str());
      return ret;
    }
    // 6.2 Isolate and delete T1
    ret = IsolateAndDeleteNode(trans_node, {});
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Isolate and delete node:%s(%s) failed",
                        trans_node->GetName().c_str(), trans_node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[IsolateAndDelete][Node] %s(%s) failed", trans_node->GetName().c_str(),
             trans_node->GetType().c_str());
      return ret;
    }
    GELOGI("Trans op symmetry eliminate successfully. Node %s has been removed.", trans_node->GetName().c_str());
  }
  return SUCCESS;
}
}  // namespace ge
