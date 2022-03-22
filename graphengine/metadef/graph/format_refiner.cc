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

#include "format_refiner.h"

#include <deque>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "graph/ref_relation.h"
#include "graph/operator.h"
#include "graph/operator_factory.h"
#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/types.h"
namespace ge {
namespace {
const size_t kDimSizeOf4D = 4UL;
const std::set<std::string> kChangeDimNodes = {PERMUTE, EXPANDDIMS, SQUEEZE};
const char_t *const kIsGraphInferred = "_is_graph_inferred";
thread_local RefRelations reflection_builder;

static graphStatus ReflectionProcess(const std::unordered_set<RefCell, RefCellHash> &reflection,
                                     std::deque<ge::NodePtr> &nodes,
                                     const ge::Format to_be_set_format) {
  for (const auto &reflection_cell : reflection) {
    const auto reflection_node = reflection_cell.node;
    const auto in_out_idx = reflection_cell.in_out_idx;
    GE_CHECK_NOTNULL(reflection_node);
    GE_CHECK_NOTNULL(reflection_node->GetOpDesc());
    if (reflection_cell.in_out == ge::NODE_IN) {
      auto desc = reflection_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(in_out_idx));
      desc.SetOriginFormat(to_be_set_format);
      desc.SetFormat(to_be_set_format);
      (void)reflection_node->GetOpDesc()->UpdateInputDesc(static_cast<uint32_t>(in_out_idx), desc);
    } else {
      auto desc = reflection_node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(in_out_idx));
      desc.SetOriginFormat(to_be_set_format);
      desc.SetFormat(to_be_set_format);
      (void)reflection_node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(in_out_idx), desc);
    }
    nodes.push_back(reflection_cell.node);
  }

  return GRAPH_SUCCESS;
}

static graphStatus BiasAddFormatFixProcess(const ge::NodePtr &graph_node_ptr) {
  // 5 meas dim num
  if (graph_node_ptr->GetType() != "BiasAdd" && graph_node_ptr->GetType() != "BiasAddGrad") {
    return GRAPH_SUCCESS;
  }
  const std::unordered_map<std::string, Format> kTfFormatFix = {
      {"NHWC", FORMAT_NDHWC},
      {"NCHW", FORMAT_NCDHW}
  };
  for (size_t i = 0UL; i < graph_node_ptr->GetOpDesc()->GetInputsSize(); i++) {
    const auto in_desc = graph_node_ptr->GetOpDesc()->MutableInputDesc(static_cast<uint32_t >(i));
    GE_CHECK_NOTNULL(in_desc);
    const auto dim_num = in_desc->MutableShape().GetDimNum();
    if (dim_num == 5UL) { // 5 means dim num
      const auto org_format = in_desc->GetOriginFormat();
      const auto key = TypeUtils::FormatToSerialString(org_format);
      const auto fixed_format = (kTfFormatFix.count(key) == 0UL) ? org_format : kTfFormatFix.at(key);
      in_desc->SetOriginFormat(fixed_format);
      in_desc->SetFormat(fixed_format);
      GELOGD("Fix the %zu'th input of node[%s]. Origin format is %s , after fixed it is %s",
             i, graph_node_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(org_format).c_str(),
             TypeUtils::FormatToSerialString(fixed_format).c_str());
    } else if (dim_num < 4UL) {
      in_desc->SetOriginFormat(FORMAT_ND);
      in_desc->SetFormat(FORMAT_ND);
      GELOGD("Fix the %zu'th input of node[%s]. Origin format is %s , after fixed it is %s",
             i, graph_node_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(FORMAT_ND).c_str(),
             TypeUtils::FormatToSerialString(FORMAT_ND).c_str());
    } else {
      // do nothing
    }
  }
  for (size_t i = 0UL; i < graph_node_ptr->GetOpDesc()->GetOutputsSize(); i++) {
    const auto out_desc = graph_node_ptr->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t >(i));
    GE_CHECK_NOTNULL(out_desc);
    if (out_desc->MutableShape().GetDimNum() != 5UL) { // 5 means dim num
      continue;
    }
    const auto org_format = out_desc->GetOriginFormat();
    const auto key = TypeUtils::FormatToSerialString(org_format);
    const auto fixed_format = (kTfFormatFix.count(key) == 0UL) ? org_format : kTfFormatFix.at(key);
    out_desc->SetOriginFormat(fixed_format);
    out_desc->SetFormat(fixed_format);
    GELOGD("fix the %zu'th output of node[%s]. Origin format is %s , after fixed it is %s",
           i, graph_node_ptr->GetName().c_str(), TypeUtils::FormatToSerialString(org_format).c_str(),
           TypeUtils::FormatToSerialString(fixed_format).c_str());
  }
  return GRAPH_SUCCESS;
}


static bool JudgeNodeIsAllNd(const OpDescPtr &one_op_desc, const ge::NodePtr &one_node_ptr,
                             std::vector<ge::NodePtr> &anchor_data_nodes) {
  bool node_is_all_nd = false;
  // consider special node save process
  // get all input desc format
  const auto input_size = static_cast<uint32_t>(one_op_desc->GetAllInputsSize());
  for (uint32_t i = 0U; i < input_size; i++) {
    // Operator pre-set format but not origin format
    GE_IF_BOOL_EXEC(one_op_desc->MutableInputDesc(i) == nullptr, continue);
    const auto input_format = one_op_desc->MutableInputDesc(i)->GetFormat();
    // Pre-save data node (only main graph data) and default infer fail
    if (one_node_ptr->GetType() == DATA) {
      anchor_data_nodes.push_back(one_node_ptr);
    }
    if ((input_format != FORMAT_ND) && (input_format != FORMAT_RESERVED)) {
      node_is_all_nd = true;
    }
  }
  // Get all output desc format
  const auto output_size = static_cast<uint32_t>(one_op_desc->GetOutputsSize());
  for (uint32_t i = 0U; i < output_size; i++) {
    GE_IF_BOOL_EXEC(one_op_desc->MutableOutputDesc(i) == nullptr, continue);
    const auto output_format = one_op_desc->MutableOutputDesc(i)->GetFormat();
    if ((output_format != FORMAT_ND) && (output_format != FORMAT_RESERVED)) {
      node_is_all_nd = true;
    }
  }
  return node_is_all_nd;
}

static graphStatus AnchorsInferProcess(std::deque<ge::NodePtr> &nodes, const OutDataAnchorPtr &out_data_anchor,
                                       const Format to_be_set_format) {
  for (const auto &peer_in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
    GE_IF_BOOL_EXEC(peer_in_data_anchor == nullptr, continue);

    const auto peer_in_data_node = peer_in_data_anchor->GetOwnerNode();
    GE_IF_BOOL_EXEC(peer_in_data_node == nullptr, continue);
    GE_IF_BOOL_EXEC(peer_in_data_node->GetOpDesc() == nullptr, continue);

    // Check format whether have been set
    const int32_t idx = peer_in_data_anchor->GetIdx();
    // do peer_out_node name and index as key to lookup reflections
    const ge::RefCell key(peer_in_data_node->GetName(), peer_in_data_node, ge::NODE_IN, idx);
    std::unordered_set<RefCell, RefCellHash> reflection;
    auto ret_status = reflection_builder.LookUpRefRelations(key, reflection);
    if (ret_status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "LookUpRefRelations failed! Node is [%s], the %d input edge",
                        (peer_in_data_node->GetName()).c_str(), idx);
      GELOGE(GRAPH_FAILED, "[Call][LookUpRefRelations] failed! Node is [%s], the %d input edge",
             (peer_in_data_node->GetName()).c_str(), idx);
      return GRAPH_FAILED;
    }
    auto ge_tensor_desc = peer_in_data_node->GetOpDesc()->GetInputDesc(static_cast<uint32_t>(idx));
    if (ge_tensor_desc.GetOriginFormat() == FORMAT_ND) {
      const auto dim_num = ge_tensor_desc.GetShape().GetDimNum();
      GE_IF_BOOL_EXEC(dim_num == 0UL,
          GELOGI("node name:%s idx:%d in is scalar. stop forward infer!", peer_in_data_node->GetName().c_str(), idx);
          continue);

      /// Check whether node to change dims ()
      /// Because some node will calculate with 5D, C dim maybe multi meaning
      const auto peer_in_data_node_type = peer_in_data_node->GetType();
      const auto iter1 = kChangeDimNodes.find(peer_in_data_node_type);
      // 4 means dims num
      if ((iter1 != kChangeDimNodes.end()) && (dim_num < 4UL)) {
        GELOGD("Node[%s] is change dim node. do not infer origin format", (peer_in_data_node->GetName()).c_str());
        continue;
      }

      if (reflection.empty()) {
        ge_tensor_desc.SetOriginFormat(to_be_set_format);
        ge_tensor_desc.SetFormat(to_be_set_format);
        (void) peer_in_data_node->GetOpDesc()->UpdateInputDesc(static_cast<uint32_t>(idx), ge_tensor_desc);

        /// Because netoutput node added before infer format ,so netoutput is end condition
        /// must set netoutput format , because saved result depend on format
        GE_IF_BOOL_EXEC(peer_in_data_node_type == NETOUTPUT, continue);

        // Call operator infer format api (forward) to get out format
        GELOGD("call infer format func[Back]!Node is [%s] ", (peer_in_data_node->GetName()).c_str());
        ret_status = peer_in_data_node->InferOriginFormat();
        GE_IF_BOOL_EXEC(ret_status != GRAPH_SUCCESS,
                        GELOGE(GRAPH_FAILED, "[Infer][Format] failed, node:%s", (peer_in_data_node->GetName()).c_str());
        return GRAPH_FAILED);
        nodes.push_back(peer_in_data_node);
      } else {
        const auto ret = ReflectionProcess(reflection, nodes, to_be_set_format);
        GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "[Reflect][Node] failed! status:%d", ret);
        return GRAPH_FAILED);
      }
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace

graphStatus FormatRefiner::RefreshConstantOutProcess(const ComputeGraphPtr &com_graph, const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(com_graph);
  GE_CHECK_NOTNULL(op_desc);
  if ((op_desc->GetType() == CONSTANTOP) && (!IsGraphInferred(com_graph))) {
    ConstGeTensorPtr tensor_value;
    if (!AttrUtils::GetTensor(op_desc, "value", tensor_value)) {
      REPORT_CALL_ERROR("E19999", "GetTensor failed, node name:%s.", op_desc->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Get][Tensor] failed, node name:%s.", op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
    GE_CHECK_NOTNULL(tensor_value);
    (void)op_desc->UpdateOutputDesc(0U, tensor_value->GetTensorDesc());
  }
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::GetAnchorPoints(const ge::ComputeGraphPtr &com_graph,
                                           std::vector<ge::NodePtr> &anchor_points,
                                           std::vector<ge::NodePtr> &anchor_data_nodes) {
  if (com_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "param graph is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] input graph is nullptr");
    return GRAPH_FAILED;
  }
  anchor_points.clear();
  // Get all anchor point nodes and switch nodes
  for (auto &one_node_ptr : com_graph->GetAllNodes()) {
    if (one_node_ptr == nullptr) {
      REPORT_INNER_ERROR("E19999", "node ptr in graph(%s) should not be null", com_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node ptr in graph(%s) should not be null", com_graph->GetName().c_str());
      return GRAPH_FAILED;
    }
    const auto one_op_desc = one_node_ptr->GetOpDesc();
    if (one_op_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "node's opdesc is nullptr，graph:%s", com_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node's opdesc is nullptr，graph:%s", com_graph->GetName().c_str());
      return GRAPH_FAILED;
    }
    graphStatus ret_status = RefreshConstantOutProcess(com_graph, one_op_desc);
    if (ret_status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][RefreshConstantOutProcess] failed! graph:%s, op:%s",
             com_graph->GetName().c_str(), one_op_desc->GetName().c_str());
      return GRAPH_FAILED;
    }

    // check anchor point valid
    if (!JudgeNodeIsAllNd(one_op_desc, one_node_ptr, anchor_data_nodes)) {
      continue;
    }
    // special process for biasAdd op
    // In tensorflow, biasAdd's format is alwayse NHWC even though set the arg
    // "data_format" to NDHWC or NCDHW.It will destroy our format-infer mechanism
    // so here do special process
    ret_status = BiasAddFormatFixProcess(one_node_ptr);
    if (ret_status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][BiasAddFormatFixProcess] failed! node:%s, graph:%s",
             one_node_ptr->GetName().c_str(), com_graph->GetName().c_str());
      return GRAPH_FAILED;
    }

    GELOGD("Node[%s] is anchor point!", one_node_ptr->GetName().c_str());
    anchor_points.push_back(one_node_ptr);
  }
  GELOGI("anchor_points number is %zu", anchor_points.size());
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::AnchorProcess(const ge::NodePtr &anchor_node) {
  if (anchor_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param anchor node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] anchor node is nullptr!");
    return GRAPH_FAILED;
  }
  std::deque<ge::NodePtr> nodes;
  nodes.push_back(anchor_node);
  while (!nodes.empty()) {
    const ge::NodePtr one_node = nodes.front();
    nodes.pop_front();
    GE_CHECK_NOTNULL(one_node);
    GE_CHECK_NOTNULL(one_node->GetOpDesc());
    graphStatus ret_status = BackInferProcess(nodes, one_node);
    if ((ret_status != GRAPH_SUCCESS) && (one_node != nullptr)) {
      GELOGE(ret_status, "[Back][InferProcess] failed! status:%d, node name [%s]",
             ret_status, one_node->GetName().c_str());
      return ret_status;
    }
    ret_status = ForwardInferProcess(nodes, one_node);
    if ((ret_status != GRAPH_SUCCESS) && (one_node != nullptr)) {
      GELOGE(ret_status, "[Forward][InferProcess] failed! status:%d, node name [%s]",
             ret_status, one_node->GetName().c_str());
      return ret_status;
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus FormatRefiner::BackInferProcess(std::deque<ge::NodePtr> &nodes, const ge::NodePtr &node) {
  GELOGD("Enter back infer process!Node is [%s]", node->GetName().c_str());
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    GELOGD("Node is [%s] [B]", node->GetName().c_str());
    const auto in_data_anchor_idx = in_anchor->GetIdx();
    const auto input_desc = node->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(in_data_anchor_idx));
    GE_IF_BOOL_EXEC(input_desc == nullptr, continue);
    const auto to_be_set_format = input_desc->GetOriginFormat();
    GE_IF_BOOL_EXEC(to_be_set_format == FORMAT_ND, GELOGD("Node [%s] [B], format is ND", node->GetName().c_str());
                    continue);
    const auto peer_out_data_anchor = in_anchor->GetPeerOutAnchor();
    GE_IF_BOOL_EXEC (peer_out_data_anchor == nullptr, continue);
    const auto peer_out_data_node = peer_out_data_anchor->GetOwnerNode();
    const int32_t idx = peer_out_data_anchor->GetIdx();
    // do peer_out_node name and index as key to lookup reflections
    const ge::RefCell key(peer_out_data_node->GetName(), peer_out_data_node, ge::NODE_OUT, idx);
    std::unordered_set<RefCell, RefCellHash> reflection;
    auto status = reflection_builder.LookUpRefRelations(key, reflection);
    GE_IF_BOOL_EXEC(status != GRAPH_SUCCESS,
                    GELOGE(GRAPH_FAILED, "[Call][LookUpRefRelations] failed! Node is [%s], the %d out edge",
                           (peer_out_data_node->GetName()).c_str(), idx); return GRAPH_FAILED);

    // Check format whether have been set
    // op_desc of node should not be null
    auto ge_tensor_desc = peer_out_data_node->GetOpDesc()->GetOutputDesc(static_cast<uint32_t>(idx));
    if (ge_tensor_desc.GetOriginFormat() == FORMAT_ND) {
      const auto dim_num = ge_tensor_desc.GetShape().GetDimNum();
      GE_IF_BOOL_EXEC(dim_num == 0UL, GELOGD("node name:%s idx:%d out is scalar. stop back infer!",
                                             peer_out_data_node->GetName().c_str(), idx); continue);

      /// Check whether node to change dims ()
      /// Because some node will calculate with 5D, C dim maybe multi meaning
      const auto peer_out_data_node_type = peer_out_data_node->GetType();
      const auto iter1 = kChangeDimNodes.find(peer_out_data_node_type);
      // 4 means dims num
      if ((iter1 != kChangeDimNodes.end()) && (dim_num < 4UL)) {
        GELOGD("Node[%s] is change dim node and shape is smaller than 4. do not modify format",
               (peer_out_data_node->GetName()).c_str());
        continue;
      }

      if (reflection.empty()) {
        ge_tensor_desc.SetOriginFormat(to_be_set_format);
        ge_tensor_desc.SetFormat(to_be_set_format);
        (void)peer_out_data_node->GetOpDesc()->UpdateOutputDesc(static_cast<uint32_t>(idx), ge_tensor_desc);

        // Call operator infer format api (forward) to get out format
        GELOGD("call infer format func[Back]!Node is [%s] ", (peer_out_data_node->GetName()).c_str());
        status = peer_out_data_node->InferOriginFormat();
        GE_IF_BOOL_EXEC(status != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "[Infer][Format] failed, Node:%s",
                                                        (peer_out_data_node->GetName()).c_str()); return GRAPH_FAILED);
        nodes.push_back(peer_out_data_node);
      } else {
        const auto ret = ReflectionProcess(reflection, nodes, to_be_set_format);
        GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS, GELOGE(GRAPH_FAILED, "[Reflect][Node] failed! status:%d", ret);
                        return GRAPH_FAILED);
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::ForwardInferProcess(std::deque<ge::NodePtr> &nodes, const ge::NodePtr &node) {
  GELOGD("Enter forward infer process!Node is [%s]", node->GetName().c_str());
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    GELOGD("Node is [%s] [F]", node->GetName().c_str());
    GE_IF_BOOL_EXEC(out_data_anchor == nullptr, continue);
    const auto out_data_anchor_idx = out_data_anchor->GetIdx();
    const auto to_be_set_format =
      node->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(out_data_anchor_idx))->GetOriginFormat();
    if (to_be_set_format == FORMAT_ND) {
      GELOGD("Node [%s] format is ND.[F]", node->GetName().c_str());
      continue;
    }
    const auto ret = AnchorsInferProcess(nodes, out_data_anchor, to_be_set_format);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }
  return GRAPH_SUCCESS;
}

void FormatRefiner::RefreshOriginFormatOfAnchor(const std::vector<ge::NodePtr> &anchor_points) {
  for (const auto &node : anchor_points) {
    if (node == nullptr || node->GetOpDesc() == nullptr) {
      continue;
    }
    for (const auto &input_desc : node->GetOpDesc()->GetAllInputsDescPtr()) {
      // single op support private format set, its origin format should not be override
      const auto ori_format = input_desc->GetOriginFormat();
      const auto format = input_desc->GetFormat();
      if (TypeUtils::IsInternalFormat(format)) {
        continue;
      }
      if ((input_desc != nullptr) && ((ori_format == FORMAT_ND) || (ori_format == FORMAT_RESERVED))) {
        input_desc->SetOriginFormat(input_desc->GetFormat());
      }
    }
    for (const auto &output_desc : node->GetOpDesc()->GetAllOutputsDescPtr()) {
      const auto ori_format = output_desc->GetOriginFormat();
      const auto format = output_desc->GetFormat();
      if (TypeUtils::IsInternalFormat(format)) {
        continue;
      }
      if ((output_desc != nullptr) && ((ori_format == FORMAT_ND) || (ori_format == FORMAT_RESERVED))) {
        output_desc->SetOriginFormat(output_desc->GetFormat());
      }
    }
  }
}

graphStatus FormatRefiner::DataNodeFormatProcess(const ComputeGraphPtr &graph,
                                                 const std::vector<ge::NodePtr> &anchor_data_nodes,
                                                 const ge::Format data_format) {
  if (!(IsGraphInferred(graph) && (!TypeUtils::IsInternalFormat(data_format)) && (data_format != FORMAT_ND))) {
    GELOGI("no necessary to do DataNodeFormatProcess. is_graph_inferred:%d, data_format:%s",
           static_cast<int32_t >(IsGraphInferred(graph)), TypeUtils::FormatToSerialString(data_format).c_str());
    return GRAPH_SUCCESS;
  }
  GELOGD("Enter DataNodeFormatProcess");
  std::vector<ge::NodePtr> uninfered_data_nodes;
  // Check and renew data nodes format
  for (const auto &data_node : anchor_data_nodes) {
    GE_CHECK_NOTNULL(data_node);
    const auto op_desc = data_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);

    const auto input_desc = op_desc->MutableInputDesc(0U);
    const auto output_desc = op_desc->MutableOutputDesc(0U);
    GE_CHECK_NOTNULL(input_desc);
    GE_CHECK_NOTNULL(output_desc);

    const auto curr_format = output_desc->GetOriginFormat();
    if (curr_format != FORMAT_ND) {
      // Data format has been infered , continue
      continue;
    }
    // keep data format be ND because lacking of defination when input shape num is smaller than 4
    if (input_desc->MutableShape().GetDimNum() < kDimSizeOf4D) {
      continue;
    }
    // Set format for un-infered data node
    input_desc->SetOriginFormat(data_format);
    input_desc->SetFormat(data_format);
    output_desc->SetOriginFormat(data_format);
    output_desc->SetFormat(data_format);
    uninfered_data_nodes.push_back(data_node);
  }
  // Reinfer format from uninfered data nodes
  for (const auto &node : uninfered_data_nodes) {
    if (node == nullptr) {
      continue;
    }
    GELOGD("data node [%s] start infer format process", node->GetName().c_str());
    const auto status = AnchorProcess(node);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][AnchorProcess] failed, status:%d, node:%s", status, node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  GELOGD("DataNodeFormatProcess success");
  return GRAPH_SUCCESS;
}

graphStatus FormatRefiner::InferOrigineFormat(const ge::ComputeGraphPtr &graph) {
  GELOGI("Enter InferOrigineFormat process!");

  // True: infered false:no-infered
  std::vector<ge::NodePtr> anchor_points;
  std::vector<ge::NodePtr> anchor_data_nodes;

  if (graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "param graph is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] input graph is nullptr");
    return GRAPH_FAILED;
  }
  // build reflection relations of boundary
  (void)reflection_builder.Clear();
  auto status = reflection_builder.BuildRefRelations(*graph);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Call][BuildRefRelations] failed, graph:%s", graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  // User set global net format
  status = GetAnchorPoints(graph, anchor_points, anchor_data_nodes);
  if (status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "GetAnchorPoints Process Faild! graph:%s", graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  // Refresh origin format of anchor point
  RefreshOriginFormatOfAnchor(anchor_points);
  // Infer format process
  for (const auto &anchor_node : anchor_points) {
    if (anchor_node == nullptr) {
      continue;
    }
    status = AnchorProcess(anchor_node);
    if (status != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Call][AnchorProcess] failed, node:%s", anchor_node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  /// According to discuss with sys-enginer, data node default format is ND.Its format
  /// should be set by infered.But if some data-node can not be got by infer, set context's
  /// format for these data nodes.
  /// Notice: ignore 5D formats
  const auto data_format = graph->GetDataFormat();
  status = DataNodeFormatProcess(graph, anchor_data_nodes, data_format);

  (void)AttrUtils::SetBool(graph, kIsGraphInferred, true);

  return status;
}

bool FormatRefiner::IsGraphInferred(const ComputeGraphPtr &graph) {
  bool is_graph_inferred = false;
  return (AttrUtils::GetBool(graph, kIsGraphInferred, is_graph_inferred) && is_graph_inferred);
}
}  // namespace ge
