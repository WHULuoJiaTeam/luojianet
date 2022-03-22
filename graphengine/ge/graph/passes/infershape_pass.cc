/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
 *+
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

#include "graph/passes/infershape_pass.h"
#include "common/util/error_manager/error_manager.h"
#include "framework/common/debug/ge_log.h"
#include "analyzer/analyzer.h"
#include "framework/common/util.h"
#include "graph/shape_refiner.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "common/omg_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

#include "external/graph/operator_factory.h"

namespace ge {
namespace {
constexpr int kSwitchExitAnchorIndex = 0;
constexpr int kSwitchPredAnchorIndex = 1;
void SerialShapeRange(const GeTensorDescPtr &desc, std::string &desc_str) {
  desc_str += "[";
  std::vector<std::pair<int64_t, int64_t>> shape_range;
  (void)desc->GetShapeRange(shape_range);
  for (const auto &pair : shape_range) {
    desc_str += "{";
    desc_str += std::to_string(pair.first) + "," + std::to_string(pair.second);
    desc_str += "},";
  }
  desc_str += "]";
  shape_range.clear();
  (void)desc->GetOriginShapeRange(shape_range);
  for (const auto &pair : shape_range) {
    desc_str += ",{";
    desc_str += std::to_string(pair.first) + "," + std::to_string(pair.second);
    desc_str += "},";
  }
}
void UpdateShapeAndDType(const GeTensorDescPtr &src, GeTensorDescPtr &dst) {
  dst->SetOriginShape(src->GetOriginShape());
  dst->SetShape(src->GetShape());
  dst->SetDataType(src->GetDataType());
  dst->SetOriginDataType(src->GetOriginDataType());
  vector<pair<int64_t, int64_t>> src_shape_range;
  src->GetShapeRange(src_shape_range);
  dst->SetShapeRange(src_shape_range);
  dst->SetOriginShapeRange(src_shape_range);
  ge::TensorUtils::SetRealDimCnt(*dst, static_cast<uint32_t>(src->GetShape().GetDims().size()));
}
}  // namespace

std::string InferShapePass::SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const {
  std::stringstream ss;
  ss << "(shape:[" << tensor_desc->MutableShape().ToString() << "]),";
  ss << "(format:" << TypeUtils::FormatToSerialString(tensor_desc->GetFormat()) << "),";
  ss << "(dtype:" << TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()) << "),";
  ss << "(origin_shape:" << tensor_desc->GetOriginShape().ToString() << "),";
  ss << "(origin_format:" << TypeUtils::FormatToSerialString(tensor_desc->GetOriginFormat()) << "),";
  ss << "(origin_dtype:" << TypeUtils::DataTypeToSerialString(tensor_desc->GetOriginDataType()) << "),";
  string range_str;
  SerialShapeRange(tensor_desc, range_str);
  ss << "(shape_range:" << range_str << ")";
  return ss.str();
}
Status InferShapePass::SuspendV1LoopExitNodes(const NodePtr &node) {
  if (node->GetType() != SWITCH) {
    return SUCCESS;
  }
  auto pred_node = NodeUtils::GetInDataNodeByIndex(*node, kSwitchPredAnchorIndex);
  GE_CHECK_NOTNULL(pred_node);
  if (pred_node->GetType() != LOOPCOND) {
    return SUCCESS;
  }

  for (const auto &anchor_2_node : NodeUtils::GetOutDataNodesWithAnchorByIndex(*node, kSwitchExitAnchorIndex)) {
    GELOGI("Found v1 loop when infershape, suspend Exit node %s, type %s.", anchor_2_node.second->GetName().c_str(),
           anchor_2_node.second->GetType().c_str());
    auto &suspend_nodes = graphs_2_suspend_nodes_[GetCurrentGraphName()];
    if (suspend_nodes.nodes_set.insert(anchor_2_node.second).second) {
      suspend_nodes.nodes.push(anchor_2_node.second);
      AddNodeSuspend(anchor_2_node.second);
    }
  }
  return SUCCESS;
}

Status InferShapePass::Infer(NodePtr &node) {
  auto ret = InferShapeAndType(node);
  if (ret != GRAPH_SUCCESS) {
    auto graph = node->GetOwnerComputeGraph();
    GE_CHECK_NOTNULL(graph);
    auto root_graph = ge::GraphUtils::FindRootGraph(graph);
    GE_CHECK_NOTNULL(root_graph);
    analyzer::DataInfo analyze_info{root_graph->GetSessionID(), root_graph->GetGraphID(),
                                    analyzer::INFER_SHAPE, node, "InferShapeFailed!"};
    (void)Analyzer::GetInstance()->DoAnalyze(analyze_info);
    (void)Analyzer::GetInstance()->SaveAnalyzerDataToFile(root_graph->GetSessionID(),
                                                          root_graph->GetGraphID());
    REPORT_CALL_ERROR("E19999", "Call InferShapeAndType for node:%s(%s) failed", node->GetName().c_str(),
                      node->GetType().c_str());
    GELOGE(GE_GRAPH_INFERSHAPE_FAILED, "[Call][InferShapeAndType] for node:%s(%s) failed", node->GetName().c_str(),
           node->GetType().c_str());
    return GE_GRAPH_INFERSHAPE_FAILED;
  }
  return SUCCESS;
}

graphStatus InferShapePass::InferShapeAndType(NodePtr &node) {
  auto ret = SuspendV1LoopExitNodes(node);
  if (ret != SUCCESS) {
    GELOGE(ret, "Suspend V1 loop exit nodes failed.");
    return ret;
  }
  bool is_unknown_graph = node->GetOwnerComputeGraph()->GetGraphUnknownFlag();
  auto opdesc = node->GetOpDesc();
  if (node->Verify() != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Verifying %s failed.", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Call][Verify] Verifying %s failed.", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  Operator op = OpDescUtils::CreateOperatorFromNode(node);

  if (!is_unknown_graph) {
    auto inference_context = ShapeRefiner::CreateInferenceContext(node);
    GE_CHECK_NOTNULL(inference_context);
    std::vector<AscendString> marks;
    inference_context->GetMarks(marks);
    GELOGD("create context for node:%s, marks %zu", node->GetName().c_str(), marks.size());
    op.SetInferenceContext(inference_context);
  }

  graphStatus status = CallInferShapeFunc(node, op);
  if (status != GRAPH_NODE_NEED_REPASS && status != GRAPH_PARAM_INVALID && status != GRAPH_SUCCESS) {
    // node like netoutput return param_invalid, but valid ?
    return GE_GRAPH_INFERSHAPE_FAILED;
  }
  UpdateCurNodeOutputDesc(node);
  if (!is_unknown_graph) {
    auto ctx_after_infer = op.GetInferenceContext();
    if (ctx_after_infer != nullptr) {
      std::vector<AscendString> marks;
      ctx_after_infer->GetMarks(marks);
      GELOGD("[%s] after infershape. mark:%zu", node->GetName().c_str(), marks.size());
      if (!ctx_after_infer->GetOutputHandleShapesAndTypes().empty() || !marks.empty()) {
        GELOGD("[%s] set inference context after. mark:%zu", node->GetName().c_str(),
               marks.size());
        ShapeRefiner::PushToContextMap(node, ctx_after_infer);
      }
    }
  }

  return (status == GRAPH_NODE_NEED_REPASS) ? GRAPH_NODE_NEED_REPASS : GRAPH_SUCCESS;
}

void InferShapePass::UpdateCurNodeOutputDesc(NodePtr &node) {
  auto op_desc = node->GetOpDesc();
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    auto output_tensor = op_desc->MutableOutputDesc(out_anchor->GetIdx());
    GE_IF_BOOL_EXEC(output_tensor == nullptr, continue);
    GE_IF_BOOL_EXEC(output_tensor->MutableShape().GetDims().empty(),
                    output_tensor->SetOriginShape(output_tensor->GetShape()));

    ge::TensorUtils::SetRealDimCnt(*output_tensor, static_cast<uint32_t>(output_tensor->GetOriginShape().GetDims()
      .size()));
    output_tensor->SetOriginDataType(output_tensor->GetDataType());
    // set output origin shape range
    std::vector<std::pair<int64_t, int64_t>> range;
    (void)output_tensor->GetShapeRange(range);
    output_tensor->SetOriginShapeRange(range);
    GELOGD("node name is %s, origin shape is %ld, origin format is %s, origin data type is %s",
           node->GetName().c_str(), output_tensor->GetOriginShape().GetShapeSize(),
           TypeUtils::FormatToSerialString(output_tensor->GetOriginFormat()).c_str(),
           TypeUtils::DataTypeToSerialString(output_tensor->GetOriginDataType()).c_str());
  }
}

bool InferShapePass::SameTensorDesc(const GeTensorDescPtr &src, const GeTensorDescPtr &dst) {
  // check shape range
  vector<std::pair<int64_t, int64_t>> src_shape_range;
  vector<std::pair<int64_t, int64_t>> dst_shape_range;
  src->GetShapeRange(src_shape_range);
  dst->GetShapeRange(dst_shape_range);
  if (src_shape_range.size() != dst_shape_range.size()) {
    GELOGI("Src shape range size is %zu, dst shape range size is %zu, not same.", src_shape_range.size(),
           dst_shape_range.size());
    return false;
  }
  for (size_t i = 0; i < src_shape_range.size(); ++i) {
    if (src_shape_range[i].first != dst_shape_range[i].first ||
        src_shape_range[i].second != dst_shape_range[i].second) {
      GELOGI("Current dim %zu. Src shape range is [%lu-%lu], dst shape range is [%lu-%lu], not same.",
             i, src_shape_range[i].first, src_shape_range[i].second, dst_shape_range[i].first, dst_shape_range[i].second);
      return false;
    }
  }

  // check shape
  auto src_shape = src->GetShape();
  auto dst_shape = dst->GetShape();
  if (src_shape.GetDims() != dst_shape.GetDims() || src->GetOriginShape().GetDims() != dst->GetOriginShape().GetDims() ||
      src->GetDataType() != dst->GetDataType() || src->GetOriginDataType() != dst->GetOriginDataType()) {
    GELOGD(
        "Src shape is %s, origin_shape is %s, data_type is %s, origin data_type is %s; "
        "Dst shape is %s, origin_shape is %s, data_type is %s, original data_type is %s, not same.",
        src_shape.ToString().c_str(), src->GetOriginShape().ToString().c_str(),
        TypeUtils::DataTypeToSerialString(src->GetDataType()).c_str(),
        TypeUtils::DataTypeToSerialString(src->GetOriginDataType()).c_str(), dst_shape.ToString().c_str(),
        dst->GetOriginShape().ToString().c_str(), TypeUtils::DataTypeToSerialString(dst->GetDataType()).c_str(),
        TypeUtils::DataTypeToSerialString(dst->GetOriginDataType()).c_str());
    return false;
  }
  return true;
}

graphStatus InferShapePass::UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) {
  changed = false;
  if (SameTensorDesc(src, dst)) {
    GELOGD("Peer dst tensor_desc is same as src tensor_desc. No need update.");
    return SUCCESS;
  }

  changed = true;
  UpdateShapeAndDType(src, dst);
  GELOGD(
      "UpdatePeerInputDesc from src Node: shape: [%s], datatype: %s, original datatype is %s."
      "To dst Node: shape: [%s], datatype: %s, original datatype is %s.",
      src->GetShape().ToString().c_str(), TypeUtils::DataTypeToSerialString(src->GetDataType()).c_str(),
      TypeUtils::DataTypeToSerialString(src->GetOriginDataType()).c_str(), dst->GetShape().ToString().c_str(),
      TypeUtils::DataTypeToSerialString(dst->GetDataType()).c_str(),
      TypeUtils::DataTypeToSerialString(dst->GetOriginDataType()).c_str());
  return SUCCESS;
}

graphStatus InferShapePass::CallInferShapeFunc(NodePtr &node, Operator &op) {
  auto op_desc = node->GetOpDesc();
  const auto &op_type = op_desc->GetType();
  auto ret = op_desc->CallInferFunc(op);
  if (ret == GRAPH_PARAM_INVALID) {
    // Op ir no infer func, try to get infer func from operator factory

    auto node_op = ge::OperatorFactory::CreateOperator("node_op", op_desc->GetType().c_str());
    if (node_op.IsEmpty()) {
      GELOGW("get op from OperatorFactory fail. opType: %s", op_type.c_str());
      return ret;
    }

    GELOGD("get op from OperatorFactory success. opType: %s", op_type.c_str());
    auto temp_op_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
    node_op.BreakConnect();
    if (temp_op_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "GetOpDescFromOperator failed, return nullptr.");
      GELOGE(GRAPH_FAILED, "[Get][OpDesc] temp op desc is null");
      return GRAPH_FAILED;
    }
    if (!op_desc->UpdateInputName(temp_op_desc->GetAllInputName())) {
      GELOGW("InferShapeAndType UpdateInputName failed");
      for (const auto &out_desc : op_desc->GetAllOutputsDescPtr()) {
        if (out_desc != nullptr && out_desc->GetShape().GetDims().empty()) {
          break;
        }
        return GRAPH_SUCCESS;
      }
    }
    if (!op_desc->UpdateOutputName(temp_op_desc->GetAllOutputName())) {
      GELOGW("InferShapeAndType UpdateOutputName failed");
    }
    op_desc->AddInferFunc(temp_op_desc->GetInferFunc());
    ret = op_desc->CallInferFunc(op);
    GELOGI("op CallInferFunc second. ret: %u", ret);
  }
  return ret;
}

graphStatus InferShapePass::UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src, GeTensorDescPtr &dst) {
  GELOGD("Enter update parent node shape for class branch op process");
  // check sub_graph shape.If not same ,do unknown shape process
  auto ref_out_tensor = src.at(0);
  ge::GeShape &ref_out_tensor_shape = ref_out_tensor->MutableShape();
  for (auto &tensor : src) {
    if (ref_out_tensor->GetDataType() != tensor->GetDataType()) {
      REPORT_INNER_ERROR("E19999", "Does not support diff dtype among all ref output, shape:%s",
                         ref_out_tensor_shape.ToString().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] node does not support diff dtype output");
      return GRAPH_FAILED;
    }
    auto shape = tensor->MutableShape();
    if (shape.GetDims().size() != ref_out_tensor_shape.GetDims().size()) {
      GELOGD("Shape from subgraph size: %lu, ref_out_tensor_shape size: %lu", shape.GetShapeSize(),
             ref_out_tensor_shape.GetShapeSize());
      ref_out_tensor_shape = GeShape(UNKNOWN_RANK);
      break;
    }
    for (size_t j = 0; j < ref_out_tensor_shape.GetDims().size(); j++) {
      if (ref_out_tensor_shape.GetDim(j) == shape.GetDim(j)) {
        continue;
      }
      GELOGD("j: %zu ,shape from subgraph size: %lu, ref_out_tensor_shape size: %lu", j, shape.GetShapeSize(),
             ref_out_tensor_shape.GetShapeSize());
      (void)ref_out_tensor_shape.SetDim(j, UNKNOWN_DIM);
    }
  }
  UpdateShapeAndDType(ref_out_tensor, dst);
  return GRAPH_SUCCESS;
}
graphStatus InferShapePass::UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                                  GeTensorDescPtr &dst) {
  // check sub_graph shape. Get max for update.
  if (src.empty()) {
    GELOGI("Src subgraph shape is empty.");
    return SUCCESS;
  }

  int64_t max_size = 0;
  size_t max_shape_index = 0;
  auto &ref_out_tensor = src.at(0);
  for (size_t j = 0; j < src.size(); ++j) {
    auto &tensor = src.at(j);
    if (ref_out_tensor->GetDataType() != tensor->GetDataType()) {
      REPORT_INNER_ERROR("E19999", "node does not support diff dtype among all ref output");
      GELOGE(GRAPH_FAILED, "[Check][Param] node does not support diff dtype among all ref output");
      return GRAPH_FAILED;
    }

    auto shape = tensor->MutableShape();
    int64_t size = 1;
    for (auto dim : shape.GetDims()) {
      if (dim != 0 && INT64_MAX / dim < size) {
        REPORT_INNER_ERROR("E19999", "The shape:%s size overflow", shape.ToString().c_str());
        GELOGE(PARAM_INVALID, "[Check][Overflow] The shape size overflow");
        return PARAM_INVALID;
      }
      size *= dim;
    }

    if (size > max_size) {
      max_size = size;
      max_shape_index = j;
    }
  }
  UpdateShapeAndDType(src.at(max_shape_index), dst);
  return GRAPH_SUCCESS;
}
Status InferShapePass::OnSuspendNodesLeaked() {
  auto iter = graphs_2_suspend_nodes_.find(GetCurrentGraphName());
  if (iter == graphs_2_suspend_nodes_.end()) {
    GELOGI("Current graph %s no suspend node.", GetCurrentGraphName().c_str());
    return SUCCESS;
  }
  if (!iter->second.nodes.empty()) {
    AddNodeResume(iter->second.PopSuspendedNode());
  }
  return SUCCESS;
}
}  // namespace ge
