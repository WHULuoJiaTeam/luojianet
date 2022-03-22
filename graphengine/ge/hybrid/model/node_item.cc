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

#include "hybrid/model/node_item.h"

#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "hybrid/executor/worker/shape_inference_engine.h"
#include "hybrid/node_executor/node_executor.h"

namespace ge {
namespace hybrid {
namespace {
const uint8_t kMaxTransCount = 3;
const uint8_t kTransOpIoSize = 1;
const char *const kAttrNameOriginalFusionGraph = "_original_fusion_graph";
const char *const kNodeTypeRetVal = "_RetVal";
const std::set<std::string> kControlOpTypes{
    IF, STATELESSIF, CASE, WHILE, STATELESSWHILE
};

const std::set<std::string> kControlFlowOpTypes{
    STREAMACTIVE, STREAMSWITCH, STREAMSWITCHN, ENTER, REFENTER, NEXTITERATION, REFNEXTITERATION, EXIT, REFEXIT,
    LABELGOTO, LABELGOTOEX, LABELSWITCH, LABELSWITCHBYINDEX
};

const std::set<std::string> kMergeOpTypes{
    MERGE, REFMERGE, STREAMMERGE
};

bool IsEnterFeedNode(NodePtr node) {
  // For: Enter -> node
  // For: Enter -> Cast -> node
  // For: Enter -> TransData -> Cast -> node
  for (uint8_t i = 0; i < kMaxTransCount; ++i) {
    if (kEnterOpTypes.count(NodeUtils::GetNodeType(node)) > 0) {
      GELOGD("Node[%s] is Enter feed node.", node->GetName().c_str());
      return true;
    }

    const auto all_nodes = node->GetInDataNodes();
    if (all_nodes.size() != kTransOpIoSize || node->GetAllInDataAnchorsSize() != kTransOpIoSize) {
      return false;
    }
    node = all_nodes.at(0);
  }
  return false;
}

Status ParseInputMapping(Node &node, OpDesc &op_desc, FusedSubgraph &fused_subgraph) {
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(FAILED, "[Invoke][GetInt][%s] Failed to get attr [%s]",
           op_desc.GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
    REPORT_CALL_ERROR("E19999", "[%s] Failed to get attr [%s]",
                      op_desc.GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return FAILED;
  }

  for (auto &node_and_anchor : node.GetOutDataNodesAndAnchors()) {
    auto dst_op_desc = node_and_anchor.first->GetOpDesc();
    GE_CHECK_NOTNULL(dst_op_desc);
    auto in_idx = node_and_anchor.second->GetIdx();
    auto tensor_desc = dst_op_desc->MutableInputDesc(in_idx);
    fused_subgraph.input_mapping[static_cast<int>(parent_index)].emplace_back(tensor_desc);
    GELOGD("Input[%u] mapped to [%s:%u]", parent_index, dst_op_desc->GetName().c_str(), in_idx);
  }

  return SUCCESS;
}

Status ParseOutputMapping(const OpDescPtr &op_desc, FusedSubgraph &fused_subgraph) {
  uint32_t parent_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, parent_index)) {
    GELOGE(FAILED, "[Invoke][GetInt][%s] Failed to get attr [%s]",
           op_desc->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
    REPORT_CALL_ERROR("E19999", "[%s] Failed to get attr [%s].",
                      op_desc->GetName().c_str(), ATTR_NAME_PARENT_NODE_INDEX.c_str());
    return FAILED;
  }

  fused_subgraph.output_mapping.emplace(static_cast<int>(parent_index), op_desc);
  return SUCCESS;
}

Status ParseFusedSubgraph(NodeItem &node_item) {
  if (!node_item.op_desc->HasAttr(kAttrNameOriginalFusionGraph)) {
    return SUCCESS;
  }

  GELOGI("[%s] Start to parse fused subgraph.", node_item.node_name.c_str());
  auto fused_subgraph = std::unique_ptr<FusedSubgraph>(new(std::nothrow)FusedSubgraph());
  GE_CHECK_NOTNULL(fused_subgraph);

  ComputeGraphPtr fused_graph;
  (void) AttrUtils::GetGraph(*node_item.op_desc, kAttrNameOriginalFusionGraph, fused_graph);
  GE_CHECK_NOTNULL(fused_graph);

  fused_graph->SetGraphUnknownFlag(true);
  fused_subgraph->graph = fused_graph;
  GE_CHK_GRAPH_STATUS_RET(fused_graph->TopologicalSorting());

  for (auto &node : fused_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const std::string node_type = NodeUtils::GetNodeType(node);
    if (node_type == DATA) {
      GE_CHK_GRAPH_STATUS_RET(ParseInputMapping(*node, *op_desc, *fused_subgraph));
    } else if (node_type == kNodeTypeRetVal) {
      GE_CHK_GRAPH_STATUS_RET(ParseOutputMapping(op_desc, *fused_subgraph));
    } else {
      fused_subgraph->nodes.emplace_back(node);
    }
  }

  node_item.fused_subgraph = std::move(fused_subgraph);
  GELOGI("[%s] Done parsing fused subgraph successfully.", node_item.NodeName().c_str());
  return SUCCESS;
}
}  // namespace

bool IsControlFlowV2Op(const std::string &op_type) {
  return kControlOpTypes.count(op_type) > 0;
}

NodeItem::NodeItem(NodePtr node) : node(std::move(node)) {
  this->op_desc = this->node->GetOpDesc().get();
  this->node_name = this->node->GetName();
  this->node_type = this->node->GetType();
}

Status NodeItem::Create(const NodePtr &node, std::unique_ptr<NodeItem> &node_item) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  std::unique_ptr<NodeItem> instance(new(std::nothrow)NodeItem(node));
  GE_CHECK_NOTNULL(instance);
  GE_CHK_STATUS_RET(instance->Init(), "[Invoke][Init]Failed to init NodeItem [%s] .", node->GetName().c_str());
  node_item = std::move(instance);
  return SUCCESS;
}

void NodeItem::ResolveOptionalInputs() {
  if (op_desc->GetAllInputsSize() != op_desc->GetInputsSize()) {
    has_optional_inputs = true;
    for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
      const auto &input_desc = op_desc->MutableInputDesc(i);
      if (input_desc == nullptr) {
        GELOGD("[%s] Input[%zu] is optional and invalid", NodeName().c_str(), i);
      } else {
        input_desc_indices_.emplace_back(static_cast<uint32_t>(i));
      }
    }
  }
}

Status NodeItem::InitInputsAndOutputs() {
  GE_CHECK_LE(op_desc->GetInputsSize(), INT32_MAX);
  GE_CHECK_LE(op_desc->GetOutputsSize(), INT32_MAX);
  num_inputs = static_cast<int>(op_desc->GetInputsSize());
  num_outputs = static_cast<int>(op_desc->GetOutputsSize());
  if (AttrUtils::GetInt(op_desc, ::ge::ATTR_STAGE_LEVEL, group)) {
    GELOGD("[%s] Got stage level from op_desc = %d", op_desc->GetName().c_str(), group);
  } else {
    if (node->GetOwnerComputeGraph() != nullptr) {
      if (AttrUtils::GetInt(node->GetOwnerComputeGraph(), ::ge::ATTR_STAGE_LEVEL, group)) {
        GELOGD("[%s] Got stage level from parent graph = %d", op_desc->GetName().c_str(), group);
      } else {
        auto parent_node = node->GetOwnerComputeGraph()->GetParentNode();
        if ((parent_node != nullptr) && (AttrUtils::GetInt(parent_node->GetOpDesc(), ::ge::ATTR_STAGE_LEVEL, group))) {
          GELOGD("[%s] Got stage level from parent node = %d", op_desc->GetName().c_str(), group);
        } else {
          GELOGD("[%s] Node do not set stage level", op_desc->GetName().c_str());
        }
      }
    }
  }
  ResolveOptionalInputs();
  return SUCCESS;
}

Status NodeItem::ResolveDynamicState() {
  (void) AttrUtils::GetBool(op_desc, ATTR_NAME_FORCE_UNKNOWN_SHAPE, is_dynamic);
  GELOGD("Node name is %s, dynamic state is %d.", this->node_name.c_str(), is_dynamic);
  if (!is_dynamic) {
    GE_CHK_STATUS_RET(NodeUtils::GetNodeUnknownShapeStatus(*node, is_dynamic),
                      "[Invoke][GetNodeUnknownShapeStatus][%s] Failed to get shape status.",
                      node->GetName().c_str());
  }
  return SUCCESS;
}

Status NodeItem::ResolveStaticInputsAndOutputs() {
  for (int i = 0; i < num_inputs; ++i) {
    // Data has unconnected input but set by framework
    if (node_type != DATA) {
      int origin_index = i;
      if (has_optional_inputs) {
        origin_index = input_desc_indices_[i];
      }
      auto in_data_anchor = node->GetInDataAnchor(origin_index);
      GE_CHECK_NOTNULL(in_data_anchor);

      // If no node was connected to the current input anchor
      // increase num_static_input_shapes in case dead wait in ShapeInferenceState::AwaitShapesReady
      if (in_data_anchor->GetPeerOutAnchor() == nullptr ||
          in_data_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
        num_static_input_shapes++;
        is_input_shape_static_.push_back(true);
        GELOGW("[%s] Peer node of input[%d] is empty", NodeName().c_str(), i);
        continue;
      }
    }
    const auto &input_desc = MutableInputDesc(i);
    GE_CHECK_NOTNULL(input_desc);
    if (input_desc->MutableShape().IsUnknownShape()) {
      is_input_shape_static_.push_back(false);
    } else {
      num_static_input_shapes++;
      is_input_shape_static_.push_back(true);
      GELOGD("[%s] The shape of input[%d] is static. shape = [%s]",
             NodeName().c_str(), i, input_desc->MutableShape().ToString().c_str());
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    const auto &output_desc = op_desc->MutableOutputDesc(i);
    GE_CHECK_NOTNULL(output_desc);
    if (output_desc->MutableShape().IsUnknownShape()) {
      is_output_shape_static = false;
      break;
    }
  }

  if (is_output_shape_static) {
    GE_CHK_STATUS_RET_NOLOG(ShapeInferenceEngine::CalcOutputTensorSizes(*this));
  }
  return SUCCESS;
}

void NodeItem::ResolveUnknownShapeType() {
  if (IsControlFlowV2Op() || (is_dynamic && node_type == PARTITIONEDCALL)) {
    shape_inference_type = DEPEND_COMPUTE;
  } else {
    int32_t unknown_shape_type_val = 0;
    (void) AttrUtils::GetInt(op_desc, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
    shape_inference_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  }
}

Status NodeItem::Init() {
  is_ctrl_flow_v2_op_ = ge::hybrid::IsControlFlowV2Op(node_type);
  is_ctrl_flow_op_ = kControlFlowOpTypes.count(node_type) > 0;
  is_merge_op_ = kMergeOpTypes.count(node_type) > 0;
  is_root_node_ = node->GetInAllNodes().empty();
  GE_CHK_STATUS_RET_NOLOG(InitInputsAndOutputs());
  GE_CHK_STATUS_RET_NOLOG(ResolveDynamicState());
  ResolveUnknownShapeType();
  if (is_dynamic) {
    GE_CHK_STATUS_RET_NOLOG(ResolveStaticInputsAndOutputs());
    GE_CHK_STATUS_RET(ParseFusedSubgraph(*this),
                      "[Invoke][ParseFusedSubgraph][%s] Failed to parse fused subgraph", node_name.c_str());
  }
  copy_mu_ = MakeShared<std::mutex>();
  GE_CHECK_NOTNULL(copy_mu_);

  return SUCCESS;
}

bool NodeItem::IsHcclOp() const {
  return NodeExecutorManager::GetInstance().ResolveExecutorType(*node) == NodeExecutorManager::ExecutorType::HCCL;
}

std::string NodeItem::DebugString() const {
  std::stringstream ss;
  ss << "Node: ";
  ss << "id = " << node_id;
  ss << ", name = [" << node->GetName();
  ss << "], type = " << node->GetType();
  ss << ", is_dynamic = " << (is_dynamic ? "True" : "False");
  ss << ", is_output_static = " << (is_output_shape_static ? "True" : "False");
  ss << ", unknown_shape_op_type = " << shape_inference_type;
  ss << ", stage = " << group;
  ss << ", input_start = " << input_start;
  ss << ", num_inputs = " << num_inputs;
  ss << ", output_start = " << output_start;
  ss << ", num_outputs = " << num_outputs;
  ss << ", dependent_nodes = [";
  for (const auto &dep_node : dependents_for_shape_inference) {
    ss << dep_node->GetName() << ", ";
  }
  ss << "]";
  int index = 0;
  for (auto &items : outputs) {
    ss << ", output[" << index++ << "]: ";
    for (auto &item : items) {
      ss << "(" << item.second->NodeName() << ":" << item.first << "), ";
    }
  }

  return ss.str();
}

void NodeItem::SetToDynamic() {
  num_static_input_shapes = 0;
  is_dynamic = true;
  for (size_t i = 0; i < is_input_shape_static_.size(); ++i) {
    is_input_shape_static_[i] = false;
  }
  if (kernel_task != nullptr && !kernel_task->IsSupportDynamicShape()) {
    GELOGD("[%s] Dynamic shape is not supported, clear node task.", node_name.c_str());
    kernel_task = nullptr;
  }
}

GeTensorDescPtr NodeItem::DoGetInputDesc(int index) const {
  if (!has_optional_inputs) {
    return op_desc->MutableInputDesc(static_cast<uint32_t>(index));
  }

  if (index < 0 || index >= num_inputs) {
    GELOGE(PARAM_INVALID, "[Check][Param:index][%s] Invalid input index, num inputs = %d, index = %d",
           node_name.c_str(), num_inputs, index);
    REPORT_INNER_ERROR("E19999", "Invalid input index, node:%s num inputs = %d, index = %d",
                       node_name.c_str(), num_inputs, index);
    return nullptr;
  }

  return op_desc->MutableInputDesc(input_desc_indices_[index]);
}

GeTensorDescPtr NodeItem::MutableInputDesc(int index) const {
  std::lock_guard<std::mutex> lk(mu_);
  return DoGetInputDesc(index);
}

Status NodeItem::GetInputDesc(int index, GeTensorDesc &tensor_desc) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto input_desc = DoGetInputDesc(index);
  GE_CHECK_NOTNULL(input_desc);
  tensor_desc = *input_desc;
  return SUCCESS;
}

Status NodeItem::GetOutputDesc(int index, GeTensorDesc &tensor_desc) const {
  std::lock_guard<std::mutex> lk(mu_);
  auto output_desc = op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
  GE_CHECK_NOTNULL(output_desc);
  tensor_desc = *output_desc;
  return SUCCESS;
}

GeTensorDescPtr NodeItem::MutableOutputDesc(int index) const {
  std::lock_guard<std::mutex> lk(mu_);
  return op_desc->MutableOutputDesc(static_cast<uint32_t>(index));
}

Status NodeItem::UpdateInputDesc(int index, const GeTensorDesc &tensor_desc) {
  std::lock_guard<std::mutex> lk(mu_);
  auto input_desc = DoGetInputDesc(index);
  GE_CHECK_NOTNULL(input_desc);
  *input_desc = tensor_desc;
  return SUCCESS;
}

Status NodeItem::GetCanonicalInputIndex(uint32_t index, int &canonical_index) const {
  if (!has_optional_inputs) {
    canonical_index = index;
    return SUCCESS;
  }

  auto iter = std::find(input_desc_indices_.begin(), input_desc_indices_.end(), index);
  if (iter == input_desc_indices_.end()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Param:index]input index:%u not in input_desc_indices_, check Invalid, node:%s",
           index, node_name.c_str());
    REPORT_INNER_ERROR("E19999", "input index:%u not in input_desc_indices_, check Invalid, node:%s",
                       index, node_name.c_str());
    return INTERNAL_ERROR;
  }

  canonical_index = static_cast<int>(iter - input_desc_indices_.begin());
  GELOGD("[%s] Canonicalize input index from [%u] to [%d]", node_name.c_str(), index, canonical_index);
  return SUCCESS;
}

bool NodeItem::IsInputShapeStatic(int index) const {
  if (!is_dynamic) {
    return true;
  }

  if (static_cast<size_t>(index) >= is_input_shape_static_.size()) {
    GELOGE(PARAM_INVALID, "[Check][Param:index]Input index(%d) out of range: [0, %zu)",
           index, is_input_shape_static_.size());
    REPORT_INNER_ERROR("E19999", "Input index(%d) out of range: [0, %zu).", index, is_input_shape_static_.size());
    return false;
  }

  return is_input_shape_static_[index];
}

void NodeItem::SetDataSend(NodeItem *node_item, int anchor_index) {
  data_send_.emplace(node_item);
  node_item->data_recv_[this] = anchor_index;
  if (is_root_node_) {
    auto &data_anchors = node_item->root_data_[this];
    data_anchors.emplace(anchor_index);
  }
  // If Enter feed Not Merge, take as root Node.
  if (IsEnterFeedNode(node) && (node_item->node_type != STREAMMERGE)) {
    auto &data_anchors = node_item->enter_data_[this];
    data_anchors.emplace(anchor_index);
  }
  GELOGI("Node[%s] will control node[%s]", NodeName().c_str(), node_item->NodeName().c_str());
}

void NodeItem::SetCtrlSend(NodeItem *node_item, uint32_t switch_index) {
  if (switch_index < switch_groups_.size()) {
    auto &switch_group = switch_groups_[switch_index];
    switch_group.emplace(node_item);
  } else {
    ctrl_send_.insert(node_item);
  }

  node_item->ctrl_recv_.emplace(this);
  if (is_root_node_) {
    node_item->root_ctrl_.emplace(this);
  }
  // If Enter feed control signal, take as root Node.
  if (IsEnterFeedNode(node) && (node_item->node_type != STREAMMERGE && node_item->node_type != STREAMACTIVE)) {
    node_item->enter_ctrl_.emplace(this);
  }
  GELOGI("Node[%s] will control node[%s]", NodeName().c_str(), node_item->NodeName().c_str());
}

void NodeItem::SetMergeCtrl(NodeItem *node_item, uint32_t merge_index) {
  if (merge_index >= switch_groups_.size()) {
    GELOGE(FAILED, "[%s] group size: %zu, merge index: %u", NodeName().c_str(), switch_groups_.size(), merge_index);
    return;
  }

  // this is StreamMerge node, node_item is StreamActive node.
  auto &switch_group = switch_groups_[merge_index];
  switch_group.emplace(node_item);

  node_item->ctrl_send_.emplace(this);
  GELOGI("Node[%s] will control node[%s]", node_item->NodeName().c_str(), NodeName().c_str());
}

size_t NodeItem::GetMergeCtrl(uint32_t merge_index) const {
  return ((node_type == STREAMMERGE) && (merge_index < switch_groups_.size())) ? switch_groups_[merge_index].size() : 0;
}

OptionalMutexGuard::OptionalMutexGuard(std::mutex *mutex, const string &name) : mu_(mutex), name_(name) {
  if (mu_ != nullptr) {
    GELOGD("lock for %s", name_.c_str());
    mu_->lock();
  }
}

OptionalMutexGuard::~OptionalMutexGuard() {
  if (mu_ != nullptr) {
    GELOGD("unlock for %s", name_.c_str());
    mu_->unlock();
    mu_ = nullptr;
  }
}
}  // namespace hybrid
}  // namespace ge
