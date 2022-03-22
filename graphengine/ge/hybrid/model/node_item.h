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

#ifndef GE_HYBRID_MODEL_NODE_ITEM_H_
#define GE_HYBRID_MODEL_NODE_ITEM_H_

#include <mutex>
#include <vector>
#include "external/ge/ge_api_error_codes.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/node_utils.h"
#include "framework/common/types.h"
#include "hybrid/common/tensor_value.h"

namespace ge {
namespace hybrid {
class NodeTask;
class NodeExecutor;

struct FusedSubgraph {
  std::map<int, std::vector<GeTensorDescPtr>> input_mapping;
  std::map<int, OpDescPtr> output_mapping;
  std::vector<NodePtr> nodes;
  ComputeGraphPtr graph;
};

bool IsControlFlowV2Op(const std::string &op_type);

class OptionalMutexGuard {
 public:
  OptionalMutexGuard(std::mutex *mutex, const std::string &name);
  ~OptionalMutexGuard();

 private:
  std::mutex *mu_{nullptr};
  std::string name_;
};

// for caching static information across execution
struct NodeItem {
  ~NodeItem() = default;
  static Status Create(const NodePtr &node, std::unique_ptr<NodeItem> &node_item);

  const std::string &NodeName() const {
    return node_name;
  }

  const std::string &NodeType() const {
    return node_type;
  }

  OpDescPtr GetOpDesc() const {
    return node->GetOpDesc();
  }

  bool IsInputShapeStatic(int index) const;

  GeTensorDescPtr MutableOutputDesc(int index) const;

  Status UpdateInputDesc(int index, const GeTensorDesc &tensor_desc);

  GeTensorDescPtr MutableInputDesc(int index) const;

  Status GetInputDesc(int index, GeTensorDesc &tensor_desc) const;

  Status GetOutputDesc(int index, GeTensorDesc &tensor_desc) const;

  Status GetCanonicalInputIndex(uint32_t index, int &canonical_index) const;

  bool IsControlFlowV2Op() const {
    return is_ctrl_flow_v2_op_;
  }

  bool IsControlFlowOp() const {
    return is_ctrl_flow_op_;
  }

  bool IsMergeOp() const {
    return is_merge_op_;
  }

  bool IsEnterOp() const {
    return kEnterOpTypes.count(node_type) > 0;
  }

  bool IsExitOp() const {
    return kExitOpTypes.count(node_type) > 0;
  }

  bool IsHcclOp() const;

  void SetToDynamic();

  void SetDataSend(NodeItem *node_item, int anchor_index);
  void SetCtrlSend(NodeItem *node_item, uint32_t switch_index);
  void SetMergeCtrl(NodeItem *node_item, uint32_t merge_index);
  size_t GetMergeCtrl(uint32_t merge_index) const;

  OptionalMutexGuard MutexGuard(const std::string &name) const {
    return OptionalMutexGuard(copy_mu_.get(), name + "_" + node_name);
  }

  std::string DebugString() const;

  NodePtr node;
  OpDesc *op_desc;
  int node_id = -1;
  int group = -1;
  int num_inputs = 0;
  int num_outputs = 0;
  int input_start = -1;
  int output_start = -1;
  bool is_dynamic = false;
  bool has_observer = false;
  bool has_optional_inputs = false;
  bool is_output_shape_static = true;
  bool is_need_force_infershape = false;
  UnknowShapeOpType shape_inference_type = DEPEND_IN_SHAPE;
  std::string node_name;
  std::string node_type;
  std::vector<ge::NodePtr> dependents_for_shape_inference;
  std::vector<ge::NodePtr> dependents_for_execution;
  std::set<int> to_const_output_id_list;

  // src_output_id, dst_anchor_id, dst_node
  std::vector<std::vector<std::pair<int, NodeItem *>>> outputs;

  // for linked drive
  bool is_root_node_ = false;
  bool is_ctrl_flow_v2_op_ = false;
  bool is_ctrl_flow_op_ = false;
  bool is_merge_op_ = false;
  bool is_enter_active_ = false;
  int64_t frame_index_ = -1;
  int64_t parent_frame_ = -1;
  std::set<const NodeItem *> root_ctrl_;  // Recv ctrl from root node
  std::map<const NodeItem *, std::set<int>> root_data_;  // Recv data from root node
  std::set<const NodeItem *> enter_ctrl_; // Recv ctrl from Enter node
  std::map<const NodeItem *, std::set<int>> enter_data_; // Recv data from Enter node
  std::set<const NodeItem *> data_send_;  // Send data notify to
  std::map<const NodeItem *, int> data_recv_;  // Recv data notify from
  std::set<const NodeItem *> ctrl_send_;  // Send ctrl notify to
  std::set<const NodeItem *> ctrl_recv_;  // Recv ctrl notify from
  std::vector<std::set<const NodeItem *>> switch_groups_;  // Send ctrl notify to

  std::shared_ptr<NodeTask> kernel_task;
  std::unique_ptr<FusedSubgraph> fused_subgraph;
  const NodeExecutor *node_executor = nullptr;
  std::map<int, ge::NodePtr> ref_outputs;
  std::map<int, int> reuse_inputs;
  std::map<int, int> reuse_outputs;
  int num_static_input_shapes = 0;
  bool is_profiling_report = false;

 private:
  explicit NodeItem(NodePtr node);
  Status Init();
  Status InitInputsAndOutputs();
  void ResolveOptionalInputs();
  Status ResolveDynamicState();
  Status ResolveStaticInputsAndOutputs();
  void ResolveUnknownShapeType();
  GeTensorDescPtr DoGetInputDesc(int index) const;

  std::vector<bool> is_input_shape_static_;
  std::vector<uint32_t> input_desc_indices_;
  std::shared_ptr<std::mutex> copy_mu_;
  mutable std::mutex mu_;
};
}  // namespace hybrid
}  // namespace ge

#endif // GE_HYBRID_MODEL_NODE_ITEM_H_
