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

#ifndef GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
#define GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
#include <map>
#include <set>
#include "common/transop_util.h"
#include "external/graph/graph.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/util/variable_accelerate_ctrl.h"
#include "inc/graph_pass.h"

namespace ge {
namespace variable_op {
struct NodeDesc {
  ge::GeTensorDesc input;
  ge::GeTensorDesc output;
  bool is_update = false;
};
}  // namespace variable_op
class VariableOpPass : public GraphPass {
 public:
  explicit VariableOpPass(VarAccelerateCtrl *ctrl) : var_accelerate_ctrl_(ctrl) {}

  ~VariableOpPass() override = default;

  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  Status DealFusion(const ge::NodePtr &src_node);

  Status CheckVariableRefLegally(const ge::NodePtr &var_node, bool &is_var_legally);

  Status UpdateVarAndRefOutputFormatInfo(const GeTensorDesc &final_output, const ge::NodePtr &node);

  Status GenerateVariableVariableRefMap(const ComputeGraphPtr &compute_graph);

  Status CheckVarAndVarRefAreAlike(const NodePtr &var_node, const NodePtr &var_ref_node,
                                   bool &is_var_and_var_ref_alike);

  bool IsOpDescSame(const GeTensorDescPtr &op_desc_a, const GeTensorDescPtr &op_desc_b);

  Status CheckTransNodeAreInverse(const NodePtr &node_a, const NodePtr &node_b, bool &is_trans_node_inverse);

  void CopyVariableFormatDataTypeAndShape(const GeTensorDesc &src_tensor_desc, GeTensorDesc &dst_tensor_desc);

  Status CheckSameAndTransOp(const ge::NodePtr &var_nodem, bool &is_matched, VarTransRoad &fusion_road);

  Status CheckIfCouldBeOptimized(const ge::NodePtr &node, bool &flag, VarTransRoad &fusion_road);

  Status FusionIfNeed(const NodePtr &var, VarTransRoad &fusion_road);

  Status UpdateIOFormatInfo(const GeTensorDesc &final_output, std::set<NodePtr> &nodes);
  Status RenewVarDesc(ge::ComputeGraphPtr &graph);
  Status RenewVarDesc(uint64_t session_id, const NodePtr &node, const VarTransRoad &fusion_road);

  std::map<NodePtr, std::set<NodePtr>> var_and_var_ref_map_;

  VarAccelerateCtrl *var_accelerate_ctrl_;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_VARIABLE_OP_PASS_H_
