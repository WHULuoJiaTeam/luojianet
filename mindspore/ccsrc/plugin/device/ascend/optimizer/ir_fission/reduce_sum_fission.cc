
/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fission/reduce_sum_fission.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
CNodePtr AddCastNode(const FuncGraphPtr &func_graph, const TypeId dst_type, const CNodePtr &input_node,
                     const bool fir_flag) {
  std::vector<AnfNodePtr> new_cast_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name()))};
  BaseShapePtr shape;
  if (fir_flag) {
    new_cast_inputs.emplace_back(input_node->inputs()[kIndex1]);
    shape = common::AnfAlgo::GetOutputDetailShape(input_node->inputs()[kIndex1], 0);
  } else {
    new_cast_inputs.emplace_back(input_node);
    shape = common::AnfAlgo::GetOutputDetailShape(input_node, 0);
  }
  CNodePtr new_cast = NewCNode(new_cast_inputs, func_graph);
  new_cast->set_scope(input_node->scope());
  new_cast->set_abstract(input_node->abstract());
  common::AnfAlgo::SetNodeAttr(kAttrDstType, MakeValue(static_cast<size_t>(dst_type)), new_cast);
  common::AnfAlgo::SetOutputTypeAndDetailShape({dst_type}, {shape}, new_cast.get());
  return new_cast;
}
}  // namespace

const BaseRef ReduceSumFission::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  auto reduce_sum_prim = std::make_shared<Primitive>(prim::kPrimReduceSum->name());
  return VectorRef({reduce_sum_prim, Xs});
}

CNodePtr AddReduceSumNode(const FuncGraphPtr &func_graph, const CNodePtr &input_node, const bool &keep_dims,
                          const std::vector<int64_t> &axis, const BaseShapePtr &out_shape) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input_node);
  auto input_type = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReduceSum->name())),
                                    input_node};
  CNodePtr reduce_sum = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(reduce_sum);
  reduce_sum->set_scope(input_node->scope());
  common::AnfAlgo::SetNodeAttr(kAttrKeepDims, MakeValue(keep_dims), reduce_sum);
  common::AnfAlgo::SetNodeAttr(kAttrAxis, MakeValue(axis), reduce_sum);
  common::AnfAlgo::SetOutputTypeAndDetailShape({input_type}, {out_shape}, reduce_sum.get());
  return reduce_sum;
}

const AnfNodePtr ReduceSumFission::Process(const FuncGraphPtr &graph, const AnfNodePtr &node, const EquivPtr &) const {
  auto input_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  if (input_type != kNumberTypeBool) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  auto keep_dims = common::AnfAlgo::GetNodeAttr<bool>(cnode, kAttrKeepDims);
  auto out_shape = common::AnfAlgo::GetOutputDetailShape(cnode, 0);
  std::vector<int64_t> inp_axis;
  auto axis_value = prim->GetAttr(kAttrAxis);
  MS_EXCEPTION_IF_NULL(axis_value);
  if (!axis_value->isa<ValueSequence>()) {
    int64_t axis = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrAxis);
    inp_axis.emplace_back(axis);
  } else {
    auto axis = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, kAttrAxis);
    inp_axis = axis;
  }
  auto cast_to_node = AddCastNode(graph, kNumberTypeFloat32, cnode, true);
  auto reduce_sum_node = AddReduceSumNode(graph, cast_to_node, keep_dims, inp_axis, out_shape);
  auto out_node = AddCastNode(graph, kNumberTypeBool, reduce_sum_node, false);
  return out_node;
}
}  // namespace opt
}  // namespace mindspore
