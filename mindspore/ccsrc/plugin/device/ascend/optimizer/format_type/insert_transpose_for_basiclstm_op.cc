/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/format_type/insert_transpose_for_basiclstm_op.h"
#include <memory>
#include <vector>
#include "include/common/utils/utils.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/kernel_info.h"
#include "kernel/oplib/oplib.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
const BaseRef InsertTranspose::DefinePattern() const {
  std::shared_ptr<Var> V = std::make_shared<CondVar>(UnVisited);
  std::shared_ptr<Var> Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

CNodePtr Insert(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::string &op_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(cnode);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  CNodePtr new_node = nullptr;

  std::vector<AnfNodePtr> transpose_inputs;
  auto prim = std::make_shared<Primitive>(prim::kPrimTranspose->name());
  transpose_inputs.push_back(NewValueNode(prim));

  if (op_name == kBasicLSTMCellInputGradOpName) {
    auto origin_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, 1);
    auto origin_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, 1);
    auto dst_shape = {origin_shape[1], origin_shape[0]};
    auto is_dynamic = AnfUtils::IsShapeDynamic(dst_shape);

    transpose_inputs.push_back(common::AnfAlgo::GetInputNode(cnode, 1));
    CNodePtr transpose = func_graph->NewCNode(transpose_inputs);
    MS_EXCEPTION_IF_NULL(transpose);
    if (is_dynamic) {
      auto shape = {SizeToLong(origin_shape[1]), SizeToLong(origin_shape[0])};
      auto max_shape = common::AnfAlgo::GetInputMaxShape(cnode, 1);
      auto min_shape = common::AnfAlgo::GetInputMinShape(cnode, 1);
      auto shape_tmp1 = {min_shape[1], min_shape[0]};
      auto shape_tmp2 = {max_shape[1], max_shape[0]};
      BaseShapePtr base_shape = std::make_shared<abstract::Shape>(shape, shape_tmp1, shape_tmp2);
      common::AnfAlgo::SetOutputTypeAndDetailShape({origin_type}, {base_shape}, transpose.get());
    } else {
      common::AnfAlgo::SetOutputInferTypeAndShape({origin_type}, {dst_shape}, transpose.get());
    }
    common::AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(std::vector<int64_t>{1, 0}), transpose);
    common::AnfAlgo::SetNodeInput(cnode, transpose, 1);
    if (kernel_graph == nullptr) {
      new_node = std::make_shared<CNode>(*cnode);
    } else {
      new_node = kernel_graph->NewCNode(cnode);
    }
  } else if (op_name == kBasicLSTMCellWeightGradOpName) {
    std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
    size_t out_num = common::AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t output_idx = 0; output_idx < out_num; output_idx++) {
      auto tuple_getitem = CreatTupleGetItemNode(func_graph, cnode, output_idx);
      auto origin_shape = common::AnfAlgo::GetOutputInferShape(cnode, output_idx);
      if (origin_shape.size() > 1 && output_idx == 0) {
        auto dtype = common::AnfAlgo::GetOutputInferDataType(cnode, output_idx);
        transpose_inputs.push_back(tuple_getitem);
        CNodePtr transpose = func_graph->NewCNode(transpose_inputs);
        MS_EXCEPTION_IF_NULL(transpose);
        if (AnfUtils::IsShapeDynamic(origin_shape)) {
          auto dst_shape = {SizeToLong(origin_shape[0]), SizeToLong(origin_shape[1])};
          auto min_shape = common::AnfAlgo::GetOutputMinShape(cnode, output_idx);
          auto max_shape = common::AnfAlgo::GetOutputMaxShape(cnode, output_idx);
          auto shape_tmp1 = {min_shape[0], min_shape[1]};
          auto shape_tmp2 = {max_shape[0], max_shape[1]};
          BaseShapePtr base_shape = std::make_shared<abstract::Shape>(dst_shape, shape_tmp1, shape_tmp2);
          common::AnfAlgo::SetOutputTypeAndDetailShape({dtype}, {base_shape}, transpose.get());
        } else {
          auto dst_shape = {origin_shape[0], origin_shape[1]};
          common::AnfAlgo::SetOutputInferTypeAndShape({dtype}, {dst_shape}, transpose.get());
        }
        common::AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(std::vector<int64_t>{1, 0}), transpose);
        make_tuple_inputs.push_back(transpose);
      } else {
        make_tuple_inputs.push_back(tuple_getitem);
      }
    }
    new_node = func_graph->NewCNode(make_tuple_inputs);
  }
  return new_node;
}

const AnfNodePtr InsertTranspose::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  CNodePtr new_node = nullptr;
  if (op_name == kBasicLSTMCellInputGradOpName || op_name == kBasicLSTMCellWeightGradOpName) {
    new_node = Insert(func_graph, cnode, op_name);
  }
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
