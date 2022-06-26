/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/format/to_format_base.h"
#include <set>
#include "ops/op_utils.h"
#include "src/common/common.h"
#include "src/common/utils.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/parser_utils.h"
#include "nnacl/op_base.h"

using mindspore::lite::NHWC_SHAPE;
namespace mindspore {
namespace opt {
STATUS ToFormatBase::GenNewInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, const std::vector<int> &perm,
                                 bool before, size_t index) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  AnfNodePtr trans_input = before ? cnode->input(index) : cnode;
  std::string trans_name = before ? cnode->fullname_with_scope() + "_pre_" + std::to_string(index - 1)
                                  : cnode->fullname_with_scope() + "_post";
  auto trans_cnode = opt::GenTransposeNode(func_graph, trans_input, perm, trans_name);
  MS_ASSERT(trans_cnode != nullptr);
  if (DecideWhetherInferShapeForNewNode()) {
    auto status = node_infer_shape_->InferShape(trans_cnode);
    if (status != lite::RET_OK && status != lite::RET_INFER_INVALID) {
      MS_LOG(ERROR) << "infer generated trans node failed.";
      return lite::RET_ERROR;
    }
  } else {
    auto abstract = trans_input->abstract();
    if (abstract != nullptr) {
      trans_cnode->set_abstract(abstract->Clone());
    }
  }
  auto trans_prim = GetValueNode<PrimitivePtr>(trans_cnode->input(0));
  MS_ASSERT(trans_prim != nullptr);
  if (perm == kNC2NH) {
    trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
  } else if (perm == kNH2NC) {
    trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
  }
  MS_ASSERT(manager_ != nullptr);
  if (before) {
    manager_->SetEdge(cnode, index, trans_cnode);
  } else {
    if (!manager_->Replace(cnode, trans_cnode)) {
      MS_LOG(ERROR) << "replace old node failed, please check.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS ToFormatBase::ModifyCNode(const CNodePtr &cnode) {
  MS_ASSERT(cnode != nullptr);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  if (prim == nullptr) {
    MS_LOG(ERROR) << "current node's prim is nullptr, " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  auto insert_pos = sensitive_ops_[prim->name()];
  if (insert_pos.empty() || std::find(insert_pos.begin(), insert_pos.end(), 1) != insert_pos.end()) {
    prim->AddAttr(ops::kFormat, MakeValue<int64_t>(format_));
  }
  auto abstract_base = cnode->abstract();
  MS_ASSERT(abstract_base != nullptr);
  std::vector<AbstractBasePtr> abstracts;
  if (utils::isa<abstract::AbstractTuple>(abstract_base)) {
    auto abstract_tuple = utils::cast<abstract::AbstractTuplePtr>(abstract_base);
    abstracts = abstract_tuple->elements();
  } else {
    abstracts.push_back(abstract_base);
  }
  for (auto &abstract : abstracts) {
    ShapeVector shape;
    if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed, " << cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
    if (shape.size() != kInputSizeFour) {
      MS_LOG(DEBUG) << "shape don't need to modify.";
      continue;
    }
    if (format_ == mindspore::NCHW) {
      ShapeVector transfer_shape = {shape[0], shape[kInputIndexThree], shape[1], shape[kInputIndexTwo]};
      abstract->set_shape(std::make_shared<abstract::Shape>(transfer_shape));
    } else {
      ShapeVector transfer_shape = {shape[0], shape[kInputIndexTwo], shape[kInputIndexThree], shape[1]};
      abstract->set_shape(std::make_shared<abstract::Shape>(transfer_shape));
    }
  }
  return lite::RET_OK;
}

STATUS ToFormatBase::InsertPreTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                        const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  auto prim_node = cnode->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  MS_ASSERT(prim != nullptr);
  if (sensitive_ops_.find(prim->name()) == sensitive_ops_.end()) {
    MS_LOG(ERROR) << "op don't meet condition.";
    return lite::RET_ERROR;
  }
  auto insert_index = sensitive_ops_.at(prim->name());
  if (insert_index.empty()) {
    if (opt::CheckPrimitiveType(cnode, prim::kPrimResizeGrad) && prim->GetAttr(ops::kMethod) != nullptr &&
        GetValue<int64_t>(prim->GetAttr(ops::kMethod)) == static_cast<int64_t>(mindspore::ResizeMethod::NEAREST)) {
      insert_index.push_back(1);
    } else {
      for (size_t i = 1; i < cnode->size(); ++i) {
        insert_index.push_back(i);
      }
    }
  }
  for (auto &index : insert_index) {
    if (GenNewInput(func_graph, cnode, perm, true, index) != lite::RET_OK) {
      MS_LOG(ERROR) << "generate a new input failed.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS ToFormatBase::InsertPostTransNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                         const std::vector<int> &perm) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  if (!cnode->abstract()->isa<abstract::AbstractTuple>()) {
    if (GenNewInput(func_graph, cnode, perm, false) != lite::RET_OK) {
      MS_LOG(ERROR) << "generate a new input failed.";
      return lite::RET_ERROR;
    }
  } else {
    auto node_users = manager_->node_users()[cnode];
    for (auto &node_user : node_users) {
      auto post_node = node_user.first;
      CNodePtr tuple_get_item = nullptr;
      if (!opt::CheckPrimitiveType(post_node, prim::kPrimTupleGetItem)) {
        if (!train_flag_) {
          MS_LOG(ERROR) << "post node is invalid.";
          return lite::RET_ERROR;
        } else {
          tuple_get_item = opt::GenTupleGetItemNode(func_graph, cnode, 0);
          manager_->Replace(cnode, tuple_get_item, post_node);
          post_node = tuple_get_item;
        }
      }
      if (manager_->node_users()[post_node].empty()) {
        continue;
      }
      auto post_cnode = post_node->cast<CNodePtr>();
      if (GenNewInput(func_graph, post_cnode, perm, false) != lite::RET_OK) {
        MS_LOG(ERROR) << "generate a new input failed.";
        return lite::RET_ERROR;
      }
      if (tuple_get_item != nullptr) {
        if (!manager_->Replace(tuple_get_item, tuple_get_item->input(1))) {
          MS_LOG(ERROR) << "replace old node failed. please check.";
          return lite::RET_ERROR;
        }
      }
    }
  }
  return lite::RET_OK;
}

bool ToFormatBase::DecideWhetherHandleGraphInput(const FuncGraphPtr &func_graph, const ParameterPtr &input,
                                                 const ShapeVector &shape) {
  MS_ASSERT(func_graph != nullptr && input != nullptr);
  if (shape.size() != kInputSizeFour) {
    return false;
  }
  MS_ASSERT(manager_ != nullptr);
  auto node_users = manager_->node_users()[input];
  for (auto &node_user : node_users) {
    auto post_node = node_user.first;
    if (!utils::isa<CNode>(post_node)) {
      continue;
    }
    auto post_cnode = post_node->cast<CNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(post_cnode->input(0));
    MS_ASSERT(prim != nullptr);
    if (prim->GetAttr(ops::kFormat) != nullptr) {
      auto node_format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
      if (node_format == format_) {
        MS_LOG(DEBUG) << "this graph input don't need to change.";
        return false;
      }
    }
  }
  return true;
}

STATUS ToFormatBase::HandleGraphInput(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto graph_input = func_graph->get_inputs();
  for (auto &input : graph_input) {
    auto input_param = input->cast<ParameterPtr>();
    MS_ASSERT(input_param != nullptr);
    auto abstract = input_param->abstract();
    MS_ASSERT(abstract != nullptr);
    ShapeVector shape;
    if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed." << input->fullname_with_scope();
      return lite::RET_ERROR;
    }
    if (!DecideWhetherHandleGraphInput(func_graph, input_param, shape)) {
      continue;
    }
    ShapeVector transfer_shape;
    if (format_ == mindspore::NCHW) {
      transfer_shape = {shape[0], shape[kInputIndexThree], shape[1], shape[kInputIndexTwo]};
    } else {
      transfer_shape = {shape[0], shape[kInputIndexTwo], shape[kInputIndexThree], shape[1]};
    }
    CNodePtr trans_cnode;
    if (format_ == mindspore::NCHW) {
      trans_cnode = opt::GenTransposeNode(func_graph, input, kNC2NH, input->fullname_with_scope() + "_nc2nh");
    } else {
      trans_cnode = opt::GenTransposeNode(func_graph, input, kNH2NC, input->fullname_with_scope() + "_nh2nc");
    }
    if (trans_cnode == nullptr) {
      MS_LOG(ERROR) << "create transpose cnode failed.";
      return lite::RET_ERROR;
    }
    auto trans_prim = GetValueNode<PrimitivePtr>(trans_cnode->input(0));
    MS_ASSERT(trans_prim != nullptr);
    if (format_ == mindspore::NCHW) {
      trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
    } else {
      trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
    }
    trans_cnode->set_abstract(abstract->Clone());
    abstract->set_shape(std::make_shared<abstract::Shape>(transfer_shape));
    if (!manager_->Replace(input, trans_cnode)) {
      MS_LOG(ERROR) << "replace old node failed, please check.";
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

STATUS ToFormatBase::HandleGraphNode(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_ASSERT(func_graph != nullptr && cnode != nullptr);
  opt::TransTypePair trans_info;
  if (GetTransNodeFormatType(cnode, &trans_info) != lite::RET_OK) {
    MS_LOG(ERROR) << "obtain node's transferring format type failed, " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (trans_info.pre_ == opt::kNONE || trans_info.post_ == opt::kNONE) {
    return lite::RET_NO_CHANGE;
  }
  auto before_perm = trans_info.pre_ == opt::kNHWC2NCHW ? kNH2NC : kNC2NH;
  auto after_perm = trans_info.post_ == opt::kNCHW2NHWC ? kNC2NH : kNH2NC;
  if (InsertPreTransNode(func_graph, cnode, before_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "insert pre node failed." << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (opt::CheckPrimitiveType(cnode, prim::kPrimAdam) || opt::CheckPrimitiveType(cnode, prim::kPrimSGD)) {
    return lite::RET_OK;
  }
  if (ModifyCNode(cnode) != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust cnode's output shape failed, " << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  if (InsertPostTransNode(func_graph, cnode, after_perm) != lite::RET_OK) {
    MS_LOG(ERROR) << "insert post node failed." << cnode->fullname_with_scope();
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

bool ToFormatBase::BasicProcess(const FuncGraphPtr &func_graph, bool main_graph) {
  MS_ASSERT(func_graph != nullptr);
  manager_->AddFuncGraph(func_graph);
  auto node_list = TopoSort(func_graph->get_return());
  int status;
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (IsSpecialType(cnode)) {
      continue;
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimIf) || opt::CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      if (!BasicProcess(sub_func_graph, false)) {
        MS_LOG(ERROR) << "process sub graph failed.";
        return false;
      }
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return false;
      }
      if (!BasicProcess(sub_func_graph, false)) {
        MS_LOG(ERROR) << "process sub graph failed.";
        return false;
      }
      continue;
    }
    status = HandleGraphNode(func_graph, cnode);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "handle node failed.";
      return false;
    }
  }
  if (main_graph) {
    status = HandleGraphInput(func_graph);
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "handle graph input failed.";
      return false;
    }
  }
  return true;
}

STATUS ToFormatBase::ConvWeightFormatTrans(const FuncGraphPtr &graph, std::set<AnfNodePtr> *has_visited) {
  MS_ASSERT(graph != nullptr && has_visited != nullptr);
  manager_->AddFuncGraph(graph);
  auto node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (CheckPrimitiveType(node, prim::kPrimIf) || CheckPrimitiveType(node, prim::kPrimWhile)) {
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(1));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return lite::RET_NULL_PTR;
      }
      if (ConvWeightFormatTrans(sub_func_graph, has_visited) != lite::RET_OK) {
        MS_LOG(ERROR) << "transform conv weight format failed.";
        return lite::RET_ERROR;
      }
      sub_func_graph = GetValueNode<FuncGraphPtr>(cnode->input(kInputIndexTwo));
      if (sub_func_graph == nullptr) {
        lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
        return lite::RET_NULL_PTR;
      }
      if (ConvWeightFormatTrans(sub_func_graph, has_visited) != lite::RET_OK) {
        MS_LOG(ERROR) << "transform conv weight format failed.";
        return lite::RET_ERROR;
      }
      continue;
    }
    if (!IsWeightNodeSensitive(cnode)) {
      continue;
    }
    if (has_visited->find(node) != has_visited->end()) {
      continue;
    }
    has_visited->insert(node);
    schema::Format src_format = schema::Format_NUM_OF_FORMAT;
    schema::Format dst_format = schema::Format_NUM_OF_FORMAT;
    if (DecideConvWeightSrcAndDstFormat(cnode, &src_format, &dst_format) != lite::RET_OK) {
      MS_LOG(ERROR) << "weight's src format and dst format get failed.";
      return lite::RET_ERROR;
    }
    auto status = lite::UnifyConvWeightFormat(graph, cnode, src_format, dst_format, has_visited);
    if (status != lite::RET_OK) {
      MS_LOG(ERROR) << "unify conv weight failed, current node name is " << cnode->fullname_with_scope();
      return status;
    }
  }
  return lite::RET_OK;
}

bool ToFormatBase::Run(const FuncGraphPtr &func_graph) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  if (format_ != mindspore::NHWC && format_ != mindspore::NCHW) {
    MS_LOG(ERROR) << "format transferring only support nc2nh or nh2nc.";
    return false;
  }
  manager_ = Manage(func_graph, true);
  if (manager_ == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  node_infer_shape_ = std::make_shared<NodeInferShape>(fmk_type_, train_flag_);
  if (node_infer_shape_ == nullptr) {
    MS_LOG(ERROR) << "create NodeInferShape object failed.";
    return false;
  }
  std::set<AnfNodePtr> has_visited;
  auto status = ConvWeightFormatTrans(func_graph, &has_visited);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Conv2D weight FormatTrans failed: " << status;
    return false;
  }
  SetSensitiveOps();
  if (!BasicProcess(func_graph, true)) {
    MS_LOG(ERROR) << "transfer format failed.";
    return false;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
