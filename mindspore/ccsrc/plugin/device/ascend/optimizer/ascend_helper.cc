/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include <set>
#include "runtime/device/ms_device_shape_transfer.h"
#include "utils/ms_utils.h"
#include "utils/check_convert_utils.h"
#include "backend/common/optimizer/helper.h"
#include "include/common/utils/utils.h"
#include "runtime/device/kernel_info.h"
#include "kernel/oplib/oplib.h"
#include "kernel/common_utils.h"
#include "base/core_ops.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_graph.h"
#include "utils/ms_context.h"
#include "utils/trace_base.h"
namespace mindspore {
namespace opt {
using KernelBuildInfoBuilder = kernel::KernelBuildInfo::KernelBuildInfoBuilder;
namespace {
bool NeedInsertTransData(const std::vector<size_t> &origin_shape, const std::string &format) {
  bool shape_check = origin_shape.size() > 1 || (origin_shape.size() == 1 && origin_shape[0] % kCubeSize != 0);
  return kCommonFormatSet.find(format) == kCommonFormatSet.end() && (shape_check || format == kOpFormat_ND_RNN_BIAS);
}

AnfNodePtr CreateReshapeNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node, const AnfNodePtr &orig_node,
                             const KernelSelectPtr &kernel_select, const std::vector<size_t> &dst_shape) {
  std::vector<AnfNodePtr> trans_inputs;
  auto prim = std::make_shared<Primitive>(prim::kPrimReshape->name());
  (void)trans_inputs.emplace_back(NewValueNode(prim));
  (void)trans_inputs.emplace_back(input_node);
  auto reshape = NewCNode(trans_inputs, func_graph, {orig_node});
  MS_EXCEPTION_IF_NULL(reshape);
  common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(input_node, 0)}, {dst_shape},
                                              reshape.get());
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), reshape);
  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(dst_shape), reshape);
  reshape->set_scope(input_node->scope());
  kernel_select->SelectKernel(reshape);
  return reshape;
}

void SetTransNodeAttr(const CNodePtr &trans_node) {
  MS_EXCEPTION_IF_NULL(trans_node);
  auto trans_opname = common::AnfAlgo::GetCNodeName(trans_node);
  if (trans_opname == kTransDataOpName || trans_opname == kTransDataRNNOpName) {
    std::string input_format = AnfAlgo::GetInputFormat(trans_node, 0);
    std::string output_format = AnfAlgo::GetOutputFormat(trans_node, 0);
    if (input_format == kOpFormat_DEFAULT) {
      input_format = common::AnfAlgo::GetCNodeName(trans_node) == kTransDataOpName ? kOpFormat_NCHW : kOpFormat_ND;
    }
    if (output_format == kOpFormat_DEFAULT) {
      output_format = common::AnfAlgo::GetCNodeName(trans_node) == kTransDataOpName ? kOpFormat_NCHW : kOpFormat_ND;
    }
    common::AnfAlgo::SetNodeAttr(kAttrSrcFormat, MakeValue(input_format), trans_node);
    common::AnfAlgo::SetNodeAttr(kAttrDstFormat, MakeValue(output_format), trans_node);
  }
}

void ReFreshInferShape(const AnfNodePtr &trans_node, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(trans_node);
  MS_EXCEPTION_IF_NULL(node);
  auto real_input_node = common::AnfAlgo::VisitKernelWithReturnType(node, 0).first;
  if (!real_input_node->isa<CNode>()) {
    return;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(real_input_node);
  if (op_name == kBasicLSTMCellWeightGradOpName &&
      common::AnfAlgo::GetCNodeName(trans_node) == prim::kPrimReshape->name()) {
    auto shape = common::AnfAlgo::GetPrevNodeOutputInferShape(trans_node, 0);
    auto type = common::AnfAlgo::GetPrevNodeOutputInferDataType(trans_node, 0);
    common::AnfAlgo::SetOutputInferTypeAndShape({type}, {{shape[0], shape[1]}}, node.get());
  }
}

void SetGroupAttr(const ParameterPtr &param, const AnfNodePtr &out_trans, const AnfNodePtr &in_trans,
                  const std::string &dest_format) {
  MS_EXCEPTION_IF_NULL(param);
  auto fz_group = param->fracz_group();
  // in the scenario of gradient freezing or infer while training, the parameters are already set with
  // fracz_group in first graph, so the inserted transdata will trans format from FracZwithgroup(param)
  // to default and default to FracZwithoutgroup(cnode, such as Conv2D, Opt). These paired TransDatas are
  // not set with groups attr and cannot be eliminated in EliminateReduntantOp. So to solve this problem,
  // set the groups and fracz_group attr here for these paired TransData nodes.
  if (fz_group > 1) {
    if (out_trans->isa<CNode>()) {
      // if has transdata after parameter
      common::AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(fz_group), out_trans);
      common::AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(fz_group), out_trans);
    }
    if (dest_format == kOpFormat_FRAC_Z) {
      common::AnfAlgo::SetNodeAttr(kAttrGroups, MakeValue(fz_group), in_trans);
      common::AnfAlgo::SetNodeAttr(kAttrFracZGroup, MakeValue(fz_group), in_trans);
    }
  }
}

AnfNodePtr GetTransInputNodePtr(const FuncGraphPtr &func_graph, const CNodePtr &node, size_t index,
                                const KernelSelectPtr &kernel_select) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto input_node = common::AnfAlgo::GetInputNode(node, index);
  if (HasAbstractMonad(input_node)) {
    // No transfer for monad inputs.
    return input_node;
  }
  auto node_with_index = common::AnfAlgo::VisitKernel(input_node, 0);
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  auto real_input = node_with_index.first;
  if (real_input->isa<ValueNode>() || real_input->isa<Parameter>()) {
    input_node = InsertTransOpForOutput(func_graph, input_node, input_node, kernel_select);
    MS_EXCEPTION_IF_NULL(input_node);
    common::AnfAlgo::SetNodeInput(node, input_node, index);
  }
  std::vector<size_t> origin_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, index);
  std::string dest_format = AnfAlgo::GetInputFormat(node, index);
  if (NeedInsertTransData(origin_shape, dest_format)) {
    MS_LOG(DEBUG) << node->DebugString() << "Insert transdata " << AnfAlgo::GetInputFormat(node, index)
                  << " To DefaultFormat , index: " << index;
    auto transdata = AddTransOpNodeToGraph(func_graph, node, kernel_select, index, true);
    if (real_input->isa<Parameter>()) {
      SetGroupAttr(real_input->cast<ParameterPtr>(), input_node, transdata, dest_format);
    }
    return transdata;
  }
  return input_node;
}

AnfNodePtr InsertTransOpForSingleOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const KernelSelectPtr &kernel_select) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::string output_format = AnfAlgo::GetOutputFormat(node, 0);
  std::vector<size_t> origin_shape = common::AnfAlgo::GetOutputInferShape(node, 0);
  if (output_format == kOpFormat_NC1KHKWHWC0) {
    MS_LOG(EXCEPTION) << "Got the hw format " << output_format << "when insert the transdata node "
                      << node->DebugString() << trace::DumpSourceLines(node);
  }
  if (NeedInsertTransData(origin_shape, output_format)) {
    MS_LOG(DEBUG) << "Inserted transdata " << output_format << " to default , index :0";
    return AddTransOpNodeToGraph(func_graph, node, kernel_select, 0, false);
  }
  return node;
}

AnfNodePtr InsertTransOpForMultipleOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &orig_node,
                                          const AnfNodePtr &node, const KernelSelectPtr &kernel_select) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto update_states = common::AnfAlgo::GetUpdateStateUsers(manager, orig_node);
  for (auto &update_state : update_states) {
    manager->SetEdge(update_state.first, update_state.second, node);
  }
  if (manager->node_users()[orig_node].empty()) {
    return node;
  }
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  size_t out_num = common::AnfAlgo::GetOutputTensorNum(node);
  for (size_t output_idx = 0; output_idx < out_num; ++output_idx) {
    std::string output_format = AnfAlgo::GetOutputFormat(node, output_idx);
    if (output_format == kOpFormat_NC1KHKWHWC0) {
      MS_LOG(EXCEPTION) << "Got the special format" << output_format << " when insert the transdata node "
                        << node->DebugString() << trace::DumpSourceLines(node);
    }
    auto tuple_getitem = CreatTupleGetItemNode(func_graph, node, output_idx);
    std::vector<size_t> origin_shape = common::AnfAlgo::GetOutputInferShape(node, output_idx);
    if (NeedInsertTransData(origin_shape, output_format)) {
      auto trans_op = AddTransOpNodeToGraph(func_graph, tuple_getitem, kernel_select, 0, false);
      if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(node, output_idx)) {
        kernel_graph->ReplaceInternalOutput(node, trans_op, output_idx, 0);
      }
      make_tuple_inputs.push_back(trans_op);
    } else {
      // No need insert trans op.
      make_tuple_inputs.push_back(tuple_getitem);
    }
  }
  AnfNodePtr make_tuple = func_graph->NewCNode(make_tuple_inputs);
  return make_tuple;
}
}  // namespace

AnfNodePtr AddTransOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select, size_t insert_index, bool is_insert_input) {
  AnfNodePtr trans_node = nullptr;
  CNodePtr trans_data = nullptr;
  MS_EXCEPTION_IF_NULL(node);
  // Init
  std::string default_format = kOpFormat_DEFAULT;
  AnfNodePtr input_node = is_insert_input ? common::AnfAlgo::GetInputNode(node->cast<CNodePtr>(), insert_index) : node;
  std::string input_format = is_insert_input ? default_format : AnfAlgo::GetOutputFormat(node, insert_index);
  std::string dst_format = is_insert_input ? AnfAlgo::GetInputFormat(node, insert_index) : default_format;
  std::string padding_axis = is_insert_input ? AnfAlgo::GetInputReshapeType(node, insert_index)
                                             : AnfAlgo::GetOutputReshapeType(node, insert_index);
  auto input_node_out_shape = is_insert_input ? common::AnfAlgo::GetPrevNodeOutputInferShape(node, insert_index)
                                              : common::AnfAlgo::GetOutputInferShape(input_node, insert_index);
  std::string spec_format = is_insert_input ? dst_format : input_format;
  bool need_padding = trans::IsNeedPadding(spec_format, input_node_out_shape.size());
  std::string trans_opname = (spec_format == kOpFormat_FRACTAL_ZN_RNN || spec_format == kOpFormat_ND_RNN_BIAS)
                               ? prim::kPrimTransDataRNN->name()
                               : prim::kPrimTransData->name();
  auto orig_node = node;
  if (!is_insert_input && node->isa<CNode>() && common::AnfAlgo::GetCNodeName(node) == prim::kTupleGetItem) {
    auto cnode = node->cast<CNodePtr>();
    orig_node = cnode->input(kRealInputNodeIndexInTupleGetItem);
  }

  if (!need_padding) {
    // don't need padding insert transdata only
    trans_data = NewTransOpNode(func_graph, input_node, orig_node, kernel_select, need_padding, trans_opname);
    trans_node = trans_data;
  } else if (is_insert_input) {
    // if need padding & is input need insert a transdata
    // reshape[padding shape] -> transdata[padding shape] -> node
    auto padding_shape = trans::PaddingShape(input_node_out_shape, AnfAlgo::GetInputFormat(node, insert_index),
                                             AnfAlgo::GetInputReshapeType(node, insert_index), node);
    auto reshape_node = CreateReshapeNode(func_graph, input_node, orig_node, kernel_select, padding_shape);
    trans_data = NewTransOpNode(func_graph, reshape_node, orig_node, kernel_select, need_padding, trans_opname);
    trans_node = trans_data;
    trans_data->set_abstract(input_node->abstract());
  } else {
    // if need padding & is output need insert a transdata
    // node -> transdata[padding shape] -> reshape[ori_shape]
    trans_data = NewTransOpNode(func_graph, input_node, orig_node, kernel_select, need_padding, trans_opname);
    auto reshape_node = CreateReshapeNode(func_graph, trans_data, orig_node, kernel_select, input_node_out_shape);
    trans_node = reshape_node;
  }
  if (trans_opname == prim::kPrimTransDataRNN->name()) {
    common::AnfAlgo::CopyNodeAttr(kAttrHiddenSize, node, trans_data);
    common::AnfAlgo::CopyNodeAttr(kAttrInputSize, node, trans_data);
  }
  if (spec_format == kOpFormat_FRAC_Z && orig_node->isa<CNode>() &&
      common::AnfAlgo::HasNodeAttr(kAttrFracZGroup, orig_node->cast<CNodePtr>())) {
    common::AnfAlgo::CopyNodeAttr(kAttrGroups, orig_node, trans_data);
    common::AnfAlgo::CopyNodeAttr(kAttrFracZGroup, orig_node, trans_data);
  }
  // refresh the transdata's format to ori format & dst format
  RefreshKernelBuildInfo(input_format, dst_format, trans_data, padding_axis);
  if (!is_insert_input) {
    ReFreshInferShape(trans_node, node);
  }
  return trans_node;
}

void RefreshKernelBuildInfo(const std::string &input_format, const std::string &output_format,
                            const AnfNodePtr &trans_node, const std::string &reshape_type, const TypeId &type_id) {
  MS_EXCEPTION_IF_NULL(trans_node);
  auto ori_build_info = AnfAlgo::GetSelectKernelBuildInfo(trans_node);
  MS_EXCEPTION_IF_NULL(ori_build_info);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>(ori_build_info);
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetInputsFormat({input_format});
  builder->SetInputsReshapeType({reshape_type});
  builder->SetOutputsReshapeType({reshape_type});
  builder->SetOutputsFormat({output_format});
  if (type_id != kTypeUnknown) {
    builder->SetOutputsDeviceType({type_id});
    builder->SetInputsDeviceType({type_id});
  }
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), trans_node.get());
  SetTransNodeAttr(trans_node->cast<CNodePtr>());
}

CNodePtr NewTransOpNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const AnfNodePtr &orig_node,
                        const KernelSelectPtr &kernel_select, const bool need_padding, const std::string &op_name,
                        const std::vector<int64_t> &perm) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(kernel_select);
  CNodePtr trans_node = NewCNode({NewValueNode(std::make_shared<Primitive>(op_name)), input}, func_graph, {orig_node});
  MS_EXCEPTION_IF_NULL(trans_node);
  auto infer_type = common::AnfAlgo::GetOutputInferDataType(input, 0);

  auto out_shape_base = common::AnfAlgo::GetOutputDetailShape(input, 0);
  MS_EXCEPTION_IF_NULL(out_shape_base);
  ShapeVector out_shape;
  ShapeVector out_shape_min;
  ShapeVector out_shape_max;
  bool is_dynamic_shape = false;
  if (out_shape_base->isa<abstract::Shape>()) {
    auto out_shape_ptr = out_shape_base->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(out_shape_ptr);
    out_shape = out_shape_ptr->shape();
    if (out_shape_ptr->IsDynamic()) {
      out_shape_min = out_shape_ptr->min_shape();
      out_shape_max = out_shape_ptr->max_shape();
      is_dynamic_shape = true;
    }
  }

  if (need_padding) {
    // if need padding we should set the transdata node's shape to the padding shape
    auto padding_axis = AnfAlgo::GetOutputReshapeType(input, 0);

    abstract::ShapePtr pad_shape_ptr;
    ShapeVector pad_shape = trans::PaddingShape(out_shape, AnfAlgo::GetOutputFormat(input, 0), padding_axis, input);
    if (is_dynamic_shape) {
      ShapeVector pad_shape_min =
        trans::PaddingShape(out_shape_min, AnfAlgo::GetOutputFormat(input, 0), padding_axis, input);
      ShapeVector pad_shape_max =
        trans::PaddingShape(out_shape_max, AnfAlgo::GetOutputFormat(input, 0), padding_axis, input);
      pad_shape_ptr = std::make_shared<abstract::Shape>(pad_shape, pad_shape_min, pad_shape_max);
    } else {
      pad_shape_ptr = std::make_shared<abstract::Shape>(pad_shape);
    }
    common::AnfAlgo::SetOutputTypeAndDetailShape({infer_type}, {pad_shape_ptr}, trans_node.get());
  } else {
    common::AnfAlgo::SetOutputTypeAndDetailShape({infer_type}, {out_shape_base}, trans_node.get());
  }
  // special handle for ut
  if (trans_node->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    trans_node->set_kernel_info(kernel_info);
  }
  if (op_name == prim::kPrimTranspose->name()) {
    common::AnfAlgo::SetNodeAttr(kAttrPerm, MakeValue(perm), trans_node);
  }
  if (is_dynamic_shape) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), trans_node);
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), trans_node);
  }
  common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), trans_node);
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, MakeValue<std::vector<std::string>>({}), trans_node);
  trans_node->set_scope(input->scope());
  kernel_select->SelectKernel(trans_node);
  return trans_node;
}

CNodePtr AddCastOpNodeToGraph(const FuncGraphPtr &func_graph, const AnfNodePtr &input, const AnfNodePtr &orig_node,
                              const std::string &format, const TypeId &input_type, const TypeId &output_type,
                              const abstract::BaseShapePtr &origin_shape, const TypeId &origin_type,
                              const std::string &reshape_type) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(origin_shape);
  std::string input_format = format;
  std::string output_format = format;
  CNodePtr cast =
    NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimCast->name())), input}, func_graph, {orig_node});
  MS_EXCEPTION_IF_NULL(cast);
  // set kernel build info
  kernel::KernelBuildInfo::KernelBuildInfoBuilder builder;
  builder.SetInputsFormat({input_format});
  builder.SetOutputsFormat({output_format});
  builder.SetInputsReshapeType({reshape_type});
  builder.SetOutputsReshapeType({reshape_type});
  builder.SetInputsDeviceType({input_type});
  builder.SetOutputsDeviceType({output_type});
  builder.SetFusionType(kernel::FusionType::OPAQUE);
  builder.SetProcessor(kernel::Processor::AICORE);
  if (kernel::OpLib::FindOp(prim::kPrimCast->name(), kernel::kTBE) != nullptr) {
    builder.SetKernelType(KernelType::TBE_KERNEL);
  } else {
    builder.SetKernelType(KernelType::AKG_KERNEL);
  }
  // if kernel info is null , it remarks this function is running ut
  if (cast->kernel_info() == nullptr) {
    auto kernel_info = std::make_shared<device::KernelInfo>();
    cast->set_kernel_info(kernel_info);
  }
  if (origin_shape->IsDynamic()) {
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(true), cast);
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(true), cast);
  }
  common::AnfAlgo::SetNodeAttr("dst_type", TypeIdToType(origin_type), cast);
  AnfAlgo::SetSelectKernelBuildInfo(builder.Build(), cast.get());
  common::AnfAlgo::SetOutputTypeAndDetailShape({origin_type}, {origin_shape}, cast.get());
  common::AnfAlgo::SetNodeAttr(kIsBackendCast, MakeValue(true), cast);
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, MakeValue<std::vector<std::string>>({}), cast);
  return cast;
}

AnfNodePtr InsertTransOpForOutput(const FuncGraphPtr &func_graph, const AnfNodePtr &orig_node, const AnfNodePtr &node,
                                  const KernelSelectPtr &kernel_select) {
  size_t outputs_num = common::AnfAlgo::GetOutputTensorNum(node);
  if (outputs_num == 0) {
    return node;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  // Single output
  if (outputs_num == 1 && (!common::AnfAlgo::IsTupleOutput(node))) {
    auto new_node = InsertTransOpForSingleOutput(func_graph, node, kernel_select);
    if (kernel_graph != nullptr && kernel_graph->IsInternalOutput(node, 0)) {
      kernel_graph->ReplaceInternalOutput(node, new_node);
    }
    return new_node;
  }
  // Multiple output
  return InsertTransOpForMultipleOutput(func_graph, orig_node, node, kernel_select);
}

AnfNodePtr InsertTransOpForInput(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                 const KernelSelectPtr &kernel_select) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(cnode)};
  size_t in_num = common::AnfAlgo::GetInputNum(cnode);  // include monads.
  for (size_t input_index = 0; input_index < in_num; ++input_index) {
    // Monad inputs keep unchanged from GetTransInputNodePtr().
    AnfNodePtr input_node = GetTransInputNodePtr(func_graph, cnode, input_index, kernel_select);
    MS_EXCEPTION_IF_NULL(input_node);
    new_inputs.push_back(input_node);
  }
  CNodePtr new_cnode = nullptr;
  // cnode changed so make a new cnode to differ from original one.
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  if (kernel_graph == nullptr) {
    new_cnode = std::make_shared<CNode>(*cnode);
  } else {
    new_cnode = kernel_graph->NewCNode(cnode);
  }
  MS_EXCEPTION_IF_NULL(new_cnode);
  new_cnode->set_inputs(new_inputs);
  return new_cnode;
}

CNodePtr InsertCastForInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> new_inputs = {common::AnfAlgo::GetCNodePrimitiveNode(cnode)};
  size_t in_num = common::AnfAlgo::GetInputNum(cnode);  // include monads.
  for (size_t input_index = 0; input_index < in_num; ++input_index) {
    auto cur_input = common::AnfAlgo::GetInputNode(cnode, input_index);
    MS_EXCEPTION_IF_NULL(cur_input);
    if (HasAbstractMonad(cur_input)) {
      // No cast for monad inputs.
      new_inputs.push_back(cur_input);
      continue;
    }
    auto prev_node = common::AnfAlgo::GetPrevNodeOutput(cnode, input_index);
    const auto infer_type = common::AnfAlgo::GetOutputInferDataType(prev_node.first, prev_node.second);
    TypeId origin_type(kTypeUnknown);

    auto kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(cur_input, 0);
    auto real_input_node = kernel_with_index.first;
    MS_EXCEPTION_IF_NULL(real_input_node);
    if (kernel::IsWeightBoundary(real_input_node)) {
      // weight
      origin_type = common::AnfAlgo::GetPrevNodeOutputPrecision(cnode, input_index);
      if (origin_type == kTypeUnknown) {
        origin_type = AnfAlgo::GetOutputDeviceDataType(prev_node.first, prev_node.second);
      }
    } else {
      // feature map
      origin_type = common::AnfAlgo::GetOutputInferDataType(prev_node.first, prev_node.second);
    }
    const std::string dev_fmt = AnfAlgo::GetInputFormat(cnode, input_index);
    const abstract::BaseShapePtr origin_shape =
      common::AnfAlgo::GetOutputDetailShape(prev_node.first, prev_node.second);
    // In graph kernel, we check parameter,
    // the eliminate pass will not eliminate this case, so we just do not insert the no used cast.
    if (TypeId device_type = AnfAlgo::GetInputDeviceDataType(cnode, input_index); origin_type != device_type) {
      auto cast =
        AddCastOpNodeToGraph(func_graph, cur_input, cnode, dev_fmt, origin_type, device_type, origin_shape, infer_type);
      MS_EXCEPTION_IF_NULL(cast);
      cast->set_scope(cnode->scope());
      common::AnfAlgo::SetNodeAttr(kAttrVisited, MakeValue(true), cast);
      new_inputs.push_back(cast);
    } else {
      new_inputs.push_back(cur_input);
    }
  }
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  CNodePtr new_node = nullptr;
  if (kernel_graph == nullptr) {
    new_node = std::make_shared<CNode>(*cnode);
  } else {
    new_node = kernel_graph->NewCNode(cnode);
  }
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_inputs(new_inputs);
  return new_node;
}

AnfNodePtr CreateTensorMoveOp(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto prim = std::make_shared<Primitive>(kTensorMoveOpName);
  std::vector<AnfNodePtr> new_node_inputs = {NewValueNode(prim), node};
  auto new_node = graph->NewCNode(new_node_inputs);
  MS_EXCEPTION_IF_NULL(new_node);
  new_node->set_abstract(node->abstract());
  new_node->set_scope(node->scope());
  common::AnfAlgo::SetNodeAttr(kAttrDatadumpOriginalNames, MakeValue<std::vector<std::string>>({}), new_node);
  return new_node;
}
}  // namespace opt
}  // namespace mindspore
