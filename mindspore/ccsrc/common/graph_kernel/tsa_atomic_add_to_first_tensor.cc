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

#include "common/graph_kernel/tsa_atomic_add_to_first_tensor.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "base/core_ops.h"
#include "ir/tensor.h"
#include "include/common/utils/utils.h"
#include "utils/log_adapter.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_graph.h"
#include "kernel/kernel.h"
#include "kernel/common_utils.h"
#include "common/graph_kernel/graph_kernel_helper.h"
#include "common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
constexpr auto kTsaInputIndex = 2;
class TsaChecker : public AtomicAddChecker {
 public:
  explicit TsaChecker(const PrimitivePtr &target) { target_type_ = target; }
  virtual ~TsaChecker() = default;

 protected:
  bool CanActivateAtomicAdd(const AnfNodePtr &anf_node) override {
    if (!FindCandidate(anf_node)) {
      return false;
    }

    for (auto atomic_add_info : atomic_add_infos_) {
      auto tsa_cnode = atomic_add_info.atomic_add_node;
      if (!utils::isa<ParameterPtr>(tsa_cnode->input(1))) {
        return false;
      }
    }

    return true;
  }
};

std::pair<AnfNodePtr, size_t> TsaAtomicAddToFirstTensor::FindTsaFirstRealInputInGraph(const KernelGraphPtr &,
                                                                                      const CNodePtr &tsa_node,
                                                                                      const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(cnode);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  auto first_input = tsa_node->input(1)->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(first_input);
  auto parameters = sub_graph->parameters();
  bool hit = false;
  size_t tsa_first_input_index = 0;
  for (size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i] == first_input) {
      tsa_first_input_index = i;
      hit = true;
      break;
    }
  }
  if (!hit) {
    MS_LOG(EXCEPTION) << "Cannot find tensor scatter add first input in sub-graph parameters!";
  }

  return {cnode->input(tsa_first_input_index + 1), tsa_first_input_index};  // CNode input have a primitive, so add 1.
}

std::pair<AnfNodePtr, size_t> TsaAtomicAddToFirstTensor::GetOrCreateNewTsaFirstNode(
  const KernelGraphPtr &main_graph, const AtomicAddInfo &atomic_add_info, const AnfNodePtr &node) {
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = Manage(main_graph, true);
    main_graph->set_manager(mng);
  }

  // Find first input of tsa
  auto tsa_first_input = FindTsaFirstRealInputInGraph(main_graph, atomic_add_info.atomic_add_node, node);
  auto users = mng->node_users()[tsa_first_input.first];
  if (users.size() == 1 &&
      !(utils::isa<ValueNodePtr>(tsa_first_input.first) || utils::isa<ParameterPtr>(tsa_first_input.first))) {
    // If current composite node is only user, and first input is not Parameter or Tensor Value, then use itself.
    return tsa_first_input;
  }

  // Create a copy of first input to atomic add to.
  // Create composite op's sub-graph.
  auto new_sub_graph = std::make_shared<FuncGraph>();
  auto parameter = new_sub_graph->add_parameter();
  auto kernel_with_index = common::AnfAlgo::VisitKernel(tsa_first_input.first, 0);
  parameter->set_abstract(GetOutputAbstract(kernel_with_index.first, kernel_with_index.second));
  parameter->set_kernel_info(std::make_shared<device::KernelInfo>());
  std::string parameter_format;
  TypeId parameter_type;
  if (utils::isa<ValueNodePtr>(kernel_with_index.first)) {
    auto tensor = GetValueNode<tensor::TensorPtr>(kernel_with_index.first);
    MS_EXCEPTION_IF_NULL(tensor);
    parameter_format = kOpFormat_DEFAULT;
    parameter_type = tensor->data_type();
  } else {
    parameter_format = AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    parameter_type = AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
  }

  kernel::KernelBuildInfo::KernelBuildInfoBuilder para_info_builder;
  para_info_builder.SetOutputsFormat({parameter_format});
  para_info_builder.SetOutputsDeviceType({parameter_type});
  para_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  para_info_builder.SetProcessor(kernel::GetProcessorFromContext());
  AnfAlgo::SetSelectKernelBuildInfo(para_info_builder.Build(), parameter.get());

  // Create inner op.
  auto identity_node =
    CreateCNode({NewValueNode(std::make_shared<Primitive>("Reshape")), parameter}, new_sub_graph,
                {.format = GetFormat(parameter), .shape = GetShape(parameter), .type = GetType(parameter)});
  SetNodeAttrSafely("shape", MakeValue(GetDeviceShape(parameter)), identity_node);

  // Makeup sub-graph.
  new_sub_graph->set_output(identity_node);
  auto new_copy_composite_node = main_graph->NewCNode({NewValueNode(new_sub_graph), tsa_first_input.first});
  new_copy_composite_node->set_abstract(identity_node->abstract());
  SetNewKernelInfo(new_copy_composite_node, new_sub_graph, {tsa_first_input.first}, {identity_node});
  auto graph_attr = GkUtils::ExtractGraphKernelName(TopoSort(new_sub_graph->get_return()), "", "tsa_identity");
  new_sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(graph_attr));
  new_sub_graph->set_attr("composite_type", MakeValue("tsa_identity"));

  return {new_copy_composite_node, tsa_first_input.second};
}

void TsaAtomicAddToFirstTensor::ChangeKernelBuildInfo(
  const AnfNodePtr &composite_node, const std::vector<std::tuple<AtomicAddInfo, AnfNodePtr, size_t>> &outer_infos) {
  // Change kernel build info with modify input
  auto kernel_info = static_cast<device::KernelInfo *>(composite_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  const auto &origin_kernel_build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  auto origin_inputs_format = origin_kernel_build_info->GetAllInputFormats();
  auto origin_outputs_format = origin_kernel_build_info->GetAllOutputFormats();
  auto origin_inputs_type = origin_kernel_build_info->GetAllInputDeviceTypes();
  auto origin_outputs_type = origin_kernel_build_info->GetAllOutputDeviceTypes();
  auto origin_processor = origin_kernel_build_info->processor();

  std::vector<std::string> &modified_inputs_format = origin_inputs_format;
  std::vector<TypeId> &modified_inputs_type = origin_inputs_type;
  std::vector<std::string> new_outputs_format;
  std::vector<TypeId> new_outputs_type;

  std::set<size_t> reduce_real_indices;
  for (auto &info : outer_infos) {
    (void)reduce_real_indices.insert(std::get<0>(info).reduce_real_output_index);
  }

  for (size_t i = 0; i < origin_outputs_format.size(); ++i) {
    if (std::get<0>(outer_infos[0]).real_output_num > 1 && reduce_real_indices.count(i) > 0) {
      continue;
    }
    new_outputs_format.push_back(origin_outputs_format[i]);
    new_outputs_type.push_back(origin_outputs_type[i]);
  }

  for (const auto &outer_info : outer_infos) {
    auto &modified_input = std::get<1>(outer_info);
    auto tsa_first_input_index = std::get<kTsaInputIndex>(outer_info);
    auto kernel_with_index = common::AnfAlgo::VisitKernel(modified_input, 0);
    modified_inputs_format[tsa_first_input_index] =
      AnfAlgo::GetOutputFormat(kernel_with_index.first, kernel_with_index.second);
    modified_inputs_type[tsa_first_input_index] =
      AnfAlgo::GetOutputDeviceDataType(kernel_with_index.first, kernel_with_index.second);
  }

  kernel::KernelBuildInfo::KernelBuildInfoBuilder new_info_builder;
  new_info_builder.SetInputsFormat(modified_inputs_format);
  new_info_builder.SetInputsDeviceType(modified_inputs_type);
  new_info_builder.SetOutputsFormat(new_outputs_format);
  new_info_builder.SetOutputsDeviceType(new_outputs_type);
  new_info_builder.SetProcessor(origin_processor);
  new_info_builder.SetKernelType(KernelType::AKG_KERNEL);
  new_info_builder.SetFusionType(kernel::FusionType::OPAQUE);
  auto new_selected_info = new_info_builder.Build();
  AnfAlgo::SetSelectKernelBuildInfo(new_selected_info, composite_node.get());
}

void TsaAtomicAddToFirstTensor::ProcessOriginalCNode(
  const AnfNodePtr &composite_node, const std::vector<std::tuple<AtomicAddInfo, AnfNodePtr, size_t>> &outer_nodes) {
  auto sub_graph = common::AnfAlgo::GetCNodeFuncGraphPtr(composite_node);
  auto mng_sub = sub_graph->manager();
  if (mng_sub == nullptr) {
    mng_sub = Manage(sub_graph, false);
    sub_graph->set_manager(mng_sub);
  }

  // Modify input
  std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> parameters_infos;
  std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> info_and_tsa_outers;
  for (const auto &[atomic_add_info, outer_node, tsa_first_input_index] : outer_nodes) {
    composite_node->cast<CNodePtr>()->set_input(tsa_first_input_index + 1, outer_node);
    auto parameter = sub_graph->parameters()[tsa_first_input_index];
    (void)parameters_infos.emplace_back(atomic_add_info, parameter);
    (void)info_and_tsa_outers.emplace_back(atomic_add_info, outer_node);
  }

  CreateInplaceAssignNodeAndCorrectReturn(sub_graph, parameters_infos);

  CorrectAbstract(composite_node, info_and_tsa_outers);
  ChangeKernelBuildInfo(composite_node, outer_nodes);

  auto old_graph_name = GetValue<std::string>(sub_graph->get_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL));
  auto new_graph_name =
    GkUtils::ExtractGraphKernelName(TopoSort(sub_graph->get_return()), "", "tensor_scatter_add_modified");
  sub_graph->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(new_graph_name));
  MS_LOG(INFO) << "Convert " << old_graph_name << " to tensor scatter add graph " << new_graph_name;
}

void TsaAtomicAddToFirstTensor::ProcessTsa(const KernelGraphPtr &main_graph, const AnfNodePtr &anf_node,
                                           const std::vector<AtomicAddInfo> &atomic_add_infos,
                                           const FuncGraphManagerPtr &mng) {
  auto origin_composite_node = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_composite_node);

  // Create identity node.  // Create broadcst node.
  std::vector<std::tuple<AtomicAddInfo, AnfNodePtr, size_t>> info_and_outer_nodes_with_index;
  std::vector<std::pair<AtomicAddInfo, AnfNodePtr>> info_and_outer_nodes;
  for (auto atomic_add_info : atomic_add_infos) {
    auto outer = GetOrCreateNewTsaFirstNode(main_graph, atomic_add_info, anf_node);
    (void)info_and_outer_nodes_with_index.emplace_back(atomic_add_info, outer.first, outer.second);
    (void)info_and_outer_nodes.emplace_back(atomic_add_info, outer.first);
  }

  // Insert extra input(broadcast node output) to composite node, and make origin TensorScatterAdd inplaceassign to it.
  // Note: if it's single output, this will increase total memory because of a fake out.
  ProcessOriginalCNode(origin_composite_node, info_and_outer_nodes_with_index);

  // Insert update_state_node to keep execution order.
  auto update_state_node = InsertUpdateState(main_graph, origin_composite_node);

  // Replace origin ReduceSum's user with atomic clean output
  ProcessOriginCNodeUser(main_graph, origin_composite_node, info_and_outer_nodes, update_state_node, mng);
  std::stringstream ss;
  ss << "Target node: " << origin_composite_node->fullname_with_scope() << ", outer nodes: ";
  for (auto iter : info_and_outer_nodes) {
    ss << iter.second->fullname_with_scope() << ", ";
  }
}

bool TsaAtomicAddToFirstTensor::Run(const FuncGraphPtr &func_graph) {
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(func_graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto mng = kernel_graph->manager();
  if (mng == nullptr) {
    mng = Manage(kernel_graph, true);
    kernel_graph->set_manager(mng);
  }

  bool changed = false;
  std::shared_ptr<AtomicAddChecker> atomic_add_checker =
    std::make_shared<TsaChecker>(std::make_shared<Primitive>("TensorScatterAdd"));
  if (atomic_add_checker == nullptr) {
    return changed;
  }

  auto topo_nodes = TopoSort(kernel_graph->get_return());
  for (const auto &node : topo_nodes) {
    if (!atomic_add_checker->Check(node)) {
      continue;
    }
    auto atomic_add_infos = atomic_add_checker->GetAtomicAddInfo();
    ProcessTsa(kernel_graph, node, atomic_add_infos, mng);
    changed = true;
  }

  if (changed) {
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
  }

  return changed;
}
}  // namespace mindspore::graphkernel
