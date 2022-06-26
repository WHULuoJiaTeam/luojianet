/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/kernel_build_ascend.h"
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <set>
#include <map>
#include "plugin/device/ascend/hal/device/kernel_select_ascend.h"
#include "plugin/device/ascend/kernel/tbe/tbe_kernel_compile.h"
#include "plugin/device/ascend/kernel/akg/akg_ascend_kernel_build.h"
#include "plugin/device/ascend/kernel/aicpu/aicpu_kernel_build.h"
#include "plugin/device/ascend/kernel/host/host_kernel_build.h"
#include "plugin/device/ascend/kernel/hccl/hccl_kernel_build.h"
#include "plugin/device/ascend/kernel/rts/rt_kernel_build.h"
#include "plugin/device/ascend/kernel/tbe/tbe_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
using mindspore::kernel::tbe::TbeUtils;
using std::make_shared;
constexpr size_t kMaxAttrMemListSize = 192;

static kernel::KernelModPtr SerialCompileImpl(const AnfNodePtr &anf_node) {
  kernel::KernelModPtr kernel_mod_ptr = nullptr;
  KernelType kernel_type = AnfAlgo::GetKernelType(anf_node);
  switch (kernel_type) {
    case KernelType::AICPU_KERNEL: {
      kernel_mod_ptr = kernel::AicpuOpBuild(anf_node);
      break;
    }
    case KernelType::HOST_KERNEL: {
      kernel_mod_ptr = kernel::HostOpBuild(anf_node);
      break;
    }
    case KernelType::RT_KERNEL: {
      kernel_mod_ptr = kernel::RtOpBuild(anf_node);
      break;
    }
    case KernelType::HCCL_KERNEL: {
      kernel_mod_ptr = kernel::HcclOpBuild(anf_node);
      break;
    }
    default: {
      MS_EXCEPTION_IF_NULL(anf_node);
      MS_LOG(EXCEPTION) << "node [" << anf_node->DebugString() << "] Unsupported kernel_type:" << kernel_type;
    }
  }
  return kernel_mod_ptr;
}

static bool KernelBuildParallelCompile(const std::vector<CNodePtr> &kernels) {
  std::vector<CNodePtr> tbe_nodes;
  std::vector<AnfNodePtr> akg_nodes;
  std::vector<AnfNodePtr> other_nodes;
  for (const auto &anf_node : kernels) {
    MS_EXCEPTION_IF_NULL(anf_node);
    if (!AnfUtils::IsRealKernel(anf_node)) {
      continue;
    }
    if (AnfAlgo::GetKernelMod(anf_node) != nullptr) {
      continue;
    }
    KernelType kernel_type = AnfAlgo::GetKernelType(anf_node);
    switch (kernel_type) {
      case KernelType::TBE_KERNEL: {
        tbe_nodes.push_back(anf_node);
        break;
      }
      case KernelType::AKG_KERNEL: {
        akg_nodes.push_back(anf_node);
        break;
      }
      default: {
        other_nodes.push_back(anf_node);
        break;
      }
    }
  }
  auto bin_map = kernel::tbe::KernelMeta::GetInstance();
  MS_EXCEPTION_IF_NULL(bin_map);
  if (!tbe_nodes.empty()) {
    auto &build_manager = kernel::ascend::TbeKernelCompileManager::GetInstance();
    build_manager.TbeSingleOpCompile(tbe_nodes);
    auto config_path = TbeUtils::GetOpDebugPath();
    std::string dir = config_path + "kernel_meta/";
    (void)bin_map->ReadIndex(dir);
  }
  bool akg_ret = true;
  if (!akg_nodes.empty()) {
    kernel::AkgAscendKernelBuilder akg_ascend_kernel_builder;
    akg_ret = akg_ascend_kernel_builder.AkgKernelParallelBuild(akg_nodes);
  }
  for (const auto &anf_node : other_nodes) {
    kernel::KernelModPtr kernel_mod_ptr = SerialCompileImpl(anf_node);
    MS_EXCEPTION_IF_NULL(kernel_mod_ptr);
    AnfAlgo::SetKernelMod(kernel_mod_ptr, anf_node.get());
  }
  return akg_ret;
}

bool KernelBuild(const std::vector<CNodePtr> &kernels) { return device::ascend::KernelBuildParallelCompile(kernels); }

namespace {
bool IsAtomicNode(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto parameters_indexes = kernel_mod->GenParameters();
  if (parameters_indexes.empty()) {
    return false;
  }
  if (common::AnfAlgo::IsDynamicShape(kernel_node)) {
    if (parameters_indexes.at(0) == 1) {
      (void)parameters_indexes.erase(parameters_indexes.begin());
    } else {
      parameters_indexes.pop_back();
    }
  }
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  size_t workspace_num = kernel_mod->GetWorkspaceSizeList().size();
  size_t param_num = parameters_indexes.size();
  size_t total_num = input_num + output_num + workspace_num;
  size_t pad_index = param_num;

  for (; pad_index < total_num; ++pad_index) {
    (void)parameters_indexes.emplace_back(0);
  }

  for (size_t j = 0; j < input_num; ++j) {
    if (parameters_indexes.at(j) == 1) {
      MS_LOG(EXCEPTION) << "Atomic clean doesn't support clean input address, input index: " << j;
    }
  }

  if (parameters_indexes.size() < total_num) {
    MS_LOG(EXCEPTION) << "Parameters indexes size: " << parameters_indexes.size()
                      << " less than total num: " << total_num;
  }
  // process output
  std::vector<size_t> output_indexes = {};
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, kernel_node)) {
    output_indexes = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(kernel_node, kAttrAtomicOutputIndexs);
  }

  for (size_t i = 0; i < output_num; ++i) {
    auto param_output = parameters_indexes.at(input_num + i);
    if (param_output == 1) {
      (void)output_indexes.emplace_back(i);
      MS_LOG(DEBUG) << "Atomic clear output index: " << i;
    }
  }

  if (!output_indexes.empty()) {
    std::set<size_t> s(output_indexes.begin(), output_indexes.end());
    output_indexes.assign(s.begin(), s.end());
    common::AnfAlgo::SetNodeAttr(kAttrAtomicOutputIndexs, MakeValue(output_indexes), kernel_node);
  }
  // process workspace
  std::vector<size_t> workspace_indexes = {};
  for (size_t k = 0; k < workspace_num; ++k) {
    auto param_workspace = parameters_indexes.at(input_num + output_num + k);
    if (param_workspace == 1) {
      (void)workspace_indexes.emplace_back(k);
      MS_LOG(DEBUG) << "Atomic clear workspace index: " << k;
    }
  }
  if (!workspace_indexes.empty()) {
    common::AnfAlgo::SetNodeAttr(kAttrAtomicWorkspaceIndexs, MakeValue(workspace_indexes), kernel_node);
  }
  return !(workspace_indexes.empty() && output_indexes.empty());
}

bool IfAtomicOpNeedFusion(const size_t clean_total_num, const CNodePtr &first_node, const CNodePtr &current_node) {
  if (first_node == nullptr || current_node == nullptr) {
    return false;
  }
  auto first_graph_id = AnfAlgo::GetGraphId(first_node.get());
  auto current_graph_id = AnfAlgo::GetGraphId(current_node.get());
  if (clean_total_num >= kMaxAttrMemListSize || first_graph_id != current_graph_id) {
    return true;
  }
  return false;
}

std::vector<size_t> GetClearSize(const CNodePtr &pre_node) {
  MS_EXCEPTION_IF_NULL(pre_node);
  auto kernel_mod = AnfAlgo::GetKernelMod(pre_node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  std::vector<size_t> clean_size_list;
  constexpr size_t kAlignBytes = 32 - 1;
  // clean output
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
    auto output_indexes = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
    auto output_men_size = kernel_mod->GetOutputSizeList();
    for (auto index : output_indexes) {
      auto clean_item = (output_men_size.at(index) + kMemAlignSize + kAlignBytes) / kMemAlignSize * kMemAlignSize;
      (void)clean_size_list.emplace_back(clean_item);
    }
  }
  // clean workspace
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
    auto workspace_indexes = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
    auto workspace_men_sizes = kernel_mod->GetWorkspaceSizeList();
    for (const auto &index : workspace_indexes) {
      auto clean_item = (workspace_men_sizes.at(index) + kMemAlignSize + kAlignBytes) / kMemAlignSize * kMemAlignSize;
      (void)clean_size_list.emplace_back(clean_item);
    }
  }
  MS_LOG(INFO) << "Clear output size:" << clean_size_list.size() << ",pre_node:" << pre_node->fullname_with_scope();
  return clean_size_list;
}

CNodePtr NewAtomicOp(const CNodePtr &pre_node, const std::vector<AnfNodePtr> &fusion_clear_inputs) {
  MS_EXCEPTION_IF_NULL(pre_node);
  auto clear_zero_prim = std::make_shared<Primitive>(kAtomicAddrCleanOpName);
  MS_EXCEPTION_IF_NULL(clear_zero_prim);
  auto new_value_node = NewValueNode(clear_zero_prim);
  MS_EXCEPTION_IF_NULL(new_value_node);
  std::vector<AnfNodePtr> inputs = {new_value_node};
  (void)inputs.insert(inputs.end(), fusion_clear_inputs.begin(), fusion_clear_inputs.end());
  auto func_graph = pre_node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  CNodePtr clear_zero = kernel_graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(clear_zero);
  AbstractBasePtr abstract = std::make_shared<abstract::AbstractNone>();
  MS_EXCEPTION_IF_NULL(abstract);
  clear_zero->set_abstract(abstract);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetKernelType(KernelType::TBE_KERNEL);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), clear_zero.get());
  return clear_zero;
}

void InsertFusionAtomicOp(const CNodePtr &first_clear_node, const std::vector<AnfNodePtr> &fusion_clear_inputs,
                          const std::vector<size_t> &clean_size_list, CleanOpsMap *clean_ops) {
  MS_EXCEPTION_IF_NULL(first_clear_node);
  MS_EXCEPTION_IF_NULL(clean_ops);
  auto clear_zero = NewAtomicOp(first_clear_node, fusion_clear_inputs);
  common::AnfAlgo::SetNodeAttr(kAttrAtomicAddMemSize, MakeValue(clean_size_list), clear_zero);
  AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(first_clear_node.get()), clear_zero.get());
  MS_LOG(DEBUG) << "The AtomicClean currently does not support dynamic shape.";
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(false), clear_zero);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(false), clear_zero);
  (void)(*clean_ops)[first_clear_node].emplace_back(clear_zero);
}

void InsertAtomicOpForNormalOp(const mindspore::CNodePtr &pre_node, CleanOpsMap *clean_ops) {
  MS_EXCEPTION_IF_NULL(pre_node);
  MS_EXCEPTION_IF_NULL(clean_ops);
  auto clear_zero = NewAtomicOp(pre_node, {pre_node});
  auto clean_size = GetClearSize(pre_node);
  common::AnfAlgo::SetNodeAttr(kAttrAtomicAddMemSize, MakeValue(clean_size), clear_zero);
  AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(pre_node.get()), clear_zero.get());
  MS_LOG(DEBUG) << "The AtomicClean currently does not support dynamic shape.";
  common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(false), clear_zero);
  common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(false), clear_zero);
  (void)(*clean_ops)[pre_node].emplace_back(clear_zero);
}

void SpecialAkgOps(const std::string &op_name, const CNodePtr &node, CleanOpsMap *clean_ops) {
  MS_EXCEPTION_IF_NULL(clean_ops);
  if (op_name == prim::kPrimMaxPoolGrad->name() && AnfAlgo::GetKernelType(node) == KernelType::AKG_KERNEL) {
    auto clear_zero_prim = std::make_shared<Primitive>(kClearZeroOpName);
    MS_EXCEPTION_IF_NULL(clear_zero_prim);
    auto new_value_node = NewValueNode(clear_zero_prim);
    MS_EXCEPTION_IF_NULL(new_value_node);
    std::vector<AnfNodePtr> inputs = {new_value_node};
    inputs.push_back(node);
    auto func_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);
    CNodePtr clear_zero = kernel_graph->NewCNode(inputs);
    MS_EXCEPTION_IF_NULL(clear_zero);
    auto kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    clear_zero->set_kernel_info(kernel_info);
    AbstractBasePtr abstract = std::make_shared<abstract::AbstractNone>();
    MS_EXCEPTION_IF_NULL(abstract);
    common::AnfAlgo::SetNodeAttr("input_names", MakeValue(std::vector<std::string>({"x"})), clear_zero);
    (void)SelectKernelInfo(clear_zero);
    // set the distinction label of clear same with anf
    AnfAlgo::SetStreamDistinctionLabel(AnfAlgo::GetStreamDistinctionLabel(node.get()), clear_zero.get());
    MS_LOG(DEBUG) << "The AtomicClean currently does not support dynamic shape.";
    common::AnfAlgo::SetNodeAttr(kAttrInputIsDynamicShape, MakeValue(false), clear_zero);
    common::AnfAlgo::SetNodeAttr(kAttrOutputIsDynamicShape, MakeValue(false), clear_zero);
    (void)(*clean_ops)[node].emplace_back(clear_zero);
  }
}

void ProcessAtomicFusion(const std::vector<CNodePtr> &kernels, CleanOpsMap *clean_ops) {
  MS_EXCEPTION_IF_NULL(clean_ops);
  std::vector<size_t> clean_size_list;
  std::vector<AnfNodePtr> fusion_clear_inputs;
  CNodePtr first_node = nullptr;
  for (const auto &anf_node : kernels) {
    MS_EXCEPTION_IF_NULL(anf_node);
    std::string apply_function_name = common::AnfAlgo::GetCNodeName(anf_node);
    SpecialAkgOps(apply_function_name, anf_node, clean_ops);
    if (common::AnfAlgo::HasNodeAttr(kAttrNeedAtomic, anf_node) &&
        common::AnfAlgo::GetNodeAttr<bool>(anf_node, kAttrNeedAtomic)) {
      auto clean_sizes = GetClearSize(anf_node);
      if (!clean_sizes.empty()) {
        auto clean_total_num = clean_size_list.size() + clean_sizes.size();
        if (IfAtomicOpNeedFusion(clean_total_num, first_node, anf_node)) {
          // create clean node
          InsertFusionAtomicOp(first_node, fusion_clear_inputs, clean_size_list, clean_ops);
          clean_size_list.clear();
          fusion_clear_inputs.clear();
          first_node = nullptr;
        }
        if (fusion_clear_inputs.empty()) {
          first_node = anf_node;
        }
        (void)clean_size_list.insert(clean_size_list.end(), clean_sizes.begin(), clean_sizes.end());
        (void)fusion_clear_inputs.emplace_back(anf_node);
        MS_LOG(DEBUG) << "The fusion_clear_inputs size: " << fusion_clear_inputs.size()
                      << ", clean_size_list: " << clean_size_list.size();
      }
    }
  }
  if (!fusion_clear_inputs.empty() && !clean_size_list.empty()) {
    // create clean node
    InsertFusionAtomicOp(first_node, fusion_clear_inputs, clean_size_list, clean_ops);
  }
}

void InsertAtomicOps(const std::vector<CNodePtr> &kernels, CleanOpsMap *clean_ops) {
  // fusion
  MS_EXCEPTION_IF_NULL(clean_ops);
  static const auto enable_fusion_clear = (common::GetEnv("ENV_FUSION_CLEAR") == "1");
  if (enable_fusion_clear) {
    ProcessAtomicFusion(kernels, clean_ops);
    return;
  }
  // single
  for (const auto &node : kernels) {
    std::string apply_function_name = common::AnfAlgo::GetCNodeName(node);
    SpecialAkgOps(apply_function_name, node, clean_ops);
    if (common::AnfAlgo::HasNodeAttr(kAttrNeedAtomic, node) &&
        common::AnfAlgo::GetNodeAttr<bool>(node, kAttrNeedAtomic)) {
      InsertAtomicOpForNormalOp(node, clean_ops);
    }
  }
}

std::map<AnfNodePtr, std::vector<size_t>> GetCommunicationOpInputInfo(const std::vector<CNodePtr> &kernels) {
  std::map<AnfNodePtr, std::vector<size_t>> comm_input_info_map;
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    if (common::AnfAlgo::IsCommunicationOp(kernel)) {
      for (size_t i = 0; i < input_num; i++) {
        auto input_node = kernel->input(i + 1);
        auto kernel_input = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0, true);
        MS_EXCEPTION_IF_NULL(kernel_input.first);
        if (!kernel_input.first->isa<CNode>()) {
          continue;
        }
        auto cnode = kernel_input.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        if (common::AnfAlgo::IsCommunicationOp(cnode) || AnfAlgo::IsIndependentNode(cnode) ||
            common::AnfAlgo::GetCNodeName(cnode) == kGetNextOpName) {
          // no need to add atomic for communication or independent or get_next op's output
          MS_LOG(INFO) << "No need insert atomic clean for op " << cnode->fullname_with_scope() << "'s output";
          continue;
        }
        MS_LOG(INFO) << "Add atomic clean for single communication op input, comm:" << kernel->fullname_with_scope()
                     << " input_node: " << kernel_input.first->fullname_with_scope()
                     << " index: " << kernel_input.second;
        auto iter = comm_input_info_map.find(kernel_input.first);
        if (iter != comm_input_info_map.end()) {
          iter->second.push_back(kernel_input.second);
        } else {
          std::vector<size_t> indexes = {kernel_input.second};
          comm_input_info_map[kernel_input.first] = indexes;
        }
      }
    }
  }

  // remove duplicate index
  for (auto &info : comm_input_info_map) {
    std::set<size_t> s(info.second.begin(), info.second.end());
    info.second.assign(s.begin(), s.end());
  }
  return comm_input_info_map;
}

void TagNeedInsertAtomicAttr(const std::vector<CNodePtr> &nodes) {
  if (nodes.empty()) {
    return;
  }
  std::map<AnfNodePtr, std::vector<size_t>> comm_input_info_map = GetCommunicationOpInputInfo(nodes);
  for (const auto &anf_node : nodes) {
    if (comm_input_info_map.find(anf_node) != comm_input_info_map.end()) {
      auto indexes = comm_input_info_map[anf_node];
      if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, anf_node)) {
        auto output_indexes = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(anf_node, kAttrAtomicOutputIndexs);
        (void)std::copy(indexes.begin(), indexes.end(), std::back_inserter(output_indexes));
        std::set<size_t> tmp(output_indexes.begin(), output_indexes.end());
        indexes.assign(tmp.begin(), tmp.end());
      }
      common::AnfAlgo::SetNodeAttr(kAttrAtomicOutputIndexs, MakeValue(indexes), anf_node);
      common::AnfAlgo::SetNodeAttr(kAttrNeedAtomic, MakeValue(true), anf_node);
    } else if (AnfAlgo::GetKernelType(anf_node) == KernelType::TBE_KERNEL && IsAtomicNode(anf_node)) {
      common::AnfAlgo::SetNodeAttr(kAttrNeedAtomic, MakeValue(true), anf_node);
    }
  }
}

std::vector<CNodePtr> GatherAllAtomicOps(const CleanOpsMap &node_maps) {
  std::vector<CNodePtr> all_atomics;
  auto iter = node_maps.begin();
  while (iter != node_maps.end()) {
    auto tmp = iter->second;
    (void)std::copy(tmp.begin(), tmp.end(), std::back_inserter(all_atomics));
    (void)iter++;
  }
  return all_atomics;
}
}  // namespace

void InsertAtomicCleanOps(const std::vector<CNodePtr> &nodes, CleanOpsMap *maps) {
  MS_EXCEPTION_IF_NULL(maps);
  // assign attr
  TagNeedInsertAtomicAttr(nodes);
  // insert atomic
  InsertAtomicOps(nodes, maps);
  std::vector<CNodePtr> all_atomics = GatherAllAtomicOps(*maps);
  // build atomic
  (void)KernelBuild(all_atomics);
}

void InsertAtomicCleanOps(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &exe_orders = kernel_graph->execution_order();
  // assign attr
  TagNeedInsertAtomicAttr(exe_orders);
  // insert atomic
  CleanOpsMap node_to_cleans;
  InsertAtomicOps(exe_orders, &node_to_cleans);
  // update exec order
  std::vector<CNodePtr> new_orders;
  for (const auto &node : exe_orders) {
    if (node_to_cleans.find(node) != node_to_cleans.end()) {
      auto atomics = node_to_cleans[node];
      (void)std::copy(atomics.begin(), atomics.end(), std::back_inserter(new_orders));
    }
    new_orders.push_back(node);
  }
  kernel_graph->set_execution_order(new_orders);
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
