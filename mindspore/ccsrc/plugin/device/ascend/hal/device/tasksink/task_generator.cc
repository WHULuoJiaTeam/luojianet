/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/tasksink/task_generator.h"

#include <runtime/rt.h>
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/task_stream.h"
#include "include/common/utils/utils.h"
#include "utils/ms_utils.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/ascend/hal/device/profiling/profiling_utils.h"
#include "plugin/device/ascend/hal/device/profiling/profiling_manager.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/task_debug_info_recorder.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
bool TaskGenerator::GenTasks(const std::vector<CNodePtr> &anf_node_list, std::vector<TaskInfoPtr> *task_info_list,
                             uint32_t graph_id) {
  MS_LOG(INFO) << "GenTasks start...";
  MS_EXCEPTION_IF_NULL(task_info_list);
  // Traverse graph applykernel list and run
  if (!LaunchAllKernel(anf_node_list, task_info_list, graph_id)) {
    MS_LOG(ERROR) << "LaunchAllKernel failed";
    return false;
  }
  MS_LOG(INFO) << "GenTasks end...";
#ifdef ENABLE_DUMP_IR
  string task_info_name = "task_info_graph." + std::to_string(graph_id);
  (void)mindspore::RDR::RecordTaskDebugInfo(SUBMODULE_ID, task_info_name, task_debug_info_list_);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
#ifndef ENABLE_SECURITY
    std::string file_path = GetSaveGraphsPathName("task_info_graph_" + std::to_string(graph_id) + ".ir");
    DumpTaskInfo(file_path);
#endif
  }
#endif
  return true;
}

void TaskGenerator::LaunchAddrCleanAkgKernel(const CNodePtr &anf_node_ptr, AddressPtrList *kernel_inputs) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  // akg process
  // set atomic clean addr
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, anf_node_ptr)) {
    auto clean_output_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(anf_node_ptr, kAttrAtomicOutputIndexs);
    auto graph = anf_node_ptr->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &node_users = manager->node_users();
    if (node_users[anf_node_ptr].empty()) {
      MS_LOG(EXCEPTION) << "Node users of " << anf_node_ptr->ToString() << " is empty.";
    }
    auto depend_node = node_users[anf_node_ptr].pop().first;
    if (!IsPrimitiveCNode(depend_node, prim::kPrimDepend)) {
      MS_LOG(EXCEPTION) << "Checking Depend node failed";
    }
    if (node_users[depend_node].empty()) {
      MS_LOG(EXCEPTION) << "Node users of " << depend_node->ToString() << " is empty.";
    }
    auto post_node = node_users[depend_node].pop().first;
    for (auto index : clean_output_indexs) {
      auto device_address = AnfAlgo::GetOutputAddr(post_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      MS_EXCEPTION_IF_NULL(device_address);
      input->addr = device_address->ptr_;
      input->size = device_address->size_;
      kernel_inputs->push_back(input);
    }
    MS_LOG(DEBUG) << "AtomicAddClean clean output size: " << clean_output_indexs.size();
  }
}

void TaskGenerator::LaunchAddrCleanKernel(const CNodePtr &anf_node_ptr, AddressPtrList *kernel_inputs) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  // akg process
  if (AnfAlgo::GetKernelType(anf_node_ptr) == KernelType::AKG_KERNEL) {
    LaunchAddrCleanAkgKernel(anf_node_ptr, kernel_inputs);
    return;
  }
  // tbe process
  auto input_tensor_num = common::AnfAlgo::GetInputTensorNum(anf_node_ptr);
  for (size_t i = 0; i < input_tensor_num; i++) {
    // set clean output addr
    MS_EXCEPTION_IF_NULL(anf_node_ptr->inputs()[i + 1]);
    auto pre_node = anf_node_ptr->input(i + 1)->cast<CNodePtr>();
    if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
      auto clean_output_indexs = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
      for (auto index : clean_output_indexs) {
        auto device_address = AnfAlgo::GetOutputAddr(pre_node, index);
        kernel::AddressPtr input = std::make_shared<kernel::Address>();
        MS_EXCEPTION_IF_NULL(input);
        MS_EXCEPTION_IF_NULL(device_address);
        input->addr = device_address->ptr_;
        MS_EXCEPTION_IF_NULL(input->addr);
        input->size = device_address->size_;
        kernel_inputs->push_back(input);
      }
      MS_LOG(DEBUG) << "AtomicAddClean clean output size:" << clean_output_indexs.size();
    }
    // set clean workspace address
    if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
      auto clean_workspace_indexs =
        common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
      for (const auto &index : clean_workspace_indexs) {
        auto device_address = AnfAlgo::GetWorkspaceAddr(pre_node, index);
        kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
        MS_EXCEPTION_IF_NULL(workspace);
        MS_EXCEPTION_IF_NULL(device_address);
        workspace->addr = device_address->ptr_;
        MS_EXCEPTION_IF_NULL(workspace->addr);
        workspace->size = device_address->size_;
        kernel_inputs->push_back(workspace);
      }
      MS_LOG(DEBUG) << "AtomicAddClean clean workspace size:" << clean_workspace_indexs.size();
    }
  }
  auto clear_mems = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(anf_node_ptr, kAttrAtomicAddMemSize);
  if (kernel_inputs->size() != clear_mems.size()) {
    MS_LOG(EXCEPTION) << "AtomicAddClean kernel inputs size not equal clear memory size, kernel inputs size:"
                      << kernel_inputs->size() << ",clean mem size" << clear_mems.size();
  }
}

bool TaskGenerator::LaunchKernel(const CNodePtr &anf_node_ptr, uint32_t stream_id,
                                 std::vector<TaskInfoPtr> *task_info_list) {
  MS_EXCEPTION_IF_NULL(task_info_list);
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  AddressPtrList kernel_inputs;
  AddressPtrList kernel_workspaces;
  AddressPtrList kernel_outputs;
  auto kernel_mod = AnfAlgo::GetKernelMod(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel_mod->set_unique_name(anf_node_ptr->UniqueName());
  kernel_mod->set_fullname(anf_node_ptr->fullname_with_scope());
  kernel_mod->set_is_monad(common::AnfAlgo::IsNodeInputContainMonad(anf_node_ptr) && HasAbstractMonad(anf_node_ptr));
  auto op_name = common::AnfAlgo::GetCNodeName(anf_node_ptr);
  if (common::AnfAlgo::IsNonTaskOp(anf_node_ptr)) {
    MS_LOG(INFO) << "Skip task generation for NonTask op " << anf_node_ptr->fullname_with_scope();
    auto debug_info = std::make_shared<TaskDebugInfo>();
    MS_EXCEPTION_IF_NULL(debug_info);
    debug_info->op_name_ = anf_node_ptr->fullname_with_scope() + "-NonTask";
    debug_info->task_num_ = 0;
    task_debug_info_list_.push_back(debug_info);
    return true;
  }

  if (op_name != kAtomicAddrCleanOpName) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(anf_node_ptr);
    for (size_t i = 0; i < input_num; ++i) {
      if (common::AnfAlgo::IsNoneInput(anf_node_ptr, i)) {
        continue;
      }
      auto real_input_index = AnfAlgo::GetRealInputIndex(anf_node_ptr, i);
      auto device_address = AnfAlgo::GetPrevNodeOutputAddr(anf_node_ptr, real_input_index);
      AddressPtr input = std::make_shared<Address>();
      MS_EXCEPTION_IF_NULL(input);
      input->addr = device_address->ptr_;
      input->size = device_address->size_;

      auto prenode_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node_ptr, i);
      MS_EXCEPTION_IF_NULL(prenode_with_index.first);
      if (AnfUtils::IsRealCNodeKernel(prenode_with_index.first)) {
        if (common::AnfAlgo::IsNonTaskOp(prenode_with_index.first->cast<CNodePtr>())) {
          // use memory offset to implement NonTask Type Split op
          // when op A -> split(NonTask) -> op B, op B's input addr is split's input0's addr + offset
          // offset is split's output index * split's output size
          auto split_input0_device_address = AnfAlgo::GetPrevNodeOutputAddr(prenode_with_index.first, 0);
          MS_EXCEPTION_IF_NULL(split_input0_device_address);
          input->addr =
            static_cast<uint8_t *>(split_input0_device_address->ptr_) + (prenode_with_index.second * input->size);
          MS_LOG(INFO) << "Change " << anf_node_ptr->fullname_with_scope() << "'s input " << i << " address to "
                       << split_input0_device_address->ptr_ << " + " << prenode_with_index.second * input->size;
        }
      }
      kernel_inputs.push_back(input);
    }

    // No kernel output if output of the cnode is monad, such as LabelSwitch.
    if (!HasAbstractMonad(anf_node_ptr)) {
      size_t output_num = common::AnfAlgo::GetOutputTensorNum(anf_node_ptr);
      for (size_t i = 0; i < output_num; ++i) {
        auto it = AnfAlgo::GetOutputAddr(anf_node_ptr, i, false);
        AddressPtr output = std::make_shared<Address>();
        output->addr = it->ptr_;
        output->size = it->size_;
        kernel_outputs.push_back(output);
      }
    }

    for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(anf_node_ptr, i);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      workspace->addr = device_address->ptr_;
      workspace->size = device_address->size_;
      kernel_workspaces.push_back(workspace);
    }
  } else {
    LaunchAddrCleanKernel(anf_node_ptr, &kernel_inputs);
  }

  auto ascend_kernel_mod = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
  MS_EXCEPTION_IF_NULL(ascend_kernel_mod);
  std::vector<TaskInfoPtr> task_info_ptrs =
    ascend_kernel_mod->GenTask(kernel_inputs, kernel_workspaces, kernel_outputs, stream_id);
  task_info_list->insert(task_info_list->end(), task_info_ptrs.begin(), task_info_ptrs.end());
  auto debug_info = std::make_shared<TaskDebugInfo>();
  MS_EXCEPTION_IF_NULL(debug_info);
  if (task_info_ptrs.empty()) {
    MS_LOG(ERROR) << "Empty task_info_ptrs.";
    return false;
  }
  debug_info->op_name_ = anf_node_ptr->fullname_with_scope();
  debug_info->task_num_ = task_info_ptrs.size();
  debug_info->stream_id_ = task_info_ptrs[0]->stream_id();
  debug_info->dump_flag_ = task_info_ptrs[0]->dump_flag();
  debug_info->input_addrs_ = kernel_inputs;
  debug_info->output_addrs_ = kernel_outputs;
  debug_info->workspace_addrs_ = kernel_workspaces;
  task_debug_info_list_.push_back(debug_info);
  return true;
}

bool TaskGenerator::LaunchAllKernel(const std::vector<CNodePtr> &anf_node_list,
                                    std::vector<TaskInfoPtr> *task_info_list, uint32_t graph_id) {
  uint32_t current_op_index = 0;
  std::vector<CNodePtr> profiling_cnode_list;
  std::vector<std::string> kernel_name_list;
  for (const auto &anf_node_ptr : anf_node_list) {
    size_t old_size = task_info_list->size();
    uint32_t stream_id = AnfAlgo::GetStreamId(anf_node_ptr);
    MS_EXCEPTION_IF_NULL(anf_node_ptr);
    MS_LOG(DEBUG) << "Task gen launch begin, current_op_idx:" << current_op_index
                  << " name:" << anf_node_ptr->fullname_with_scope() << ", stream id:" << stream_id;
    if (!LaunchKernel(anf_node_ptr, stream_id, task_info_list)) {
      MS_LOG(ERROR) << "LaunchKernel failed.";
      return false;
    }
    for (size_t i = old_size; i < task_info_list->size(); ++i) {
      profiling_cnode_list.emplace_back(anf_node_ptr);
      kernel_name_list.emplace_back(anf_node_ptr->fullname_with_scope());
    }
    current_op_index++;
  }

#ifndef ENABLE_SECURITY
  ProfilingUtils::SetGraphKernelName(graph_id, kernel_name_list);
  if (ProfilingManager::GetInstance().IsProfilingInitialized()) {
    ProfilingUtils::SetGraphProfilingCNode(graph_id, profiling_cnode_list);
  }
#endif

  return true;
}

#ifdef ENABLE_DUMP_IR
void TaskGenerator::DumpTaskInfo(const string &real_filename,
                                 const std::vector<TaskDebugInfoPtr> &task_debug_info_list) {
  ChangeFileMode(real_filename, S_IRWXU);
  SaveTaskDebugInfoToFile(real_filename, task_debug_info_list);
  // set file mode to read only by user
  ChangeFileMode(real_filename, S_IRUSR);
}

void TaskGenerator::DumpTaskInfo(const std::string &real_filename) {
  if (real_filename.size() >= PATH_MAX) {
    MS_LOG(ERROR) << "File path " << real_filename << " is too long.";
    return;
  }
  char real_path[PATH_MAX] = {0};
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, filename.c_str(), PATH_MAX) == nullptr) {
    MS_LOG(DEBUG) << "dir " << filename << " does not exit.";
  }
#else
  if (realpath(real_filename.c_str(), real_path) == nullptr) {
    MS_LOG(DEBUG) << "Dir " << real_filename << " does not exit.";
  }
#endif

  std::string path_string = real_path;
  ChangeFileMode(path_string, S_IRWXU);
  SaveTaskDebugInfoToFile(path_string, task_debug_info_list_);
  // set file mode to read only by user
  ChangeFileMode(path_string, S_IRUSR);
}
#else
void TaskGenerator::DumpTaskInfo(const std::string &real_filename) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping task debug info is disabled, "
                  << "please recompile the source codes with '-D on' option.";
}
void TaskGenerator::DumpTaskInfo(const string &real_filename,
                                 const std::vector<TaskDebugInfoPtr> &task_debug_info_list) {
  static bool already_printed = false;
  if (already_printed) {
    return;
  }
  already_printed = true;
  MS_LOG(WARNING) << "The functionality of dumping task debug info is disabled, "
                  << "please recompile the source codes with '-D on' option.";
}
#endif

void TaskGenerator::SaveTaskDebugInfoToFile(const std::string &real_filename,
                                            const std::vector<TaskDebugInfoPtr> &task_debug_info_list) {
  std::ofstream fout(real_filename);

  if (!fout.is_open()) {
    MS_LOG(ERROR) << "Open dump file '" << real_filename << "' failed!";
    return;
  }

  size_t index = 0;
  for (auto &task_debug_info : task_debug_info_list) {
    MS_EXCEPTION_IF_NULL(task_debug_info);
    fout << "op_name:" << task_debug_info->op_name_ << "\n"
         << "task_index:" << index << "\t"
         << "task_num:" << task_debug_info->task_num_ << "\t"
         << "task0_stream_id:" << task_debug_info->stream_id_ << "\t"
         << "task0_type:" << task_debug_info->type_ << "\t"
         << "task0_dump_flag:" << task_debug_info->dump_flag_ << "\n";
    index++;
    if (!task_debug_info->input_addrs_.empty()) {
      fout << "input address:";
      for (auto &input : task_debug_info->input_addrs_) {
        MS_EXCEPTION_IF_NULL(input);
        fout << input->addr << "(" << input->size << ")\t";
      }
      fout << "\n";
    }

    if (!task_debug_info->output_addrs_.empty()) {
      fout << "output address:";
      for (auto &output : task_debug_info->output_addrs_) {
        MS_EXCEPTION_IF_NULL(output);
        fout << output->addr << "(" << output->size << ")\t";
      }
      fout << "\n";
    }

    if (!task_debug_info->workspace_addrs_.empty()) {
      fout << "workspace address:";
      for (auto &workspace : task_debug_info->workspace_addrs_) {
        MS_EXCEPTION_IF_NULL(workspace);
        fout << workspace->addr << "(" << workspace->size << ")\t";
      }
      fout << "\n";
    }
    fout << "\n";
  }

  fout.close();
}
}  // namespace tasksink
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
