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
#include "graph/passes/compile_nodes_pass.h"

#include <utility>
#include <vector>

#include "common/ge/ge_util.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "common/ge_call_wrapper.h"
#include "graph/op_desc.h"

using domi::ImplyType;

namespace {
const char *const kAICPUEngineName = "DNN_VM_AICPU";
const char *const kAICPUKernelLibName = "aicpu_tf_kernel";
}  // namespace

namespace ge {
graphStatus CompileNodesPass::Run(ComputeGraphPtr graph) {
  GE_TIMESTAMP_START(CompileNodesPass);
  GELOGD("[CompileNodesPass]: optimize begin.");
  if (graph == nullptr) {
    return GRAPH_SUCCESS;
  }
  std::shared_ptr<GELib> instance = ge::GELib::GetInstance();
  if (instance == nullptr || !instance->InitFlag()) {
    REPORT_INNER_ERROR("E19999", "Gelib not init before, check invalid");
    GELOGE(ge::GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] Gelib not init before.");
    return ge::GE_CLI_GE_NOT_INITIALIZED;
  }
  std::unordered_map<string, vector<NodePtr>> kernel_to_compile_nodes;
  for (auto &node : graph->GetDirectNode()) {
    if (node == nullptr) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }
    auto node_need_compile = false;
    (void)ge::AttrUtils::GetBool(op_desc, ATTR_NEED_COMPILE, node_need_compile);
    if (!node_need_compile) {
      continue;
    }
    // collect all supported compile node
    string kernel_lib_name;
    auto ret = GetSupportedKernel(node, instance, kernel_lib_name);
    if (ret == GRAPH_SUCCESS) {
      auto iter = kernel_to_compile_nodes.find(kernel_lib_name);
      if (iter != kernel_to_compile_nodes.end()) {
        iter->second.emplace_back(node);
      } else {
        std::vector<NodePtr> node_vec{node};
        kernel_to_compile_nodes.insert(std::make_pair(kernel_lib_name, node_vec));
      }
    } else {
      GELOGE(GRAPH_FAILED, "[Get][SupportedKernel] for node:%s(%s) failed.", node->GetName().c_str(),
             node->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  // compile node follow different kernel, currently only TBE kernel
  auto result = CompileNodes(instance, kernel_to_compile_nodes);
  if (result != GRAPH_SUCCESS) {
    GELOGE(result, "[Compile][Op] failed, ret:%u.", result);
    return result;
  }
  GELOGD("[CompileNodesPass]: Optimize success.");
  GE_TIMESTAMP_EVENT_END(CompileNodesPass, "OptimizeStage2::ControlAttrOptimize::CompileNodesPass");
  return GRAPH_SUCCESS;
}

graphStatus CompileNodesPass::GetSupportedKernel(const NodePtr &node, const std::shared_ptr<GELib> instance,
                                                 string &kernel_lib_name) {
  auto op_desc = node->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "[Get][OpDesc] failed, op of param node is nullptr.");
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }
  // reset op kernel lib, find supported kernel
  kernel_lib_name = op_desc->GetOpKernelLibName();
  if (kernel_lib_name.empty()) {
    (void)instance->DNNEngineManagerObj().GetDNNEngineName(node);
    kernel_lib_name = op_desc->GetOpKernelLibName();
    if (kernel_lib_name.empty()) {
      REPORT_INNER_ERROR("E19999", "kernel_lib_name in op:%s(%s) is empty, check invalid",
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Get][OpKernelLib] for node:%s(%s) failed.", node->GetName().c_str(),
             op_desc->GetType().c_str());
      return GRAPH_FAILED;
    }
  }
  OpsKernelInfoStorePtr kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_lib_name);
  if (kernel_info == nullptr) {
    REPORT_INNER_ERROR("E19999", "Find ops kernel by name:%s failed for op:%s(%s)",
                       kernel_lib_name.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "[Get][OpsKernelInfoStore] for op:%s failed", node->GetName().c_str());
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }

  std::map<std::string, std::string> unsupported_reasons;
  std::string unsupported_reason;
  // begin accuracy supported check
  if (!CheckAccuracySupport(kernel_info, instance, node, unsupported_reason)) {
    // if check accuracy support failed , try to go to other engine.
    GELOGD("Check Accuracy Supported return not support, node name is %s. Try to go to other engine.",
           op_desc->GetName().c_str());
    string kernel_name_origin = kernel_lib_name;
    OpsKernelManager &ops_kernel_manager = instance->OpsKernelManagerObj();
    auto kernel_map = ops_kernel_manager.GetAllOpsKernelInfoStores();
    for (auto it = kernel_map.begin(); it != kernel_map.end(); ++it) {
      string tmp_kernel_name = it->first;
      if (tmp_kernel_name == kernel_name_origin) {
        continue;
      }
      OpsKernelInfoStorePtr tmp_kernel_info = it->second;
      if (CheckAccuracySupport(tmp_kernel_info, instance, node, unsupported_reason)) {
        kernel_lib_name = tmp_kernel_name;
        GELOGD("Find kernel lib %s support node:%s, type:%s , get kernel lib success.", tmp_kernel_name.c_str(),
               node->GetName().c_str(), op_desc->GetType().c_str());
        return GRAPH_SUCCESS;
      } else {
        unsupported_reasons.emplace(tmp_kernel_name, unsupported_reason);
      }
    }
    for (const auto &it : unsupported_reasons) {
      REPORT_INPUT_ERROR("E13002", std::vector<std::string>({"optype", "opskernel", "reason"}),
                         std::vector<std::string>({op_desc->GetType(), it.first, it.second}));
      GELOGE(GE_GRAPH_ASSIGN_ENGINE_FAILED,
             "[Call][CheckAccuracySupport] for Op type %s of ops kernel %s is unsupported, reason:%s",
             op_desc->GetType().c_str(), it.first.c_str(), it.second.c_str());
    }

    REPORT_INPUT_ERROR("E13003", std::vector<std::string>({"opname", "optype"}),
                       std::vector<std::string>({op_desc->GetName(), op_desc->GetType()}));
    GELOGE(GRAPH_FAILED, "[Check][Param] Cannot find kernel lib support node:%s, type:%s , get kernel lib failed.",
           node->GetName().c_str(), op_desc->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool CompileNodesPass::CheckAccuracySupport(
    const OpsKernelInfoStorePtr &kernel_info, const std::shared_ptr<GELib> instance,
    const NodePtr &node, string& unsupported_reason) {
  if (!(kernel_info->CheckAccuracySupported(node, unsupported_reason, true))) {
    return false;
  }
  return true;
}

graphStatus CompileNodesPass::CompileNodes(const std::shared_ptr<GELib> instance,
                                           std::unordered_map<string, vector<NodePtr>> &kernel_to_compile_nodes) {
  // compile nodes, if kernel is aicpu, check support and set engine info.
  OpsKernelInfoStorePtr kernel_info;
  for (auto &kernel_nodes : kernel_to_compile_nodes) {
    kernel_info = instance->OpsKernelManagerObj().GetOpsKernelInfoStore(kernel_nodes.first);
    if (kernel_info == nullptr) {
      REPORT_INNER_ERROR("E19999", "Find ops kernel by name:%s failed", kernel_nodes.first.c_str());
      GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "[Get][OpsKernelInfoStore] for op %s failed", kernel_nodes.first.c_str());
      return ge::GE_GRAPH_PARAM_NULLPTR;
    }
    string reason;
    if (kernel_nodes.first == kAICPUKernelLibName) {
      for (auto node : kernel_nodes.second) {
        // this node will go to aicpu engine ,no need compile
        node->GetOpDesc()->SetOpEngineName(kAICPUEngineName);
        node->GetOpDesc()->SetOpKernelLibName(kAICPUKernelLibName);
        AttrUtils::SetInt(node->GetOpDesc(), ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(ImplyType::AI_CPU));
      }
      continue;
    }
    auto ret = kernel_info->CompileOp(kernel_nodes.second);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Call CompileOp failed, kernel_lib_name:%s, ret:%d",
                        kernel_nodes.first.c_str(), ret);
      GELOGE(ret, "[Compile][Op] failed, kernel name is %s", kernel_nodes.first.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
}  // namespace ge
