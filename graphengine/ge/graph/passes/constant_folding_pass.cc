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

#include "graph/passes/constant_folding_pass.h"

#include <vector>
#include "external/graph/operator_factory.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "init/gelib.h"

namespace ge {
const int64_t kStartCallNum = 1;
const std::string kKernelLibName = "aicpu_tf_kernel";
const std::string kOpsFlagClose = "0";

const map<string, pair<uint64_t, uint64_t>> &ConstantFoldingPass::GetGeConstantFoldingPerfStatistic() const {
  return statistic_of_ge_constant_folding_;
}
const map<string, pair<uint64_t, uint64_t>> &ConstantFoldingPass::GetOpConstantFoldingPerfStatistic() const {
  return statistic_of_op_constant_folding_;
}

Status ConstantFoldingPass::RunOpKernelWithCheck(NodePtr &node, const vector<ConstGeTensorPtr> &inputs,
                                                 std::vector<GeTensorPtr> &outputs) {
  std::shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  if ((instance_ptr == nullptr) || (!instance_ptr->InitFlag())) {
    GELOGE(GE_CLI_GE_NOT_INITIALIZED, "[Check][Param] GE is not initialized or is finalized.");
    return UNSUPPORTED;
  }
  OpsKernelInfoStorePtr kernel_info = instance_ptr->OpsKernelManagerObj().GetOpsKernelInfoStore(kKernelLibName);
  if (kernel_info == nullptr) {
    GELOGE(FAILED, "[Get][OpsKernelInfoStore] %s failed", kKernelLibName.c_str());
    return UNSUPPORTED;
  }

  std::string ops_flag;
  kernel_info->opsFlagCheck(*node, ops_flag);
  if (ops_flag == kOpsFlagClose) {
    return UNSUPPORTED;
  }
  return RunOpKernel(node, inputs, outputs);
}

Status ConstantFoldingPass::RunOpKernel(NodePtr &node,
                                        const vector<ConstGeTensorPtr> &inputs,
                                        std::vector<GeTensorPtr> &outputs) {
  return HostCpuEngine::GetInstance().Run(node, inputs, outputs);
}

Status ConstantFoldingPass::Run(ge::NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  GELOGD("Begin to run constant folding on node %s", node->GetName().c_str());

  if (folding_pass::IsNoNeedConstantFolding(node)) {
    return SUCCESS;
  }
  OpDescPtr node_desc = node->GetOpDesc();
  DataType data_type = node_desc->GetOutputDesc(0).GetDataType();
  Format format = node_desc->GetOutputDesc(0).GetFormat();
  GELOGD("Current [node:%s, type:%s] info: format: %s, datatype:%s", node->GetName().c_str(), node->GetType().c_str(),
         TypeUtils::FormatToSerialString(format).c_str(), TypeUtils::DataTypeToSerialString(data_type).c_str());
  auto input_nodes = OpDescUtils::GetConstInputNode(*node);
  if (input_nodes.empty() || input_nodes.size() != node_desc->GetInputsSize()) {
    GELOGD("Node:%s, const input nodes size is %zu, and nodeDesc inputsSize is %zu.", node->GetName().c_str(),
           input_nodes.size(), node_desc->GetInputsSize());
    return SUCCESS;
  }

  auto inputs = OpDescUtils::GetInputData(input_nodes);
  vector<GeTensorPtr> outputs;
  // Statistic of ge constant folding kernel
  uint64_t start_time = GetCurrentTimestamp();
  auto ret = RunOpKernelWithCheck(node, inputs, outputs);
  if (ret != SUCCESS) {
    auto op_kernel = folding_pass::GetKernelByType(node);
    if (op_kernel == nullptr) {
      GELOGD("No op kernel for node %s type %s, skip the constant folding", node->GetName().c_str(),
             node->GetType().c_str());
      return SUCCESS;
    }

    // Statistic of op and fe constant folding kernel
    start_time = GetCurrentTimestamp();
    ret = op_kernel->Compute(node_desc, inputs, outputs);
    uint64_t cost_time = GetCurrentTimestamp() - start_time;
    if (statistic_of_ge_constant_folding_.find(node->GetType()) != statistic_of_ge_constant_folding_.end()) {
      uint64_t &cnt = statistic_of_ge_constant_folding_[node->GetType()].first;
      uint64_t &cur_cost_time = statistic_of_ge_constant_folding_[node->GetType()].second;
      cnt++;
      cur_cost_time += cost_time;
    } else {
      statistic_of_ge_constant_folding_[node->GetType()] = std::pair<uint64_t, uint64_t>(kStartCallNum, cost_time);
    }
    if (ret != SUCCESS) {
      if (ret == NOT_CHANGED) {
        GELOGD("Node %s type %s, compute terminates and exits the constant folding.", node->GetName().c_str(),
               node->GetType().c_str());
        return SUCCESS;
      }
      REPORT_CALL_ERROR("E19999", "Calculate for node %s(%s) failed",
                        node->GetName().c_str(), node->GetType().c_str());
      GELOGE(INTERNAL_ERROR, "[Call][Calculate] for node %s failed in constant folding", node->GetName().c_str());
      return ret;
    }
    GELOGI("Node %s type %s, constant folding compute success.", node->GetName().c_str(), node->GetType().c_str());
  } else {
    if (statistic_of_op_constant_folding_.find(node->GetType()) != statistic_of_op_constant_folding_.end()) {
      uint64_t &cnt = statistic_of_op_constant_folding_[node->GetType()].first;
      uint64_t &cost_time = statistic_of_op_constant_folding_[node->GetType()].second;
      cnt++;
      cost_time += GetCurrentTimestamp() - start_time;
    } else {
      statistic_of_op_constant_folding_[node->GetType()] =
          std::pair<uint64_t, uint64_t>(kStartCallNum, GetCurrentTimestamp() - start_time);
    }
  }

  if (outputs.empty()) {
    REPORT_INNER_ERROR("E19999", "After calculate for node %s(%s), output weight is empty, check invalid",
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] After calculate for node %s(%s), output weight is empty",
           node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }

  return Folding(node, outputs);
}
}  // namespace ge
