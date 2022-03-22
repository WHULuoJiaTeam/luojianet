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

#include "single_op/task/build_task_utils.h"

#include "runtime/rt.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/type_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/types.h"

namespace ge {
namespace {
const uint64_t kSessionId = UINT64_MAX;
uint8_t *kVarBase = nullptr;
const uint64_t kLogicVarBase = 0;
const uint64_t kVarSize = 0;
}

std::vector<std::vector<void *>> BuildTaskUtils::GetAddresses(const OpDescPtr &op_desc,
                                                              const SingleOpModelParam &param,
                                                              bool keep_workspace) {
  std::vector<std::vector<void *>> ret;
  RuntimeParam runtime_para;
  runtime_para.mem_size = param.memory_size;
  runtime_para.logic_mem_base = param.base_addr;
  runtime_para.mem_base = param.mem_base;
  runtime_para.weight_size = param.weight_size;
  runtime_para.logic_weight_base = param.weight_addr;
  runtime_para.weight_base = param.weight_base;
  runtime_para.var_size = kVarSize;
  runtime_para.logic_var_base = kLogicVarBase;
  runtime_para.var_base = kVarBase;
  runtime_para.session_id = kSessionId;
  runtime_para.is_single_op = true;

  ret.emplace_back(ModelUtils::GetInputDataAddrs(runtime_para, op_desc));
  ret.emplace_back(ModelUtils::GetOutputDataAddrs(runtime_para, op_desc));
  if (keep_workspace) {
    ret.emplace_back(ModelUtils::GetWorkspaceDataAddrs(runtime_para, op_desc));
  }
  return ret;
}

std::vector<void *> BuildTaskUtils::JoinAddresses(const std::vector<std::vector<void *>> &addresses) {
  std::vector<void *> ret;
  for (auto &address : addresses) {
    ret.insert(ret.end(), address.begin(), address.end());
  }
  return ret;
}

std::vector<void *> BuildTaskUtils::GetKernelArgs(const OpDescPtr &op_desc,
                                                  const SingleOpModelParam &param) {
  auto addresses = GetAddresses(op_desc, param);
  return JoinAddresses(addresses);
}

std::string BuildTaskUtils::InnerGetTaskInfo(const OpDescPtr &op_desc,
                                             const std::vector<const void *> &input_addrs,
                                             const std::vector<const void *> &output_addrs) {
  std::stringstream ss;
  if (op_desc != nullptr) {
    auto op_type = op_desc->GetType();
    if (op_type == ge::NETOUTPUT || op_type == ge::DATA) {
      return ss.str();
    }
    // Conv2D IN[DT_FLOAT16 NC1HWC0[256, 128, 7, 7, 16],DT_FLOAT16 FRACTAL_Z[128, 32, 16, 16]]
    // OUT[DT_FLOAT16 NC1HWC0[256, 32, 7, 7, 16]]
    ss << op_type << " IN[";
    for (uint32_t idx = 0; idx < op_desc->GetAllInputsSize(); idx++) {
      const GeTensorDescPtr &input = op_desc->MutableInputDesc(idx);
      if (input == nullptr) {
        continue;
      }
      ss << TypeUtils::DataTypeToSerialString(input->GetDataType()) << " ";
      ss << TypeUtils::FormatToSerialString(input->GetFormat());
      ss << VectorToString(input->GetShape().GetDims()) << " ";
      if (idx < input_addrs.size()) {
        ss << input_addrs[idx];
      }
      if (idx < op_desc->GetInputsSize() - 1) {
        ss << ",";
      }
    }
    ss << "] OUT[";

    for (uint32_t idx = 0; idx < op_desc->GetOutputsSize(); idx++) {
      const GeTensorDescPtr &output = op_desc->MutableOutputDesc(idx);
      ss << TypeUtils::DataTypeToSerialString(output->GetDataType()) << " ";
      Format out_format = output->GetFormat();
      const GeShape &out_shape = output->GetShape();
      const auto &dims = out_shape.GetDims();
      ss << TypeUtils::FormatToSerialString(out_format);
      ss << VectorToString(dims) << " ";
      if (idx < output_addrs.size()) {
        ss << output_addrs[idx];
      }
      if (idx < op_desc->GetOutputsSize() - 1) {
        ss << ",";
      }
    }
    ss << "]\n";
  }
  return ss.str();
}

std::string BuildTaskUtils::GetTaskInfo(const OpDescPtr &op_desc) {
  vector<const void *> input_addrs;
  vector<const void *> output_addrs;
  return InnerGetTaskInfo(op_desc, input_addrs, output_addrs);
}

std::string BuildTaskUtils::GetTaskInfo(const OpDescPtr &op_desc,
                                        const std::vector<DataBuffer> &inputs,
                                        const std::vector<DataBuffer> &outputs) {
  vector<const void *> input_addrs;
  vector<const void *> output_addrs;
  GE_CHECK_NOTNULL_EXEC(op_desc, return "");
  if (op_desc->GetAllInputsSize() == inputs.size()) {
    std::for_each(inputs.begin(), inputs.end(), [&](const DataBuffer &db) { input_addrs.push_back(db.data); });
  }
  if (op_desc->GetOutputsSize() == outputs.size()) {
    std::for_each(outputs.begin(), outputs.end(), [&](const DataBuffer &db) { output_addrs.push_back(db.data); });
  }
  return InnerGetTaskInfo(op_desc, input_addrs, output_addrs);
}

std::string BuildTaskUtils::GetTaskInfo(const hybrid::TaskContext &task_context) {
  auto &node_item = task_context.GetNodeItem();
  auto op_desc = node_item.GetOpDesc();
  GE_CHECK_NOTNULL_EXEC(op_desc, return "");
  vector<const void *> input_addrs;
  vector<const void *> output_addrs;
  if (op_desc->GetAllInputsSize() == static_cast<uint32_t>(task_context.NumInputs())) {
    for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
      input_addrs.push_back(task_context.GetInput(i)->GetData());
    }
  }
  if (op_desc->GetOutputsSize() == static_cast<uint32_t>(task_context.NumOutputs())) {
    for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
      output_addrs.push_back(task_context.GetOutput(i)->GetData());
    }
  }
  return InnerGetTaskInfo(op_desc, input_addrs, output_addrs);
}
}  // namespace ge
