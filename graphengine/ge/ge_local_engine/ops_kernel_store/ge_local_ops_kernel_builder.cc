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

#include "ge_local_engine/ops_kernel_store/ge_local_ops_kernel_builder.h"
#include <memory>
#include "framework/common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "ge_local_engine/ops_kernel_store/op/op_factory.h"
#include "ge_local_engine/common/constant/constant.h"
#include "register/ops_kernel_builder_registry.h"

namespace ge {
namespace ge_local {
REGISTER_OPS_KERNEL_BUILDER(kGeLocalOpKernelLibName, GeLocalOpsKernelBuilder);

namespace {
const char *const kConstantOpType = "Constant";
const char *const kConstantOpAttrName = "value";
const char *const kDataOpType = "Data";
}  // namespace

GeLocalOpsKernelBuilder::~GeLocalOpsKernelBuilder() {
  GELOGI("GeLocalOpsKernelBuilder destroyed");
}

Status GeLocalOpsKernelBuilder::Initialize(const map<std::string, std::string> &options) {
  return SUCCESS;
}

Status GeLocalOpsKernelBuilder::Finalize() {
  return SUCCESS;
}

Status GeLocalOpsKernelBuilder::CalcOpRunningParam(Node &ge_node) {
  GELOGD("[%s] CalcOpRunningParam In.", ge_node.GetName().c_str());
  OpDescPtr op_desc = ge_node.GetOpDesc();
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "param ge_node has no opdesc, check invalid.");
    GELOGE(FAILED, "[Get][OpDesc] CalcOpRunningParam failed, as op desc is null");
    return FAILED;
  }

  bool is_shape_unknown = false;
  if (NodeUtils::GetNodeUnknownShapeStatus(ge_node, is_shape_unknown) == GRAPH_SUCCESS) {
    if (is_shape_unknown) {
      GELOGI("op:%s is unknown shape, does not need to calc output size.", ge_node.GetName().c_str());
      return SUCCESS;
    }
  }

  const string node_name = ge_node.GetName();
  const string node_type = ge_node.GetType();
  size_t output_size = op_desc->GetOutputsSize();
  GELOGD("Calc op[%s:%s] running param, output size=%zu.", node_name.c_str(), node_type.c_str(), output_size);

  for (size_t i = 0; i < output_size; ++i) {
    GeTensorDesc output_tensor = op_desc->GetOutputDesc(static_cast<uint32_t>(i));
    Format format = output_tensor.GetFormat();
    DataType data_type = output_tensor.GetDataType();

    int64_t mem_size = 0;
    graphStatus graph_status = TensorUtils::GetSize(output_tensor, mem_size);
    // If mem size has been set, no need reset.
    if ((graph_status == GRAPH_SUCCESS) && (mem_size > 0) && (data_type != DT_STRING)) {
      GELOGD("Op[%s:%s] out[%zu] mem size has been set, no need calc again, format=%s, data_type=%s, mem_size=%ld.",
             node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), mem_size);
      continue;
    }

    int64_t output_mem_size = 0;
    GeShape output_shape = output_tensor.GetShape();
    if ((node_type == kConstantOpType) && (data_type == DT_STRING)) {
      graph_status = CalcConstantStrMemSize(op_desc, output_mem_size);
    } else if (node_type == kDataOpType) {
      int64_t o_size = 0;
      graph_status = TensorUtils::GetTensorMemorySizeInBytes(output_tensor, o_size);
      output_mem_size = o_size;
    } else {
      graph_status = TensorUtils::CalcTensorMemSize(output_shape, format, data_type, output_mem_size);
    }

    if (graph_status != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "calc op[%s:%s] out[%zu] mem size failed, format=%s, data_type=%s, error=%u.",
                        node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
                        TypeUtils::DataTypeToSerialString(data_type).c_str(), graph_status);
      GELOGE(FAILED, "[Calc][MemSize] for op[%s:%s] out[%zu] failed, format=%s, data_type=%s, error=%u.",
             node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), graph_status);
      return FAILED;
    }

    if (output_mem_size < 0) {
      REPORT_INNER_ERROR("E19999", "Calc op[%s:%s] out[%zu] mem size is negative(not support),"
                         " format=%s, data_type=%s, mem_size=%ld.",
                         node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
                         TypeUtils::DataTypeToSerialString(data_type).c_str(), output_mem_size);
      GELOGE(FAILED, "[Calc][MemSize] op[%s:%s] out[%zu] mem size is negative(not support),"
             " format=%s, data_type=%s, mem_size=%ld.",
             node_name.c_str(), node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), output_mem_size);
      return FAILED;
    }
    GELOGD(
        "Calc op[%s:%s] out[%zu] mem size is %ld,"
        " format=%s, data_type=%s.",
        node_name.c_str(), node_type.c_str(), i, output_mem_size, TypeUtils::FormatToSerialString(format).c_str(),
        TypeUtils::DataTypeToSerialString(data_type).c_str());

    TensorUtils::SetSize(output_tensor, output_mem_size);

    graph_status = op_desc->UpdateOutputDesc(static_cast<uint32_t>(i), output_tensor);
    if (graph_status != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Update op[%s:%s] out[%zu] desc failed, format=%s, data_type=%s, error=%u.", node_name.c_str(),
             node_type.c_str(), i, TypeUtils::FormatToSerialString(format).c_str(),
             TypeUtils::DataTypeToSerialString(data_type).c_str(), graph_status);
      return FAILED;
    }
  }
  GELOGD("Calc op[%s:%s] running param success.", node_name.c_str(), node_type.c_str());
  return SUCCESS;
}

Status GeLocalOpsKernelBuilder::CalcConstantStrMemSize(const OpDescPtr &op_desc, int64_t &mem_size) {
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "param op_desc is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] CalcConstantStrMemSize failed, as op desc is null");
    return FAILED;
  }
  ConstGeTensorPtr value = MakeShared<const GeTensor>();
  if (value == nullptr) {
    REPORT_CALL_ERROR("E19999", "make shared ConstGeTensor exception.");
    GELOGE(FAILED, "[Create][GeTensor] make shared ConstGeTensor exception.");
    return FAILED;
  }
  // Constant op attr name is "value"
  if (!AttrUtils::GetTensor(op_desc, kConstantOpAttrName, value)) {
    REPORT_CALL_ERROR("E19999", "get op:%s attr value failed", op_desc->GetName().c_str());
    GELOGE(FAILED, "[Get][Value] of Constant op attr failed");
    return FAILED;
  }
  mem_size = static_cast<int64_t>(value->GetData().size());
  return SUCCESS;
}

Status GeLocalOpsKernelBuilder::GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) {
  bool is_shape_unknown = false;
  if (NodeUtils::GetNodeUnknownShapeStatus(node, is_shape_unknown) == GRAPH_SUCCESS) {
    if (is_shape_unknown) {
      GELOGI("op:%s is unknown shape, does not need to generate task",
             node.GetName().c_str());
      return SUCCESS;
    }
  }
  string name = node.GetName();
  string type = node.GetType();
  GELOGD("Ge local generate task for node:%s(%s) begin, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());

  auto op = OpFactory::Instance().CreateOp(node, context);
  if (op == nullptr) {
    REPORT_CALL_ERROR("E19999", "create op for node:%s(%s) failed.", name.c_str(), type.c_str());
    GELOGE(FAILED, "[Create][Op] for node:%s(%s) failed.", name.c_str(), type.c_str());
    return FAILED;
  }

  Status ret = op->Run();
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Node:%s(%s) op run failed.", name.c_str(), type.c_str());
    GELOGE(ret, "[Call][Run] for Node:%s(%s) op failed.", name.c_str(), type.c_str());
    return ret;
  }
  GELOGD("Ge local generate task for node:%s(%s) end, tasks.size()=%zu.", name.c_str(), type.c_str(), tasks.size());
  return ret;
}
}  // namespace ge_local
}  // namespace ge
