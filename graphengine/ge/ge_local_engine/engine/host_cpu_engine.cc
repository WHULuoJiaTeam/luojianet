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
#include "ge_local_engine/engine/host_cpu_engine.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_adapter.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_kernel_registry.h"
#include "register/host_cpu_context.h"
#include "common/ge/ge_util.h"
#include "common/ge/plugin_manager.h"
#include "common/fp16_t.h"
#include "common/math/math_util.h"

namespace {
#define CREATE_OUTPUT_CASE(DTYPE)                                                                                      \
  case (DTYPE): {                                                                                                      \
    GeTensorPtr ge_tensor = nullptr;                                                                                   \
    if (need_create_flag) {                                                                                            \
      int64_t size = ge::GetSizeInBytes(static_cast<int64_t>(data_num), DTYPE);                                        \
      if (size < 0) {                                                                                                  \
        return INTERNAL_ERROR;                                                                                         \
      }                                                                                                                \
      ge_tensor = MakeShared<GeTensor>(out_desc, static_cast<size_t>(size));                                           \
      GE_CHECK_NOTNULL(ge_tensor);                                                                                     \
      GELOGD("node:%s allocate output %zu success, size=%ld", op_desc->GetName().c_str(), i, size);                    \
      ge_tensor->MutableTensorDesc().SetDataType(out_desc.GetDataType());                                              \
      ge_tensor->MutableTensorDesc().SetShape(out_desc.GetShape());                                                    \
    } else {                                                                                                           \
      ge_tensor = outputs[i];                                                                                          \
      GE_CHECK_NOTNULL(ge_tensor);                                                                                     \
      GELOGD("node:%s existed output %zu", op_desc->GetName().c_str(), i);                                             \
    }                                                                                                                  \
    auto tensor = TensorAdapter::AsTensor(*ge_tensor);                                                                 \
    auto tensor_name = op_desc->GetOutputNameByIndex(i);                                                               \
    GE_RETURN_WITH_LOG_IF_TRUE(tensor_name.empty(), "[Get][OutputName] failed. node = %s, index = %zu",                \
                               op_desc->GetName().c_str(), i);                                                         \
    named_outputs.emplace(tensor_name, tensor);                                                                        \
    break;                                                                                                             \
  }
}

namespace ge {
namespace {
const char *kEnvKeyOppPath = "ASCEND_OPP_PATH";
const char *kHostCpuLibRelativePath = "/op_impl/built-in/host_cpu";
const std::string kConstantFoldingName = "libconstant_folding_ops.so";
}

Status GetDataNumber(const GeTensorDesc &out_desc, uint64_t &data_num) {
  int64_t num_size = out_desc.GetShape().IsScalar() ? 1 : out_desc.GetShape().GetShapeSize();
  if (out_desc.GetShape().IsUnknownShape()) {
    std::vector<std::pair<int64_t, int64_t>> range;
    if (out_desc.GetShapeRange(range) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "GetShapeRange failed.");
      GELOGE(INTERNAL_ERROR, "[Get][ShapeRange] failed.");
      return INTERNAL_ERROR;
    }
    int64_t max_range_size = 1;
    for (const auto& item : range) {
      FMK_INT64_MULCHECK(max_range_size, item.second);
      max_range_size *= item.second;
    }
    num_size = max_range_size;
  }
  if (num_size < 0) {
    REPORT_INNER_ERROR("E19999", "Get negative size, num_size=%ld.", num_size);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Get negative size, num_size=%ld.", num_size);
    return INTERNAL_ERROR;
  }
  data_num = static_cast<uint64_t>(num_size);
  return SUCCESS;
}

void HostCpuEngine::CloseSo() {
  for (auto handle : lib_handles_) {
    if (mmDlclose(handle) != 0) {
      const char *error = mmDlerror();
      error = (error == nullptr) ? "" : error;
      GELOGW("failed to close handle, message: %s", error);
    }
  }
  lib_handles_.clear();
}

ge::Status HostCpuEngine::Initialize() {
  std::lock_guard<std::mutex> lock(mu_);
  if (initialized_) {
      GELOGI("HostCpuEngine is already initialized");
      return SUCCESS;
  }
  std::string lib_dir;
  GE_CHK_STATUS_RET_NOLOG(GetLibPath(lib_dir));

  std::vector<std::string> so_paths;
  if (ListSoFiles(lib_dir, so_paths) == SUCCESS) {
    (void) LoadLibs(so_paths);
  }

  initialized_ = true;
  return SUCCESS;
}

void HostCpuEngine::Finalize() {
  GELOGI("start HostCpuEngine::Finalize");
}

bool HostCpuEngine::CheckSupported(const string &op_type) {
  return OpKernelRegistry::GetInstance().IsRegistered(op_type);
}

Status HostCpuEngine::FindOpKernel(const ge::NodePtr &node, std::unique_ptr<HostCpuOp> &op_kernel) {
  const std::string op_type = NodeUtils::GetNodeType(node);
  auto kernel = OpKernelRegistry::GetInstance().CreateHostCpuOp(op_type);
  if (kernel == nullptr) {
    GELOGD("Op of type %s is not supported by host cpu engine", op_type.c_str());
    return UNSUPPORTED;
  }

  GELOGD("Successfully created op kernel. op type = %s", op_type.c_str());
  op_kernel = std::move(kernel);
  return SUCCESS;
}

Status HostCpuEngine::PrepareInputs(const ge::ConstOpDescPtr &op_desc,
                                    const vector<ConstGeTensorPtr> &inputs,
                                    map<std::string, const Tensor> &named_inputs) {
  auto num_inputs = op_desc->GetInputsSize();
  if (num_inputs != inputs.size()) {
    REPORT_INNER_ERROR("E19999", "Mismatching input sizes. op_desc:%s(%s) has %zu input(s), but given %zu",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), num_inputs, inputs.size());
    GELOGE(PARAM_INVALID, "[Check][Param] Mismatching input sizes. op_desc:%s(%s) has %zu input(s), but given %zu",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), num_inputs, inputs.size());
    return PARAM_INVALID;
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    auto ge_tensor = inputs[i];
    GE_CHECK_NOTNULL(ge_tensor);
    auto tensor = TensorAdapter::AsTensor(*ge_tensor);
    auto tensor_name = op_desc->GetInputNameByIndex(i);
    GE_RETURN_WITH_LOG_IF_TRUE(tensor_name.empty(), "[Get][InputName] failed. node = %s, index = %zu",
                               op_desc->GetName().c_str(), i);
    GELOGD("Successfully inserted input tensor. node = %s, index = %zu, input name = %s",
           op_desc->GetName().c_str(), i, tensor_name.c_str());
    named_inputs.emplace(tensor_name, tensor);
  }

  return SUCCESS;
}

Status HostCpuEngine::PrepareOutputs(const ge::ConstOpDescPtr &op_desc,
                                     vector<GeTensorPtr> &outputs,
                                     map<std::string, Tensor> &named_outputs) {
  if (!outputs.empty() && (outputs.size() != op_desc->GetOutputsSize())) {
    GELOGW("size of outputs not match, size of outputs = %zu, exactly output_num=%zu.",
           outputs.size(), op_desc->GetOutputsSize());
    outputs.clear();
  }
  bool need_create_flag = (outputs.size() != op_desc->GetOutputsSize());
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    const auto &out_desc = op_desc->GetOutputDesc(i);
    uint64_t data_num = 0;
    if (need_create_flag) {
      if (GetDataNumber(out_desc, data_num) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Get][Number] node:%s get size for output %zu failed", op_desc->GetName().c_str(), i);
        return INTERNAL_ERROR;
      }
    }
    switch (out_desc.GetDataType()) {
      CREATE_OUTPUT_CASE(DT_BOOL)
      CREATE_OUTPUT_CASE(DT_INT8)
      CREATE_OUTPUT_CASE(DT_INT16)
      CREATE_OUTPUT_CASE(DT_INT32)
      CREATE_OUTPUT_CASE(DT_INT64)
      CREATE_OUTPUT_CASE(DT_UINT8)
      CREATE_OUTPUT_CASE(DT_UINT16)
      CREATE_OUTPUT_CASE(DT_UINT32)
      CREATE_OUTPUT_CASE(DT_UINT64)
      CREATE_OUTPUT_CASE(DT_FLOAT16)
      CREATE_OUTPUT_CASE(DT_FLOAT)
      CREATE_OUTPUT_CASE(DT_DOUBLE)
      CREATE_OUTPUT_CASE(DT_INT4)
      default:
        GELOGW("data type %s not support.",
               TypeUtils::DataTypeToSerialString(out_desc.GetDataType()).c_str());
        return NOT_CHANGED;
    }
  }

  return SUCCESS;
}

Status HostCpuEngine::RunInternal(const ge::OpDescPtr &op_desc,
                                  HostCpuOp &op_kernel,
                                  map<std::string, const Tensor> &named_inputs,
                                  map<std::string, Tensor> &named_outputs) {
  GELOGD("Run operation on host cpu, op name: %s", op_desc->GetName().c_str());
  Operator op = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  auto ret = op_kernel.Compute(op, named_inputs, named_outputs);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("Failed to compute host cpu op. node = %s", op_desc->GetName().c_str());
    return FAILED;
  }
  op.BreakConnect();

  return SUCCESS;
}

Status HostCpuEngine::Run(NodePtr &node, const vector<ConstGeTensorPtr> &inputs, std::vector<GeTensorPtr> &outputs) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());

  GELOGD("Run node by host cpu engine. node name = %s", node->GetName().c_str());
  std::unique_ptr<HostCpuOp> op_kernel;
  GE_CHK_STATUS_RET_NOLOG(FindOpKernel(node, op_kernel));
  std::map<std::string, const Tensor> named_inputs;
  std::map<std::string, Tensor> named_outputs;
  auto op_desc = node->GetOpDesc();
  GE_CHK_STATUS_RET_NOLOG(PrepareInputs(op_desc, inputs, named_inputs));
  GE_CHK_STATUS_RET_NOLOG(PrepareOutputs(op_desc, outputs, named_outputs));
  GE_CHK_STATUS_RET_NOLOG(RunInternal(op_desc, *op_kernel, named_inputs, named_outputs));

  std::vector<GeTensorPtr> tmp_outputs;
  for (size_t i = 0; i < op_desc->GetOutputsSize(); i++) {
    auto tensor_name = op_desc->GetOutputNameByIndex(i);
    if (tensor_name.empty()) {
      REPORT_INNER_ERROR("E19999", "GetOutputNameByIndex failed, node = %s, index = %zu",
                         op_desc->GetName().c_str(), i);
      GELOGE(INTERNAL_ERROR, "[Get][OutputName] failed. node = %s, index = %zu", op_desc->GetName().c_str(), i);
      return INTERNAL_ERROR;
    }
    auto iter = named_outputs.find(tensor_name);
    if (iter == named_outputs.end()) {
       REPORT_INNER_ERROR("E19999", "get output tensor failed, node = %s, index = %zu, tensor_name = %s",
                          op_desc->GetName().c_str(), i, tensor_name.c_str());
       GELOGE(INTERNAL_ERROR, "[Get][OutputTensor] failed. node = %s, index = %zu, tensor_name = %s",
              op_desc->GetName().c_str(), i, tensor_name.c_str());
      return INTERNAL_ERROR;
    }
    auto ge_tensor = MakeShared<GeTensor>(TensorAdapter::AsGeTensor(iter->second));
    GE_CHECK_NOTNULL(ge_tensor);
    tmp_outputs.emplace_back(ge_tensor);
  }
  GELOGD("Run node by host cpu engine successfully. name node = %s", node->GetName().c_str());
  outputs.swap(tmp_outputs);
  return SUCCESS;
}

ge::Status HostCpuEngine::GetLibPath(std::string &lib_path) {
  GELOGI("Start to get host cpu lib path");
  const char *path_env = std::getenv(kEnvKeyOppPath);
  if (path_env != nullptr) {
    lib_path = path_env;
    if (!lib_path.empty()) {
      lib_path += kHostCpuLibRelativePath;
      GELOGI("Get host cpu so path from env: %s", lib_path.c_str());
      return SUCCESS;
    }
  }

  lib_path = PluginManager::GetPath();
  GELOGI("path_base is %s", lib_path.c_str());
  lib_path = lib_path.substr(0, lib_path.rfind('/'));
  lib_path = lib_path.substr(0, lib_path.rfind('/'));
  lib_path += "/opp";
  lib_path += kHostCpuLibRelativePath;

  GELOGI("Get host cpu so path from PluginManager::GetPath: %s", lib_path.c_str());
  return SUCCESS;
}

static int RegularFileFilterFn(const mmDirent *entry) {
  return entry->d_type == DT_REG;
}

Status HostCpuEngine::ListSoFiles(const std::string &base_dir, std::vector<std::string> &names) {
  std::string real_path = base_dir;
  GE_CHK_STATUS_RET_NOLOG(GetRealPath(real_path));
  real_path.push_back('/');
  mmDirent **entries = nullptr;
  auto ret = mmScandir(real_path.c_str(), &entries, RegularFileFilterFn, nullptr);
  if (ret < 0) {
    GELOGW("scan dir failed. path = %s, ret = %d, errmsg = %s", real_path.c_str(), ret, strerror(errno));
    return INTERNAL_ERROR;
  }

  for (int i = 0; i < ret; ++i) {
    mmDirent *dir_ent = entries[i];
    string name = string(dir_ent->d_name);
    if (IsSoFile(name)) {
      names.emplace_back(real_path + name);
    }
  }

  mmScandirFree(entries, ret);
  GELOGI("Found %d libs to load", ret);
  return SUCCESS;
}

bool HostCpuEngine::IsSoFile(const std::string &file_name) {
  static const std::string so_suffix(".so");
  auto pos = file_name.rfind(so_suffix);
  if (pos == string::npos) {
    return false;
  }

  return pos == file_name.size() - so_suffix.size();
}

Status HostCpuEngine::LoadLibs(std::vector<std::string> &lib_paths) {
  for (auto &so_path : lib_paths) {
    GE_CHK_STATUS_RET_NOLOG(GetRealPath(so_path));
    GE_CHK_STATUS_RET_NOLOG(LoadLib(so_path));
  }

  return SUCCESS;
}

Status HostCpuEngine::LoadLib(const std::string &lib_path) {
  GELOGI("To invoke dlopen on lib: %s", lib_path.c_str());
  auto handle = mmDlopen(lib_path.c_str(), MMPA_RTLD_NOW | MMPA_RTLD_GLOBAL);
  if (handle == nullptr) {
    const char *error = mmDlerror();
    error = (error == nullptr) ? "" : error;
    REPORT_CALL_ERROR("E19999", "mmDlopen failed, path = %s, error = %s", lib_path.c_str(), error);
    GELOGE(INTERNAL_ERROR, "[Invoke][DlOpen] failed. path = %s, error = %s", lib_path.c_str(), error);
    return INTERNAL_ERROR;
  }

  auto initialize = (Status (*)(const HostCpuContext &))mmDlsym(handle, "Initialize");
  if (initialize != nullptr) {
    GELOGI("Invoke function Initialize in lib: %s", lib_path.c_str());
    if (initialize(HostCpuContext()) != SUCCESS) {
      GELOGW("Failed to invoke function Initialize in lib: %s", lib_path.c_str());
    }
  }

  GELOGI("Lib: %s has been opened", lib_path.c_str());
  if (lib_path.find(kConstantFoldingName) != lib_path.npos) {
    constant_folding_handle_ = handle;
  }
  lib_handles_.emplace_back(handle);
  return SUCCESS;
}

Status HostCpuEngine::GetRealPath(std::string &path) {
  std::string real_path = RealPath(path.c_str());
  if (real_path.empty()) {
    GELOGW("File path %s is invalid.", path.c_str());
    return INTERNAL_ERROR;
  }

  path = real_path;
  return SUCCESS;
}
} // namespace ge
