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

#include "single_op/single_op_model.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "runtime/rt.h"
#include "single_op/task/aicpu_task_builder.h"
#include "single_op/task/aicpu_kernel_task_builder.h"
#include "single_op/task/rts_kernel_task_builder.h"
#include "single_op/task/tbe_task_builder.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/node_executor/node_executor.h"

static std::atomic<std::uint64_t> aicpu_kernel_id(0);

using domi::TaskDef;
using std::unique_ptr;
using std::vector;

namespace ge {
namespace {
const size_t kDataOutputNum = 1;
const uint32_t kInputIndexOfData = 0;
const uint32_t kOutputIndexOfData = 0;
const size_t kNumTaskWithAtomicAddrCleanTask = 2;
const size_t kNumTaskWithMemCpyTask = 2;
constexpr char const *kAttrSupportDynamicShape = "support_dynamicshape";
const char *const kEngineNameAiCore = "AIcoreEngine";
const char *const kEngineNameAiCpu = "aicpu_ascend_kernel";
const char *const kEngineNameAiCpuTf = "aicpu_tf_kernel";

Status CheckHostMem(const std::vector<string> &dependencies, const NodePtr &node, bool &is_host_mem) {
  auto op_desc = node->GetOpDesc();
  for (const auto &input_name : dependencies) {
    int input_index = op_desc->GetInputIndexByName(input_name);
    if (input_index < 0) {
      GELOGE(INTERNAL_ERROR, "[Get][InputIndex]failed, node:[%s] inputname: %s.",
             node->GetName().c_str(), input_name.c_str());
      REPORT_CALL_ERROR("E19999", "GetInputIndexByName failed, node:[%s] inputname: %s.",
                        node->GetName().c_str(), input_name.c_str());
      return INTERNAL_ERROR;
    }

    const auto &src_node = NodeUtils::GetInDataNodeByIndex(*node, input_index);
    GE_CHECK_NOTNULL(src_node);
    auto src_op_desc = src_node->GetOpDesc();
    GE_CHECK_NOTNULL(src_op_desc);
    if (src_op_desc->GetType() == DATA) {
      auto tensor = src_op_desc->MutableInputDesc(kInputIndexOfData);
      if (AttrUtils::HasAttr(tensor, ATTR_NAME_VALUE)) {
        GELOGD("Get hostmem from node %s, inputname: %s.", src_node->GetName().c_str(), input_name.c_str());
        continue;
      }
    }
    is_host_mem = false;
    return SUCCESS;
  }
  is_host_mem = true;
  return SUCCESS;
}

Status CheckInferDepend(GeModelPtr &ge_model, bool &is_infer_depend, bool &is_host_mem) {
  auto comp_graph = GraphUtils::GetComputeGraph(ge_model->GetGraph());
  GE_CHECK_NOTNULL(comp_graph);
  for (const auto &node : comp_graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    const auto &depends = op_desc->GetOpInferDepends();
    bool support_dynamic_shape = false;
    (void)AttrUtils::GetBool(op_desc, kAttrSupportDynamicShape, support_dynamic_shape);
    if (!depends.empty() && support_dynamic_shape) {
      is_infer_depend = true;
      return CheckHostMem(depends, node, is_host_mem);
    }
  }
  return SUCCESS;
}
}  // namespace

SingleOpModel::SingleOpModel(const std::string &model_name, const void *model_data, uint32_t model_size)
    : model_name_(model_name), ori_model_data_(model_data), ori_model_size_(model_size) {}

Status SingleOpModel::Init() {
  GE_CHK_STATUS_RET_NOLOG(InitModel());
  return LoadAllNodes();
}

Status SingleOpModel::InitModel() {
  ge::ModelData model;
  model.model_len = ori_model_size_;
  model.model_data = const_cast<void *>(ori_model_data_);

  auto ret = model_helper_.LoadModel(model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] failed.");
    REPORT_CALL_ERROR("E19999", "InitModel fail for ModelHelper LoadModel failed.");
    return ret;
  }

  return SUCCESS;
}

void SingleOpModel::ParseOpModelParams(ModelHelper &model_helper, SingleOpModelParam &param) {
  int64_t value = 0;
  bool ret = false;
  std::shared_ptr<ge::GeModel> model = model_helper.GetGeModel();
  GE_CHECK_NOTNULL_JUST_RETURN(model);
  ret = ge::AttrUtils::GetInt(model, ATTR_MODEL_MEMORY_SIZE, value);
  param.memory_size = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(model, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, value);
  param.zero_copy_mem_size = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(model, ATTR_MODEL_WEIGHT_SIZE, value);
  param.weight_size = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(model, MODEL_ATTR_TASK_GEN_BASE_ADDR, value);
  param.base_addr = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, value);
  param.weight_addr = ret ? static_cast<uint64_t>(value) : 0;
  ret = ge::AttrUtils::GetInt(model, ATTR_MODEL_CORE_TYPE, value);
  param.core_type = ret ? value : 0;

  GELOGI("ParseOpModelParams(), total_memory_size:%lu, zero_copy_size:%lu, weight_size:%lu, core_type = %lu",
         param.memory_size, param.zero_copy_mem_size, param.weight_size, param.core_type);
}

Status SingleOpModel::InitModelMem(StreamResource &res) {
  ParseOpModelParams(model_helper_, model_params_);

  if (model_params_.memory_size > model_params_.zero_copy_mem_size) {
    const string purpose("malloc feature map memory on model execute.");
    GELOGI("total memory: %lu, zero_copy_mem: %lu", model_params_.memory_size, model_params_.zero_copy_mem_size);
    model_params_.mem_base =
        res.MallocMemory(purpose, model_params_.memory_size - model_params_.zero_copy_mem_size, false);
    if (model_params_.mem_base == nullptr) {
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }
  }

  if (model_params_.weight_size > 0 && has_weight_) {
    const string purpose("malloc weights memory on model execute.");
    model_params_.weight_base = res.MallocWeight(purpose, model_params_.weight_size);
    if (model_params_.weight_base == nullptr) {
      // no need to free memory, for that was handled by StreamResources
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }

    auto weight_buffer = model_helper_.GetGeModel()->GetWeight();
    GELOGI("To copy weight to device. weight size = %zu", weight_buffer.GetSize());
    GE_CHK_RT_RET(rtMemcpy(model_params_.weight_base,
                           model_params_.weight_size,
                           weight_buffer.GetData(),
                           weight_buffer.GetSize(),
                           RT_MEMCPY_HOST_TO_DEVICE));
  }

  return SUCCESS;
}

Status SingleOpModel::ParseInputNode(const OpDescPtr &op_desc) {
  vector<int64_t> offsets = op_desc->GetOutputOffset();
  if (offsets.size() != kDataOutputNum) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Parse][InputNode]Data op should have only one output, but got %zu, op_name:%s, op_type:%s.",
           op_desc->GetOutputOffset().size(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    REPORT_INNER_ERROR("E19999", "ParseInputNode fail for Data op should have only one output, but got %zu,"
                       "op_name:%s, op_type:%s.", op_desc->GetOutputOffset().size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  auto output_desc = op_desc->GetOutputDescPtr(0);
  GE_CHECK_NOTNULL(output_desc);
  int64_t tensor_size = 0;
  (void)TensorUtils::GetSize(*output_desc, tensor_size);
  input_offset_list_.emplace_back(offsets[0]);
  input_sizes_.emplace_back(tensor_size);
  GELOGI("[%s] parse input node: %s, size = %ld, offset = %u", model_name_.c_str(), op_desc->GetName().c_str(),
         tensor_size, static_cast<uint32_t>(offsets[0]));
  return SUCCESS;
}

void SingleOpModel::ParseOutputNode(const OpDescPtr &op_desc) {
  vector<int64_t> offsets = op_desc->GetInputOffset();
  for (uint32_t k = 0; k < static_cast<uint32_t>(offsets.size()); ++k) {
    auto input_desc = op_desc->GetInputDescPtr(k);
    if (input_desc == nullptr) {
      continue;
    }
    int64_t tensor_size = 0;
    (void)TensorUtils::GetSize(*input_desc, tensor_size);
    output_offset_list_.emplace_back(offsets[k]);
    output_sizes_.emplace_back(tensor_size);
    GELOGI("[%s] parse output node: %s, size = %ld, offset = %u", model_name_.c_str(), op_desc->GetName().c_str(),
           tensor_size, static_cast<uint32_t>(offsets[k]));
  }
}

Status SingleOpModel::LoadAllNodes() {
  auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);
  Graph graph = ge_model->GetGraph();
  model_id_ = ge_model->GetModelId();
  auto compute_graph = GraphUtils::GetComputeGraph(graph);
  if (compute_graph == nullptr) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][ComputeGraph] fail, model_name:%s.", model_name_.c_str());
    REPORT_CALL_ERROR("E19999", "LoadAllNodes fail for GetComputeGraph return nullptr, model_name:%s.",
        model_name_.c_str());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  auto nodes = compute_graph->GetDirectNode();
  size_t model_op_size = nodes.size();
  GELOGI("[%s] node size = %zu", model_name_.c_str(), model_op_size);

  for (size_t i = 0; i < model_op_size; ++i) {
    auto node = nodes.at(i);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    op_list_[op_desc->GetId()] = node;
    auto op_type = op_desc->GetType();
    GELOGI("[%s] node[%zu] = %s, type = %s", model_name_.c_str(), i, node->GetName().c_str(), op_type.c_str());

    if (op_type == DATA_TYPE || op_type == AIPP_DATA_TYPE) {
      data_ops_.emplace_back(op_desc);
      auto tensor = op_desc->MutableInputDesc(0);
      if (AttrUtils::HasAttr(tensor, ATTR_NAME_VALUE)) {
        int32_t index = 0;
        (void) AttrUtils::GetInt(op_desc, ATTR_NAME_INDEX, index);
        GELOGD("Node %s, index %d, has host mem.", node->GetName().c_str(), index);
        op_with_hostmem_[index] = node;
      }
      continue;
    }

    if (op_type == CONSTANT || op_type == CONSTANTOP) {
      has_weight_ = true;
      continue;
    }

    if (op_type == NETOUTPUT) {
      netoutput_op_ = op_desc;
      continue;
    }

    ge_model->GetTBEKernelStore().LoadTBEKernelBinToOpDesc(op_desc);
    ge_model->GetCustAICPUKernelStore().LoadCustAICPUKernelBinToOpDesc(op_desc);
  }

  return SUCCESS;
}

Status SingleOpModel::ParseInputsAndOutputs() {
  for (auto &op_desc : data_ops_) {
    GE_CHK_STATUS_RET_NOLOG(ParseInputNode(op_desc));
  }
  if (netoutput_op_ != nullptr) {
    ParseOutputNode(netoutput_op_);
  }
  return SUCCESS;
}

Status SingleOpModel::SetInputsAndOutputs(SingleOp &single_op) {
  int arg_index = 0;
  for (size_t i = 0; i < input_offset_list_.size(); ++i) {
    auto *addr = model_params_.mem_base + input_offset_list_[i];
    model_params_.addr_mapping_.emplace(reinterpret_cast<uintptr_t>(addr), arg_index++);
    single_op.input_sizes_.emplace_back(input_sizes_[i]);
    single_op.input_addr_list_.emplace_back(addr);
  }

  for (size_t i = 0; i < output_offset_list_.size(); ++i) {
    auto *addr = model_params_.mem_base + output_offset_list_[i];
    model_params_.addr_mapping_.emplace(reinterpret_cast<uintptr_t>(addr), arg_index++);
    single_op.output_sizes_.emplace_back(output_sizes_[i]);
    single_op.output_addr_list_.emplace_back(addr);
  }

  single_op.args_.resize(arg_index);
  return SUCCESS;
}

Status SingleOpModel::BuildTaskList(StreamResource *stream_resource, SingleOp &single_op) {
  auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);
  single_op.arg_table_.resize(single_op.input_sizes_.size() + single_op.output_sizes_.size());
  auto tasks = ge_model->GetModelTaskDefPtr()->task();
  for (int i = 0; i < tasks.size(); ++i) {
    const TaskDef &task_def = tasks[i];
    GELOGI("[%s] Task[%d], type = %u, DebugString = %s", model_name_.c_str(), i, task_def.type(),
           task_def.DebugString().c_str());
    auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
    if (task_type == RT_MODEL_TASK_KERNEL || task_type == RT_MODEL_TASK_ALL_KERNEL) {
      const auto &context = task_type == RT_MODEL_TASK_KERNEL ? task_def.kernel().context() :
                                                                task_def.kernel_with_handle().context();
      auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
      if (kernel_type == ccKernelType::TE) {
        GELOGD("Building TBE task");
        TbeOpTask *tbe_task = nullptr;
        auto ret = BuildKernelTask(task_def, &tbe_task);
        if (ret != SUCCESS) {
          return ret;
        }

        ParseArgTable(tbe_task, single_op);
        tbe_task->SetModelArgs(model_name_, model_id_);
        if (tbe_task->tiling_buffer_ != nullptr) {
          tbe_task->stream_resource_ = stream_resource;
        }
        single_op.tasks_.emplace_back(tbe_task);
      } else if (kernel_type == ccKernelType::AI_CPU || kernel_type == ccKernelType::CUST_AI_CPU) {
        GELOGD("Building AICPU_CC task");
        OpTask *task = nullptr;
        uint64_t singleop_kernel_id = aicpu_kernel_id++;
        GELOGI("Build singleOp CCTask, kernel_id = %lu", singleop_kernel_id);
        GE_CHK_STATUS_RET_NOLOG(BuildCpuKernelTask(task_def.kernel(), &task, singleop_kernel_id));
        task->SetModelArgs(model_name_, model_id_);
        ParseArgTable(task, single_op);
        single_op.tasks_.emplace_back(task);
      } else {
        GELOGE(ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID,
            "[Check][KernelType]Only TBE, AI_CPU, CUST_AI_CPU kernel are supported, but got %u", 
            context.kernel_type());
        REPORT_INNER_ERROR("E19999", 
            "BuildTaskList fail for %u not supported, Only TBE, AI_CPU, CUST_AI_CPU kernel are supported.",
            context.kernel_type());
        return ACL_ERROR_GE_OP_KERNEL_TYPE_INVALID;
      }
    } else if (task_type == RT_MODEL_TASK_KERNEL_EX) {
      GELOGD("Building AICPU_TF task");
      AiCpuTask *aicpu_task = nullptr;
      uint64_t singleop_kernel_id = aicpu_kernel_id++;
      GELOGI("Build singleOp TfTask, kernel_id = %lu", singleop_kernel_id);
      GE_CHK_STATUS_RET_NOLOG(
          BuildKernelExTask(task_def.kernel_ex(), &aicpu_task, singleop_kernel_id));
      aicpu_task->SetModelArgs(model_name_, model_id_);
      ParseArgTable(aicpu_task, single_op);
      single_op.tasks_.emplace_back(aicpu_task);
    } else if ((task_type == RT_MODEL_TASK_MEMCPY_ASYNC) || (task_type == RT_MODEL_TASK_MEMCPY_ADDR_ASYNC)) {
      auto kernel_def = task_def.memcpy_async();
      auto node = op_list_[kernel_def.op_index()];
      GE_CHECK_NOTNULL(node);
      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      std::unique_ptr<MemcpyAsyncTask> task;
      GE_CHK_STATUS_RET_NOLOG(RtsKernelTaskBuilder::BuildMemcpyAsyncTask(op_desc, kernel_def, model_params_, task));
      task->SetModelArgs(model_name_, model_id_);
      ParseArgTable(task.get(), single_op);
      single_op.tasks_.emplace_back(task.release());
    } else {
      // skip
      GELOGD("Skip task type: %d", static_cast<int>(task_type));
    }
  }
  return SUCCESS;
}

void SingleOpModel::ParseArgTable(OpTask *task, SingleOp &op) {
  if (task == nullptr) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Parse][ArgTable] fail for input OpTask is nullptr.");
    REPORT_INNER_ERROR("E19999", "ParseArgTable fail for input OpTask is nullptr.");
    return;
  }

  // args: addr1, addr2, addr3 ...
  uintptr_t *arg_base = nullptr;
  size_t arg_num = 0;
  task->GetIoAddr(arg_base, arg_num);
  for (size_t i = 0; i < arg_num; ++i) {
    uintptr_t *ptr_to_addr = arg_base + i;
    uintptr_t addr = *ptr_to_addr;
    auto iter = model_params_.addr_mapping_.find(addr);
    if (iter != model_params_.addr_mapping_.end()) {
      int arg_index = iter->second;
      GELOGI("%s args[%zu] mapped to user designated args[%d]", task->GetOpdesc()->GetName().c_str(), i, arg_index);
      op.arg_table_[iter->second].emplace_back(ptr_to_addr);
    }
  }
}
 
Status SingleOpModel::BuildKernelTask(const domi::TaskDef &task_def, TbeOpTask **task) {
  GE_CHECK_NOTNULL(task);
  auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
  const auto &context = task_type == RT_MODEL_TASK_KERNEL ? task_def.kernel().context() :
                                                            task_def.kernel_with_handle().context();
  auto iter = op_list_.find(context.op_index());
  if (iter == op_list_.end()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Param:TaskDef]op desc not found. op index = %u", context.op_index());
    REPORT_INNER_ERROR("E19999", "BuildKernelTask fail for op desc not found. op index = %u", context.op_index());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  std::unique_ptr<TbeOpTask> tbe_task(new (std::nothrow) TbeOpTask());
  if (tbe_task == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][TbeOpTask]failed.");
    REPORT_INNER_ERROR("E19999", "BuildKernelTask fail for new TbeOpTask.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  auto builder = TbeTaskBuilder(model_name_, iter->second, task_def);
  auto ret = builder.BuildTask(*tbe_task, model_params_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][TbeOpTask]failed.");
    REPORT_INNER_ERROR("E19999", "[Build][TbeOpTask]failed.");
    return ret;
  }

  *task = tbe_task.release();
  return SUCCESS;
}

Status SingleOpModel::BuildAtomicTask(const domi::TaskDef &task_def, AtomicAddrCleanOpTask **task) {
  GE_CHECK_NOTNULL(task);
  const auto &context = task_def.kernel().context();
  auto iter = op_list_.find(context.op_index());
  if (iter == op_list_.end()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][Param:TaskDef]op desc not found. op index = %u", context.op_index());
    REPORT_INNER_ERROR("E19999", "BuildKernelTask fail for op desc not found. op index = %u", context.op_index());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  std::unique_ptr<AtomicAddrCleanOpTask> atomic_task(new (std::nothrow) AtomicAddrCleanOpTask());
  if (atomic_task == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][AtomicAddrCleanOpTask]failed.");
    REPORT_INNER_ERROR("E19999", "BuildKernelTask fail for new AtomicAddrCleanOpTask.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  auto builder = AtomicAddrCleanTaskBuilder(model_name_, iter->second, task_def);
  auto ret = builder.BuildTask(*atomic_task, model_params_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][AtomicAddrCleanOpTask]failed.");
    REPORT_INNER_ERROR("E19999", "[Build][AtomicAddrCleanOpTask]failed.");
    return ret;
  }

  *task = atomic_task.release();
  return SUCCESS;
}

Status SingleOpModel::BuildKernelExTask(const domi::KernelExDef &kernel_def, AiCpuTask **task, uint64_t kernel_id) {
  auto iter = op_list_.find(kernel_def.op_index());
  if (iter == op_list_.end()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, 
        "[Check][Param:KernelExDef]op not found. op index = %u", kernel_def.op_index());
    REPORT_INNER_ERROR("E19999", 
        "BuildKernelExTask fail for param kernel_def, because op of kernel_def not found, op index:%u.", 
        kernel_def.op_index());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  std::unique_ptr<AiCpuTask> aicpu_task(new (std::nothrow) AiCpuTask());
  if (aicpu_task == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][AiCpuTask] failed.");
    REPORT_INNER_ERROR("E19999", "BuildKernelExTask fail for new AiCpuTask, model_name:%s.", model_name_.c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }
  auto builder = AiCpuTaskBuilder(iter->second->GetOpDesc(), kernel_def);
  auto ret = builder.BuildTask(*aicpu_task, model_params_, kernel_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][Task] failed, kernel_id:%lu.", kernel_id);
    return ret;
  }

  *task = aicpu_task.release();
  return SUCCESS;
}

Status SingleOpModel::BuildCpuKernelTask(const domi::KernelDef &kernel_def, OpTask **task, uint64_t kernel_id) {
  const auto &context = kernel_def.context();
  auto iter = op_list_.find(context.op_index());
  if (iter == op_list_.end()) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, 
        "[Check][Param:KernelDef] op desc not found. op index = %u", context.op_index());
    REPORT_INNER_ERROR("E19999", 
        "BuildCpuKernelTask fail for kernel_def is invalid, because op of kernel_def not found, op index:%u.", 
        context.op_index());
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  std::unique_ptr<AiCpuCCTask> aicpucc_task(new (std::nothrow) AiCpuCCTask());
  if (aicpucc_task == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][AiCpuCCTask] failed");
    REPORT_INNER_ERROR("E19999", "BuildCpuKernelTask fail for new AiCpuCCTask, model_name:%s.", model_name_.c_str());
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  auto builder = AiCpuCCTaskBuilder(iter->second->GetOpDesc(), kernel_def);
  auto ret = builder.BuildTask(*aicpucc_task, kernel_id, model_params_);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Build][AiCpuCCTask]failed, kernel_id:%lu.", kernel_id);
    REPORT_CALL_ERROR("E19999", "BuildCpuKernelTask fail for build AiCpuTask, kernel_id:%lu.", kernel_id);
    return ret;
  }

  *task = aicpucc_task.release();
  return SUCCESS;
}
Status SingleOpModel::InitHybridModelExecutor(const StreamResource &resource, const GeModelPtr &ge_model,
                                              SingleOp &single_op) {
  for (const auto &op_desc : data_ops_) {
    auto output_tensor_desc = op_desc->GetOutputDesc(kOutputIndexOfData);
    GeTensorDesc tensor_desc(output_tensor_desc);
    single_op.inputs_desc_.emplace_back(tensor_desc);
    GELOGD("Init inputs desc from %s.", op_desc->GetName().c_str());
  }
  GE_CHK_STATUS_RET_NOLOG(hybrid::NodeExecutorManager::GetInstance().EnsureInitialized());
  auto root_model = model_helper_.GetGeRootModel();
  GE_CHECK_NOTNULL(root_model);
  root_model->SetRootGraph(GraphUtils::GetComputeGraph(ge_model->GetGraph()));
  root_model->SetSubgraphInstanceNameToModel(root_model->GetRootGraph()->GetName(), ge_model);
  single_op.hybrid_model_.reset(new (std::nothrow)hybrid::HybridModel(root_model));
  GE_CHECK_NOTNULL(single_op.hybrid_model_);
  GE_CHK_STATUS_RET(single_op.hybrid_model_->Init(true), "[Init][HybridModel]Failed.");
  int32_t device_id = 0;
  GE_CHK_RT_RET(rtGetDevice(&device_id));
  single_op.hybrid_model_executor_.reset(new (std::nothrow)hybrid::HybridModelExecutor(single_op.hybrid_model_.get(),
                                                                                       device_id,
                                                                                       resource.GetStream()));
  GE_CHECK_NOTNULL(single_op.hybrid_model_executor_);
  GE_CHK_STATUS_RET(single_op.hybrid_model_executor_->Init(), "[Init][HybridModelExecutor]Failed.");
  return SUCCESS;
}

Status SingleOpModel::BuildOp(StreamResource &resource, SingleOp &single_op) {
  GE_CHK_STATUS_RET_NOLOG(ParseInputsAndOutputs());
  GE_CHK_STATUS_RET_NOLOG(InitModelMem(resource));
  single_op.running_param_.reset(new (std::nothrow)SingleOpModelParam(model_params_));
  GE_CHECK_NOTNULL(single_op.running_param_);
  GE_CHK_STATUS_RET_NOLOG(SetInputsAndOutputs(single_op));
  auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);
  bool infer_depend_flag = false;
  bool is_host_mem = false;
  GE_CHK_STATUS_RET(CheckInferDepend(ge_model, infer_depend_flag, is_host_mem), "[Check][InferDepend] failed.");
  if (infer_depend_flag) {
    // construct single_op, do single op with HybridModelExecutor
    GELOGD("Init hybrid model params of single op, and will do execute with hybrid model executor.");
    return InitHybridModelExecutor(resource, ge_model, single_op);
  }
  return BuildTaskList(&resource, single_op);
}

Status SingleOpModel::BuildTaskListForDynamicOp(StreamResource *stream_resource, DynamicSingleOp &single_op) {
  auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);

  auto compute_graph = GraphUtils::GetComputeGraph(ge_model->GetGraph());
  GE_CHECK_NOTNULL(compute_graph);
  single_op.compute_graph_ = compute_graph;

  if (node_tasks_.size() != 1) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]Node size must be 1, but get %zu.", node_tasks_.size());
    REPORT_INNER_ERROR("E19999", "[Check][Size]Node size must be 1, but get %zu.", node_tasks_.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  auto iter = node_tasks_.begin();
  auto node = iter->first;
  const auto &task_defs = iter->second;
  if (task_defs.size() <= 0 || task_defs.size() > kNumTaskWithAtomicAddrCleanTask) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]Node size must be 1, but get %zu.", node_tasks_.size());
    REPORT_INNER_ERROR("E19999", "[Check][Size]task_defs size must be 1 or 2, but get %zu.", task_defs.size());
    return ACL_ERROR_GE_PARAM_INVALID;
  }
      
  GE_CHECK_NOTNULL(node);
  auto op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  const auto &lib_name = op_desc->GetOpKernelLibName();
  if (lib_name == kEngineNameAiCore) {
    GELOGD("Building TBE task.");
    const auto &task_def = task_defs.back();
    TbeOpTask *tbe_task = nullptr;
    GE_CHK_STATUS_RET_NOLOG(BuildKernelTask(task_def, &tbe_task));
    tbe_task->SetModelArgs(model_name_, model_id_);
    if (tbe_task->tiling_buffer_ != nullptr) {
      GELOGD("tiling buffer is not nullptr.");
      tbe_task->stream_resource_ = stream_resource;
    }
    if (task_defs.size() == kNumTaskWithAtomicAddrCleanTask) {
      const auto &atomic_task_def = task_defs.front();
      AtomicAddrCleanOpTask *atomic_task = nullptr;
      GE_CHK_STATUS_RET_NOLOG(BuildAtomicTask(atomic_task_def, &atomic_task));
      GE_CHK_STATUS_RET_NOLOG(atomic_task->InitAtomicAddrCleanIndices());
      tbe_task->SetAtomicAddrCleanTask(atomic_task);
    }
    single_op.op_task_.reset(tbe_task);
  } else if (lib_name == kEngineNameAiCpu) {
    const auto &task_def = task_defs[0];
    GELOGD("Building AICPU_CC task");
    OpTask *task = nullptr;
    uint64_t dynamic_singleop_kernel_id = aicpu_kernel_id++;
    GELOGI("Build dynamic singleOp CCTask, kernel_id = %lu", dynamic_singleop_kernel_id);
    GE_CHK_STATUS_RET_NOLOG(BuildCpuKernelTask(task_def.kernel(), &task, dynamic_singleop_kernel_id));
    task->SetModelArgs(model_name_, model_id_);
    single_op.op_task_.reset(task);
  } else if (lib_name == kEngineNameAiCpuTf) {
    const auto &task_def = task_defs[0];
    GELOGD("Building AICPU_TF task");
    AiCpuTask *aicpu_task = nullptr;
    uint64_t dynamic_singleop_kernel_id = aicpu_kernel_id++;
    GELOGI("Build dynamic singleOp TfTask, kernel_id = %lu", dynamic_singleop_kernel_id);
    GE_CHK_STATUS_RET_NOLOG(BuildKernelExTask(task_def.kernel_ex(), &aicpu_task, dynamic_singleop_kernel_id));
    if (aicpu_task->GetUnknownType() == DEPEND_COMPUTE) {
      if (task_defs.size() < kNumTaskWithMemCpyTask) {
        GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Task]The copy task of the fourth operator was not found.");
        REPORT_INNER_ERROR("E19999", "The copy task of the fourth operator was not found.");
        return ACL_ERROR_GE_PARAM_INVALID;
      }
      const TaskDef &copy_task_def = task_defs[1];
      GE_CHK_STATUS_RET_NOLOG(aicpu_task->SetMemCopyTask(copy_task_def.kernel_ex()));
    }
    aicpu_task->SetModelArgs(model_name_, model_id_);
    single_op.op_task_.reset(aicpu_task);
  }

  return SUCCESS;
}

Status SingleOpModel::NeedHybridModel(GeModelPtr &ge_model, bool &need_hybrid_model) {
  bool is_infer_depend = false;
  bool is_host_mem = false;
  GE_CHK_STATUS_RET(CheckInferDepend(ge_model, is_infer_depend, is_host_mem), "[Check][InferDepend] failed.");
  bool need_d2h_cpy = is_infer_depend && !is_host_mem;
  need_hybrid_model = need_d2h_cpy || node_tasks_.size() > 1;
  return SUCCESS;
}

Status SingleOpModel::ParseTasks() {
  auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);

  auto tasks = ge_model->GetModelTaskDefPtr()->task();
  for (int i = 0; i < tasks.size(); ++i) {
    TaskDef &task_def = tasks[i];
    GELOGI("[%s] Task[%d], type = [%u], DebugString = [%s]", model_name_.c_str(), i, task_def.type(),
           task_def.DebugString().c_str());
    auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
    uint32_t op_index = 0;
    if (task_type == RT_MODEL_TASK_KERNEL) {
      op_index = task_def.kernel().context().op_index();
    } else if (task_type == RT_MODEL_TASK_KERNEL_EX) {
      op_index = task_def.kernel_ex().op_index();
    } else if (task_type == RT_MODEL_TASK_ALL_KERNEL) {
      op_index = task_def.kernel_with_handle().context().op_index();
    } else {
      GELOGD("Skip task type: %d", static_cast<int>(task_type));
      continue;
    }
    GELOGD("op_index = %u, task_type = %d", op_index, task_type);

    auto iter = op_list_.find(op_index);
    if (iter == op_list_.end()) {
      GELOGE(INTERNAL_ERROR, "[Find][Node]Failed to get node by op_index = %u", op_index);
      REPORT_INNER_ERROR("E19999", "Failed to get node by op_index = %u.", op_index);
      return INTERNAL_ERROR;
    }
    auto &node = iter->second;
    node_tasks_[node].emplace_back(task_def);
  }
  return SUCCESS;
}

Status SingleOpModel::BuildDynamicOp(StreamResource &resource, DynamicSingleOp &single_op) {
  single_op.num_inputs_ = data_ops_.size();
  single_op.num_outputs_ = netoutput_op_->GetAllInputsSize();
  GE_CHK_STATUS_RET_NOLOG(InitModelMem(resource));
  model_params_.memory_size = UINT64_MAX;
  model_params_.graph_is_dynamic = true;
  GE_CHK_STATUS_RET(ParseTasks(), "[Parse][Tasks] failed.");

  auto ge_model = model_helper_.GetGeModel();
  GE_CHECK_NOTNULL(ge_model);
  bool need_hybrid_model = false;
  GE_CHK_STATUS_RET(NeedHybridModel(ge_model, need_hybrid_model), "[Check][NeedHybridModel] failed.");
  if (need_hybrid_model) {
    GELOGD("Build single op HybridModel.");
    GE_CHK_STATUS_RET_NOLOG(hybrid::NodeExecutorManager::GetInstance().EnsureInitialized());
    GE_CHK_STATUS(SetHostMemTensor(single_op), "[Init][HostMem]Failed.");
    auto root_model = model_helper_.GetGeRootModel();
    GE_CHECK_NOTNULL(root_model);
    root_model->SetRootGraph(GraphUtils::GetComputeGraph(ge_model->GetGraph()));
    root_model->SetSubgraphInstanceNameToModel(root_model->GetRootGraph()->GetName(), ge_model);
    single_op.hybrid_model_.reset(new (std::nothrow)hybrid::HybridModel(root_model));
    GE_CHECK_NOTNULL(single_op.hybrid_model_);
    GE_CHK_STATUS_RET(single_op.hybrid_model_->Init(true), "[Init][HybridModel]Failed.");
    int32_t device_id = 0;
    GE_CHK_RT_RET(rtGetDevice(&device_id));
    single_op.hybrid_model_executor_.reset(new (std::nothrow)hybrid::HybridModelExecutor(single_op.hybrid_model_.get(),
                                                                                         device_id,
                                                                                         resource.GetStream()));
    GE_CHECK_NOTNULL(single_op.hybrid_model_executor_);
    ThreadPool *thread_pool = nullptr;
    GE_CHK_STATUS_RET_NOLOG(resource.GetThreadPool(&thread_pool));
    GE_CHK_STATUS_RET(single_op.hybrid_model_executor_->Init(thread_pool), "[Init][HybridModelExecutor]Failed.");
    return SUCCESS;
  }
  return BuildTaskListForDynamicOp(&resource, single_op);
}

Status SingleOpModel::SetHostMemTensor(DynamicSingleOp &single_op) {
  for (auto &node_map : op_with_hostmem_) {
    auto node = node_map.second;
    auto out_anchor = node->GetOutDataAnchor(0);
    GE_CHECK_NOTNULL(out_anchor);
    auto in_anchors = out_anchor->GetPeerInDataAnchors();
    vector<GeTensorDescPtr> tensor_descs;
    auto idx = node_map.first;
    for (auto anchor : in_anchors) {
      GE_CHECK_NOTNULL(anchor);
      auto output_node = anchor->GetOwnerNode();
      GE_CHECK_NOTNULL(output_node);
      auto op_desc = output_node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      auto tensor_desc = op_desc->MutableInputDesc(anchor->GetIdx());
      tensor_descs.emplace_back(tensor_desc);
      GELOGD("Get %d th input tensor desc of %s by %d data node: %s.", anchor->GetIdx(),
             output_node->GetName().c_str(), idx, node->GetName().c_str());
    }
    single_op.tensor_with_hostmem_[idx] = tensor_descs;
  }
  return SUCCESS;
}
}  // namespace ge
