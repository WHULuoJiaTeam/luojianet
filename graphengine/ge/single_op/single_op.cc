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

#include "single_op/single_op.h"

#include "framework/common/fmk_types.h"
#include "framework/common/ge_types.h"
#include "common/math/math_util.h"
#include "common/profiling/profiling_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "graph/load/model_manager/model_utils.h"
#include "runtime/mem.h"
#include "single_op/single_op_manager.h"
#include "single_op/task/build_task_utils.h"
#include "graph/load/model_manager/model_manager.h"

namespace ge {
namespace {
const size_t kDataMemAlignSize = 32;
const size_t kDataMemAlignUnit = 2;
const string kShapeTypeDynamic = "dynamic";
const string kShapeTypeStatic = "static";
const int64_t kHostMemType = 1;
const uint32_t kFuzzDeviceBufferSize = 1 * 1024 * 1024;
const uint32_t kAlignBytes = 512;

size_t GetAlignedSize(size_t size) {
  size_t aligned_size = (size + kDataMemAlignUnit * kDataMemAlignSize - 1) / kDataMemAlignSize * kDataMemAlignSize;
  return aligned_size;
}

Status ProfilingTaskInfo(OpTask *op_task, const string &shape_type) {
  if (!ProfilingManager::Instance().ProfilingModelLoadOn()) {
    return SUCCESS;
  }

  TaskDescInfo tmp_task_desc_info;
  uint32_t model_id;
  if (op_task->GetProfilingArgs(tmp_task_desc_info, model_id) != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Get][ProfilingArgs] failed.");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  GELOGD("ProfilingReport of op[%s] model[%s] start.",
         tmp_task_desc_info.op_name.c_str(), tmp_task_desc_info.model_name.c_str());

  tmp_task_desc_info.shape_type = shape_type;
  tmp_task_desc_info.cur_iter_num = ProfilingManager::Instance().GetStepInfoIndex();
  tmp_task_desc_info.task_type = op_task->GetTaskType();

  std::vector<TaskDescInfo> task_desc_info;
  task_desc_info.emplace_back(tmp_task_desc_info);

  auto &profiling_manager = ProfilingManager::Instance();
  profiling_manager.ReportProfilingData(model_id, task_desc_info);
  return SUCCESS;
}

Status CalInputsHostMemSize(const std::vector<DataBuffer> &inputs,
                            std::vector<std::pair<size_t, uint64_t>> &inputs_size) {
  int64_t total_size = 0;
  size_t index = 0;
  for (auto &input_buffer : inputs) {
    int64_t input_size = 0;
    if (input_buffer.placement == kHostMemType) {
      GE_CHECK_LE(input_buffer.length, INT64_MAX);
      input_size = input_buffer.length;
      // input_size pad to 512
      GE_CHK_STATUS_RET(CheckInt64AddOverflow(input_size, (kAlignBytes - 1)), "Padding size is beyond the INT64_MAX.");
      input_size = ((input_size + kAlignBytes - 1) / kAlignBytes) * kAlignBytes;
      inputs_size.emplace_back(index, input_size);
      GE_CHK_STATUS_RET(CheckInt64AddOverflow(total_size, input_size), "Total size is beyond the INT64_MAX.");
      total_size += input_size;
      GELOGD("The %zu input mem type is host, the tensor size is %ld.", index, input_size);
    }
    index++;
  }
  if (total_size > kFuzzDeviceBufferSize) {
    GELOGE(FAILED, "[Check][Size]Total size is %ld, larger than 1M.", total_size);
    return FAILED;
  }
  return SUCCESS;
}

Status UpdateInputsBufferAddr(StreamResource *stream_resource, rtStream_t stream,
                              const std::vector<std::pair<size_t, uint64_t>> &inputs_size,
                              std::vector<DataBuffer> &update_buffers) {
  GE_CHECK_NOTNULL(stream_resource);
  auto dst_addr = reinterpret_cast<uint8_t *>(stream_resource->GetDeviceBufferAddr());
  // copy host mem from input_buffer to device mem of dst_addr
  for (const auto &input_size : inputs_size) {
    auto index = input_size.first;
    auto size = input_size.second;
    GELOGD("Do h2d for %zu input, dst size is %zu, src length is %lu.", index, size, update_buffers[index].length);
    GE_CHK_RT_RET(rtMemcpyAsync(dst_addr, size, update_buffers[index].data, update_buffers[index].length,
                                RT_MEMCPY_HOST_TO_DEVICE_EX, stream));
    update_buffers[index].data = dst_addr;
    dst_addr = dst_addr + size;
  }
  return SUCCESS;
}

Status ModifyTensorDesc(GeTensorDesc &tensor) {
  int64_t storage_format_val = static_cast<Format>(FORMAT_RESERVED);
  (void)AttrUtils::GetInt(tensor, ge::ATTR_NAME_STORAGE_FORMAT, storage_format_val);
  auto storage_format = static_cast<Format>(storage_format_val);
  auto format = tensor.GetFormat();
  if (storage_format != FORMAT_RESERVED && storage_format != format) {
    std::vector<int64_t> storage_shape;
    if (!AttrUtils::GetListInt(tensor, ge::ATTR_NAME_STORAGE_SHAPE, storage_shape)) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][storage_shape]failed while storage_format was set.");
      REPORT_INNER_ERROR("E19999", "Get storage_shape failed while storage_format was set.");
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }

    GELOGD("Storage format set. update shape to [%s], and original shape to [%s]",
           GeShape(storage_shape).ToString().c_str(), tensor.GetShape().ToString().c_str());
    tensor.SetOriginShape(tensor.GetShape());
    tensor.SetOriginFormat(format);
    tensor.SetShape(GeShape(storage_shape));
    tensor.SetFormat(storage_format);
  }

  return SUCCESS;
}

Status InitHybridModelArgs(const std::vector<DataBuffer> &input_buffers,
                           const std::vector<DataBuffer> &output_buffers,
                           const std::vector<GeTensorDesc> &inputs_desc,
                           hybrid::HybridModelExecutor::ExecuteArgs &args) {
  for (auto &input : input_buffers) {
    args.inputs.emplace_back(hybrid::TensorValue(input.data, input.length));
  }
  for (auto &output : output_buffers) {
    args.outputs.emplace_back(hybrid::TensorValue(output.data, output.length));
  }
  for (auto &tensor_desc : inputs_desc) {
    auto desc = MakeShared<GeTensorDesc>(tensor_desc);
    GE_CHECK_NOTNULL(desc);
    GE_CHK_STATUS_RET_NOLOG(ModifyTensorDesc(*desc));
    args.input_desc.emplace_back(desc);
  }
  return SUCCESS;
}
}  // namespace

SingleOp::SingleOp(StreamResource *stream_resource, std::mutex *stream_mutex, rtStream_t stream)
    : stream_resource_(stream_resource), stream_mutex_(stream_mutex), stream_(stream) {
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY SingleOp::~SingleOp() {
  for (auto task : tasks_) {
    delete task;
    task = nullptr;
  }
}

Status SingleOp::ValidateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  auto num_inputs = inputs.size();
  if (num_inputs != input_sizes_.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, 
        "[Check][Param:inputs]Input num mismatch. model expect %zu, but given %zu", input_addr_list_.size(),
           inputs.size());
    REPORT_INPUT_ERROR("E10401", std::vector<std::string>({"expect_num", "input_num"}), 
        std::vector<std::string>({std::to_string(input_addr_list_.size()), std::to_string(num_inputs)}));
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    // preventing from read out of bound
    size_t aligned_size = GetAlignedSize(inputs[i].length);
    GELOGI("Input [%zu], aligned_size:%zu, inputs.length:%lu, input_sizes_:%zu",
           i, aligned_size, inputs[i].length, input_sizes_[i]);
    if (aligned_size < input_sizes_[i]) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, 
          "[Check][Param:inputs]Input size mismatch. index = %zu, model expect %zu, but given %zu(after align)", 
          i, input_sizes_[i], aligned_size);
      REPORT_INPUT_ERROR("E10402", std::vector<std::string>({"index", "expect_size", "input_size"}), 
          std::vector<std::string>({std::to_string(i), std::to_string(input_sizes_[i]), std::to_string(aligned_size)})
          );
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  }

  auto num_outputs = outputs.size();
  if (num_outputs != output_sizes_.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param:outputs]output num mismatch. model expect %zu, but given %zu",
        output_sizes_.size(), outputs.size());
    REPORT_INPUT_ERROR("E10403", std::vector<std::string>({"expect_num", "input_num"}), 
        std::vector<std::string>({std::to_string(output_sizes_.size()), std::to_string(outputs.size())}));
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    // preventing from write out of bound
    size_t aligned_size = GetAlignedSize(outputs[i].length);
    GELOGI("Output [%zu], aligned_size:%zu, outputs.length:%lu, output_sizes_:%zu",
           i, aligned_size, outputs[i].length, output_sizes_[i]);
    if (aligned_size < output_sizes_[i]) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, 
          "[Check][Param:outputs]Output size mismatch. index = %zu, model expect %zu, but given %zu(after align)",
          i, output_sizes_[i], aligned_size);
      REPORT_INPUT_ERROR("E10404", std::vector<std::string>({"index", "expect_size", "input_size"}),
          std::vector<std::string>({std::to_string(i), std::to_string(output_sizes_[i]), std::to_string(aligned_size)})
          );
      return ACL_ERROR_GE_PARAM_INVALID;
    }
  }

  return SUCCESS;
}

Status SingleOp::GetArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  size_t arg_index = 0;
  for (auto &input : inputs) {
    args_[arg_index++] = reinterpret_cast<uintptr_t>(input.data);
  }

  for (auto &output : outputs) {
    args_[arg_index++] = reinterpret_cast<uintptr_t>(output.data);
  }
  return SUCCESS;
}

Status SingleOp::UpdateArgs(const std::vector<DataBuffer> &inputs, const std::vector<DataBuffer> &outputs) {
  Status ret = GetArgs(inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }
  // update tbe task args
  size_t num_args = arg_table_.size();
  for (size_t i = 0; i < num_args; ++i) {
    std::vector<uintptr_t *> &ptr_to_arg_in_tasks = arg_table_[i];
    if (ptr_to_arg_in_tasks.empty()) {
      GELOGW("found NO arg address to update for arg[%lu]", i);
      continue;
    }

    for (uintptr_t *arg_addr : ptr_to_arg_in_tasks) {
      *arg_addr = args_[i];
    }
  }
  return SUCCESS;
}

FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY Status SingleOp::ExecuteAsync(const std::vector<DataBuffer> &inputs,
                                                                               const std::vector<DataBuffer> &outputs) {
  GELOGD("Start SingleOp::ExecuteAsync.");
  Status ret = ValidateArgs(inputs, outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  GE_CHECK_NOTNULL(stream_resource_);
  vector<pair<size_t, uint64_t>> inputs_size;
  GE_CHK_STATUS_RET_NOLOG(CalInputsHostMemSize(inputs, inputs_size));
  std::lock_guard<std::mutex> lk(*stream_mutex_);
  vector<DataBuffer> update_buffers = inputs;
  if (!inputs_size.empty()) {
    GE_CHK_STATUS_RET_NOLOG(UpdateInputsBufferAddr(stream_resource_, stream_, inputs_size, update_buffers));
  }

  if (hybrid_model_executor_ != nullptr) {
    GELOGD("Execute multi-task single op by hybrid model executor");
    hybrid::HybridModelExecutor::ExecuteArgs args;
    GE_CHK_STATUS_RET_NOLOG(InitHybridModelArgs(update_buffers, outputs, inputs_desc_, args));
    return hybrid_model_executor_->Execute(args);
  }

  auto current_mem_base = stream_resource_->GetMemoryBase();
  if (running_param_->mem_base != current_mem_base) {
    running_param_->mem_base = const_cast<uint8_t *>(current_mem_base);
    GELOGD("Memory base changed, new memory base = %p", current_mem_base);
    for (auto &task : tasks_) {
      auto new_address = BuildTaskUtils::GetAddresses(task->GetOpdesc(), *running_param_);
      GE_CHK_STATUS_RET(task->UpdateArgTable(*running_param_), "[Update][ArgTable] failed, single op:%s.",
          task->GetOpdesc()->GetName().c_str());
    }
  }
  ret = UpdateArgs(update_buffers, outputs);
  if (ret != SUCCESS) {
    return ret;
  }

  for (auto &task : tasks_) {
    ret = task->LaunchKernel(stream_);
    GELOGD("[DEBUG_TASK_INFO : Static Task] %s %s",
           task->GetTaskName().c_str(),
           BuildTaskUtils::GetTaskInfo(task->GetOpdesc(), inputs, outputs).c_str());
    if (ret != SUCCESS) {
      return ret;
    }
    GE_CHK_STATUS_RET(task->OpenDump(stream_), "[Open][Dump]failed, single op:%s.", 
        task->GetOpdesc()->GetName().c_str());
    GE_CHK_STATUS_RET_NOLOG(ProfilingTaskInfo(task, kShapeTypeStatic));
  }

  return ret;
}

void SingleOp::SetStream(rtStream_t stream) {
  stream_ = stream;
}

DynamicSingleOp::DynamicSingleOp(uintptr_t resource_id, std::mutex *stream_mutex, rtStream_t stream)
    : resource_id_(resource_id), stream_mutex_(stream_mutex), stream_(stream) {
}

Status DynamicSingleOp::ValidateParams(const vector<GeTensorDesc> &input_desc,
                                       const std::vector<DataBuffer> &inputs,
                                       std::vector<GeTensorDesc> &output_desc,
                                       std::vector<DataBuffer> &outputs) const {
  if (inputs.size() != input_desc.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
        "[Check][Param:inputs]Input number mismatches input desc number. Input num = %zu, input desc num = %zu",
        inputs.size(), input_desc.size());
    REPORT_INPUT_ERROR("E10405", std::vector<std::string>({"input_num", "input_desc_num"}),
        std::vector<std::string>({std::to_string(inputs.size()), std::to_string(input_desc.size())}));
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (outputs.size() != output_desc.size()) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
        "[Check][Param:outputs]Output number mismatches output desc number. Output num = %zu, output desc num = %zu",
        outputs.size(), output_desc.size());
    REPORT_INPUT_ERROR("E10406", std::vector<std::string>({"out_num", "out_desc_num"}),
        std::vector<std::string>({std::to_string(outputs.size()), std::to_string(output_desc.size())}));
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (input_desc.size() != num_inputs_) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param:input_desc]Input number mismatches. expect %zu, but given %zu",
        num_inputs_, input_desc.size());
    REPORT_INPUT_ERROR("E10401", std::vector<std::string>({"expect_num", "input_num"}),
        std::vector<std::string>({std::to_string(num_inputs_), std::to_string(input_desc.size())}));
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (output_desc.size() != num_outputs_) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param:output_desc]Output number mismatches. expect %zu, but given %zu",
        num_outputs_, output_desc.size());
    REPORT_INPUT_ERROR("E10403", std::vector<std::string>({"expect_num", "input_num"}),
        std::vector<std::string>({std::to_string(num_outputs_), std::to_string(output_desc.size())}));
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  return SUCCESS;
}

Status DynamicSingleOp::SetHostTensorValue(const std::vector<std::pair<size_t, uint64_t>> &inputs_size,
                                           const vector<GeTensorDesc> &input_desc,
                                           const std::vector<DataBuffer> &input_buffers) {
  auto op_desc = op_task_->GetOpdesc();
  GE_CHECK_NOTNULL(op_desc);
  GELOGD("Start update inputs tensor value of %s.", op_desc->GetName().c_str());
  for (const auto &input_size : inputs_size) {
    size_t index = input_size.first;
    auto ge_tensor_desc = input_desc.at(index);
    // reconstruct GeTensor by DataBuffer
    GeTensorPtr ge_tensor = MakeShared<GeTensor>(ge_tensor_desc);
    GE_CHECK_NOTNULL(ge_tensor);
    GELOGD("The %zu tensor input type is host, desc data type is %d, input buffer addr is %p, size is %ld.",
           index, ge_tensor_desc.GetDataType(), input_buffers[index].data, input_buffers[index].length);
    if (ge_tensor->SetData(reinterpret_cast<uint8_t *>(input_buffers[index].data),
                           static_cast<size_t>(input_buffers[index].length)) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Set][Data]Failed to set data of ge tensor.");
      return INTERNAL_ERROR;
    }
    auto tensor_desc = op_desc->MutableInputDesc(index);
    GE_CHECK_NOTNULL(tensor_desc);
    if (!AttrUtils::SetTensor(tensor_desc, ATTR_NAME_VALUE, ge_tensor)) {
      GELOGE(FAILED, "[Set][ATTR_NAME_VALUE]Failed to set ATTR_NAME_VALUE to %s.", op_desc->GetName().c_str());
      return FAILED;
    }
  }
  return SUCCESS;
}

Status DynamicSingleOp::SetHostTensorValue(const vector<GeTensorDesc> &input_desc,
                                           const vector<DataBuffer> &input_buffers) {
  for (auto &tensor_map : tensor_with_hostmem_) {
    auto index = static_cast<size_t>(tensor_map.first);
    if (index >= input_desc.size() || index >= input_buffers.size()) {
      GELOGE(INTERNAL_ERROR, "[Check][Size]Index %zu should smaller then input desc size %zu "
             "and input buffers size %zu.", index, input_desc.size(), input_buffers.size());
      return INTERNAL_ERROR;
    }
    auto ge_tensor_desc = input_desc[index];
    // reconstruct GeTensor by DataBuffer
    GeTensorPtr ge_tensor = MakeShared<GeTensor>(ge_tensor_desc);
    GE_CHECK_NOTNULL(ge_tensor);
    GELOGD("The %zu tensor input type is host, desc data type is %d, input buffer addr is %p, size is %ld.",
           index, ge_tensor_desc.GetDataType(), input_buffers[index].data, input_buffers[index].length);
    if (ge_tensor->SetData(reinterpret_cast<uint8_t *>(input_buffers[index].data),
                           static_cast<size_t>(input_buffers[index].length)) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Set][Data]Failed to set data of ge tensor.");
      return INTERNAL_ERROR;
    }
    for (auto &tensor_desc : tensor_map.second) {
      GE_CHECK_NOTNULL(tensor_desc);
      if (!AttrUtils::SetTensor(tensor_desc, ATTR_NAME_VALUE, ge_tensor)) {
        GELOGE(FAILED, "[Set][ATTR_NAME_VALUE]Failed to set ATTR_NAME_VALUE.");
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status DynamicSingleOp::ExecuteAsync(const vector<GeTensorDesc> &input_desc,
                                     const vector<DataBuffer> &input_buffers,
                                     vector<GeTensorDesc> &output_desc,
                                     vector<DataBuffer> &output_buffers) {
  GELOGD("Start DynamicSingleOp::ExecuteAsync.");
  GE_CHK_STATUS_RET_NOLOG(ValidateParams(input_desc, input_buffers, output_desc, output_buffers));
  vector<pair<size_t, uint64_t>> inputs_size;
  GE_CHK_STATUS_RET_NOLOG(CalInputsHostMemSize(input_buffers, inputs_size));
  vector<DataBuffer> update_buffers = input_buffers;
  std::lock_guard<std::mutex> lk(*stream_mutex_);
  if (!inputs_size.empty()) {
    StreamResource *stream_resource  = SingleOpManager::GetInstance().GetResource(resource_id_, stream_);
    GE_CHK_STATUS_RET_NOLOG(UpdateInputsBufferAddr(stream_resource, stream_, inputs_size, update_buffers));
    GE_CHK_STATUS_RET_NOLOG(SetHostTensorValue(input_desc, input_buffers));
  }

  if (hybrid_model_executor_ != nullptr) {
    GELOGD("Execute multi-task dynamic single op by hybrid model executor");
    hybrid::HybridModelExecutor::ExecuteArgs args;
    GE_CHK_STATUS_RET_NOLOG(InitHybridModelArgs(update_buffers, output_buffers, input_desc, args));

    return hybrid_model_executor_->Execute(args);
  }
  GE_CHECK_NOTNULL(op_task_);
  if (!inputs_size.empty()) {
    GE_CHK_STATUS_RET_NOLOG(SetHostTensorValue(inputs_size, input_desc, input_buffers));
    GE_CHK_STATUS_RET_NOLOG(op_task_->LaunchKernel(input_desc, update_buffers, output_desc, output_buffers, stream_));
  } else {
    GE_CHK_STATUS_RET_NOLOG(op_task_->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream_));
  }
  GELOGD("[DEBUG_TASK_INFO : Dynamic Task] %s",
         BuildTaskUtils::GetTaskInfo(op_task_->GetOpdesc(), input_buffers, output_buffers).c_str());
  GE_CHK_STATUS_RET_NOLOG(op_task_->OpenDump(stream_));
  GE_CHK_STATUS_RET_NOLOG(ProfilingTaskInfo(op_task_.get(), kShapeTypeDynamic));
  return SUCCESS;
}
}  // namespace ge
