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

#include "common/dump/dump_op.h"

#include "common/dump/dump_manager.h"
#include "common/ge/datatype_util.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "graph/anchor.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "proto/ge_ir.pb.h"
#include "proto/op_mapping.pb.h"
#include "runtime/mem.h"
#include "aicpu/common/aicpu_task_struct.h"

namespace {
const uint32_t kAicpuLoadFlag = 1;
const char *const kDumpOutput = "output";
const char *const kDumpInput = "input";
const char *const kDumpAll = "all";
const char *const kDumpKernelsDumpOp = "DumpDataInfo";
}  // namespace

namespace ge {
DumpOp::~DumpOp() {
  if (proto_dev_mem_ != nullptr) {
    (void)rtFree(proto_dev_mem_);
  }
  if (proto_size_dev_mem_ != nullptr) {
    (void)rtFree(proto_size_dev_mem_);
  }
  proto_dev_mem_ = nullptr;
  proto_size_dev_mem_ = nullptr;
}

void DumpOp::SetLoopAddr(void *global_step, void *loop_per_iter, void *loop_cond) {
  global_step_ = reinterpret_cast<uintptr_t>(global_step);
  loop_per_iter_ = reinterpret_cast<uintptr_t>(loop_per_iter);
  loop_cond_ = reinterpret_cast<uintptr_t>(loop_cond);
}

void DumpOp::SetDynamicModelInfo(const string &dynamic_model_name, const string &dynamic_om_name,
                                 uint32_t dynamic_model_id) {
  dynamic_model_name_ = dynamic_model_name;
  dynamic_om_name_ = dynamic_om_name;
  dynamic_model_id_ = dynamic_model_id;
}

static void SetOpMappingLoopAddr(uintptr_t step_id, uintptr_t loop_per_iter, uintptr_t loop_cond,
                                 toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  if (step_id != 0) {
    GELOGI("Exists step_id.");
    op_mapping_info.set_step_id_addr(static_cast<uint64_t>(step_id));
  } else {
    GELOGI("step_id is null.");
  }

  if (loop_per_iter != 0) {
    GELOGI("Exists loop_per_iter.");
    op_mapping_info.set_iterations_per_loop_addr(static_cast<uint64_t>(loop_per_iter));
  } else {
    GELOGI("loop_per_iter is null.");
  }

  if (loop_cond != 0) {
    GELOGI("Exists loop_cond.");
    op_mapping_info.set_loop_cond_addr(static_cast<uint64_t>(loop_cond));
  } else {
    GELOGI("loop_cond is null.");
  }
}

Status DumpOp::DumpOutput(toolkit::aicpu::dump::Task &task) {
  GELOGI("Start dump output in Launch dump op");
  const auto &output_descs = op_desc_->GetAllOutputsDesc();
  for (size_t i = 0; i < output_descs.size(); ++i) {
    toolkit::aicpu::dump::Output output;
    output.set_data_type(static_cast<int32_t>(DataTypeUtil::GetIrDataType(output_descs.at(i).GetDataType())));
    output.set_format(static_cast<int32_t>(output_descs.at(i).GetFormat()));
    for (auto dim : output_descs.at(i).GetShape().GetDims()) {
      output.mutable_shape()->add_dim(dim);
    }
    for (auto dim : output_descs.at(i).GetOriginShape().GetDims()) {
      output.mutable_origin_shape()->add_dim(dim);
    }
    int64_t output_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(output_descs.at(i), output_size) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TensorSize]Failed, output %zu, node %s(%s),",
             i, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      REPORT_CALL_ERROR("E19999", "Get output %zu tensor size of node %s(%s) failed",
                        i, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    GELOGD("Get output size in lanch dump op is %ld", output_size);
    output.set_size(output_size);
    output.set_address(static_cast<uint64_t>(output_addrs_[i]));
    task.mutable_output()->Add(std::move(output));
  }
  return SUCCESS;
}

Status DumpOp::DumpInput(toolkit::aicpu::dump::Task &task) {
  GELOGI("Start dump input in Launch dump op");
  const auto &input_descs = op_desc_->GetAllInputsDesc();
  for (size_t i = 0; i < input_descs.size(); ++i) {
    toolkit::aicpu::dump::Input input;
    input.set_data_type(static_cast<int32_t>(DataTypeUtil::GetIrDataType(input_descs.at(i).GetDataType())));
    input.set_format(static_cast<int32_t>(input_descs.at(i).GetFormat()));

    for (auto dim : input_descs.at(i).GetShape().GetDims()) {
      input.mutable_shape()->add_dim(dim);
    }
    for (auto dim : input_descs.at(i).GetOriginShape().GetDims()) {
      input.mutable_origin_shape()->add_dim(dim);
    }
    int64_t input_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(input_descs.at(i), input_size) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Get][TensorSize]Failed, input %zu, node %s(%s)",
             i, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      REPORT_CALL_ERROR("E19999", "Get input %zu tensor size of node %s(%s) failed",
                        i, op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return ACL_ERROR_GE_INTERNAL_ERROR;
    }
    GELOGD("Get input size in lanch dump op is %ld", input_size);
    input.set_size(input_size);
    input.set_address(static_cast<uint64_t>(input_addrs_[i]));
    task.mutable_input()->Add(std::move(input));
  }
  return SUCCESS;
}

void DumpOp::SetDumpInfo(const DumpProperties &dump_properties, const OpDescPtr &op_desc, vector<uintptr_t> input_addrs,
                         vector<uintptr_t> output_addrs, rtStream_t stream) {
  dump_properties_ = dump_properties;
  op_desc_ = op_desc;
  input_addrs_ = input_addrs;
  output_addrs_ = output_addrs;
  stream_ = stream;
}

Status DumpOp::ExecutorDumpOp(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  std::string proto_msg;
  size_t proto_size = op_mapping_info.ByteSizeLong();
  bool ret = op_mapping_info.SerializeToString(&proto_msg);
  if (!ret || proto_size == 0) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Serialize][Protobuf]Failed, proto_size is %zu",
           proto_size);
    REPORT_CALL_ERROR("E19999", "[Serialize][Protobuf]Failed, proto_size is %zu", proto_size);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  rtError_t rt_ret = rtMalloc(&proto_dev_mem_, proto_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Call][rtMalloc]Failed, ret: 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtMemcpy(proto_dev_mem_, proto_size, proto_msg.c_str(), proto_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Call][rtMemcpy]Failed, ret: 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtMalloc(&proto_size_dev_mem_, sizeof(size_t), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Call][rtMalloc]Failed, ret: 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtMemcpy(proto_size_dev_mem_, sizeof(size_t), &proto_size, sizeof(size_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Call][rtMemcpy]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  constexpr int32_t io_addr_num = 2;
  constexpr uint32_t args_size = sizeof(aicpu::AicpuParamHead) + io_addr_num * sizeof(uint64_t);
  char args[args_size] = {0};
  auto param_head = reinterpret_cast<aicpu::AicpuParamHead *>(args);
  param_head->length = args_size;
  param_head->ioAddrNum = io_addr_num;
  auto io_addr = reinterpret_cast<uint64_t *>(args + sizeof(aicpu::AicpuParamHead));
  io_addr[0] = reinterpret_cast<uintptr_t>(proto_dev_mem_);
  io_addr[1] = reinterpret_cast<uintptr_t>(proto_size_dev_mem_);
  rt_ret = rtCpuKernelLaunch(nullptr, kDumpKernelsDumpOp,
                             1,  // blockDim default 1
                             args, args_size,
                             nullptr,  // no need smDesc
                             stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Call][rtCpuKernelLaunch]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtCpuKernelLaunch failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGI("Kernel launch dump op success");
  return SUCCESS;
}

Status DumpOp::SetDumpModelName(toolkit::aicpu::dump::OpMappingInfo &op_mapping_info) {
  if (dynamic_model_name_.empty() && dynamic_om_name_.empty()) {
    GELOGI("Single op dump, no need set model name");
    return SUCCESS;
  }
  std::set<std::string> model_list = dump_properties_.GetAllDumpModel();
  bool not_find_by_omname = model_list.find(dynamic_om_name_) == model_list.end();
  bool not_find_by_modelname = model_list.find(dynamic_model_name_) == model_list.end();
  std::string dump_model_name = not_find_by_omname ? dynamic_model_name_ : dynamic_om_name_;
  if (model_list.find(DUMP_ALL_MODEL) == model_list.end()) {
    if (not_find_by_omname && not_find_by_modelname) {
      std::string model_list_str;
      for (auto &model : model_list) {
        model_list_str += "[" + model + "].";
      }
      GELOGW("Model %s will not be set to dump, dump list: %s", dump_model_name.c_str(), model_list_str.c_str());
      return FAILED;
    }
  }
  if (!dump_model_name.empty() && dump_properties_.IsDumpOpen()) {
    GELOGI("Dump model name is %s", dump_model_name.c_str());
    op_mapping_info.set_model_name(dump_model_name);
  }
  return SUCCESS;
}

Status DumpOp::LaunchDumpOp() {
  GELOGI("Start to launch dump op %s", op_desc_->GetName().c_str());
  int32_t device_id = 0;
  rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(rt_ret, "[Call][rtGetDevice]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "[Call][rtGetDevice]Failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if (device_id < 0) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Check][DeviceId]Failed, device_id %d", device_id);
    REPORT_INNER_ERROR("E19999", "Check device_id %d failed", device_id);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }
  toolkit::aicpu::dump::OpMappingInfo op_mapping_info;
  auto dump_path = dump_properties_.GetDumpPath() + std::to_string(device_id) + "/";
  op_mapping_info.set_dump_path(dump_path);
  op_mapping_info.set_flag(kAicpuLoadFlag);
  op_mapping_info.set_dump_step(dump_properties_.GetDumpStep());
  op_mapping_info.set_model_id(dynamic_model_id_);

  if (SetDumpModelName(op_mapping_info) != SUCCESS) {
    return SUCCESS;
  }
  SetOpMappingLoopAddr(global_step_, loop_per_iter_, loop_cond_, op_mapping_info);
  GELOGI("Dump step is %s ,dump path is %s in Launch dump op", dump_properties_.GetDumpStep().c_str(),
         dump_path.c_str());
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  rt_ret = rtGetTaskIdAndStreamID(&task_id, &stream_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGW("call rtGetTaskIdAndStreamID failed, ret = 0x%X", rt_ret);
  }
  toolkit::aicpu::dump::Task task;
  task.set_task_id(task_id);
  task.set_stream_id(stream_id);
  task.mutable_op()->set_op_name(op_desc_->GetName());
  task.mutable_op()->set_op_type(op_desc_->GetType());
  if (dump_properties_.GetDumpMode() == kDumpOutput) {
    auto ret = DumpOutput(task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Output]Failed, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_CALL_ERROR("E19999", "Dump Output failed, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  if (dump_properties_.GetDumpMode() == kDumpInput) {
    auto ret = DumpInput(task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Input]Failed, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_CALL_ERROR("E19999", "Dump Input failed, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  if (dump_properties_.GetDumpMode() == kDumpAll || dump_properties_.IsOpDebugOpen()) {
    auto ret = DumpOutput(task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Output]Failed when in dumping all, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_CALL_ERROR("E19999", "Dump Output failed when in dumping all, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    ret = DumpInput(task);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Dump][Input]Failed when in dumping all, node %s(%s), ret 0x%X",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      REPORT_CALL_ERROR("E19999", "Dump Input failed when in dumping all, node %s(%s), ret 0x%X",
                        op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ret);
      return ret;
    }
    op_mapping_info.mutable_task()->Add(std::move(task));
  }
  auto ret = ExecutorDumpOp(op_mapping_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Dump][Op]Failed, ret 0x%X", ret);
    REPORT_CALL_ERROR("E19999", "Executor dump op failed, ret 0x%X", ret);
    return ret;
  }
  return SUCCESS;
}
}  // namespace ge
