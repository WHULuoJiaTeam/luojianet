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

#include "common/dump/exception_dumper.h"

#include "common/ge/datatype_util.h"
#include "common/debug/memory_dumper.h"
#include "framework/common/debug/log.h"
#include "graph/manager/util/debug.h"
#include "graph/utils/tensor_utils.h"
#include "graph/load/model_manager/model_utils.h"
#include "proto/dump_task.pb.h"

namespace {
static uint64_t GetNowTime() {
  uint64_t ret = 0;
  mmTimeval tv;
  if (mmGetTimeOfDay(&tv, nullptr) == 0) {
    ret = tv.tv_sec * 1000000ULL + tv.tv_usec;
  }

  return ret;
}

static void ReplaceStringElem(std::string &str) {
  for_each(str.begin(), str.end(), [](char &ch) {
    if ((ch == ' ') || (ch == '.') || (ch == '/') || (ch == '\\')) {
      ch = '_';
    }
  });
}

static void SetDumpData(const ge::OpDescInfo &op_desc_info, toolkit::dump::DumpData &dump_data) {
  dump_data.set_version("2.0");
  dump_data.set_dump_time(GetNowTime());
  dump_data.set_op_name(op_desc_info.op_name);
  for (size_t i = 0; i < op_desc_info.input_format.size(); ++i) {
    toolkit::dump::OpInput input;
    input.set_data_type(toolkit::dump::OutputDataType(
        ge::DataTypeUtil::GetIrDataType(op_desc_info.input_data_type[i])));
    input.set_format(toolkit::dump::OutputFormat(op_desc_info.input_format[i]));
    for (auto dim : op_desc_info.input_shape[i]) {
      input.mutable_shape()->add_dim(dim);
    }
    input.set_size(op_desc_info.input_size[i]);
    GELOGI("[Set][DumpData] The input size int exception is %ld", op_desc_info.input_size[i]);
    dump_data.mutable_input()->Add(std::move(input));
  }

  for (size_t j = 0; j < op_desc_info.output_format.size(); ++j) {
    toolkit::dump::OpOutput output;
    output.set_data_type(toolkit::dump::OutputDataType(
        ge::DataTypeUtil::GetIrDataType(op_desc_info.output_data_type[j])));
    output.set_format(toolkit::dump::OutputFormat(op_desc_info.output_format[j]));
    for (auto dim : op_desc_info.output_shape[j]) {
      output.mutable_shape()->add_dim(dim);
    }
    output.set_size(op_desc_info.output_size[j]);
    GELOGI("[Set][DumpData] The output size int exception is %ld", op_desc_info.output_size[j]);
    dump_data.mutable_output()->Add(std::move(output));
  }
}
}  // namespace

namespace ge {
ExceptionDumper::~ExceptionDumper() {}

void ExceptionDumper::SaveDumpOpInfo(const OpDescPtr &op, uint32_t task_id, uint32_t stream_id,
                                     vector<void *> &input_addrs, vector<void *> &output_addrs) {
  OpDescInfo op_desc_info;
  SaveOpDescInfo(op, task_id, stream_id, op_desc_info);
  op_desc_info.input_addrs = input_addrs;
  op_desc_info.output_addrs = output_addrs;
  op_desc_info_.emplace_back(std::move(op_desc_info));
}

void ExceptionDumper::SaveDumpOpInfo(const RuntimeParam &model_param, const OpDescPtr &op,
                                     uint32_t task_id, uint32_t stream_id) {
  OpDescInfo op_desc_info;
  SaveOpDescInfo(op, task_id, stream_id, op_desc_info);
  op_desc_info.input_addrs = ModelUtils::GetInputDataAddrs(model_param, op);
  op_desc_info.output_addrs = ModelUtils::GetOutputDataAddrs(model_param, op);
  op_desc_info_.emplace_back(std::move(op_desc_info));
}

void ExceptionDumper::SaveOpDescInfo(const OpDescPtr &op, uint32_t task_id, uint32_t stream_id,
                                     OpDescInfo &op_desc_info) {
  if (op == nullptr) {
    GELOGW("[Save][OpExceptionInfo] op desc ptr is null.");
    return;
  }
  GELOGD("[Save][OpExceptionInfo] Start to save dump op [%s] info of task_id: %u, stream_id: %u",
         op->GetName().c_str(), task_id, stream_id);
  op_desc_info.op_name = op->GetName();
  op_desc_info.op_type = op->GetType();
  op_desc_info.task_id = task_id;
  op_desc_info.stream_id = stream_id;
  for (size_t i = 0; i < op->GetAllInputsSize(); ++i) {
    GeTensorDescPtr input_tensor_desc = op->MutableInputDesc(i);
    if (input_tensor_desc == nullptr) {
      continue;
    }
    op_desc_info.input_format.emplace_back(input_tensor_desc->GetFormat());
    op_desc_info.input_shape.emplace_back(input_tensor_desc->GetShape().GetDims());
    op_desc_info.input_data_type.emplace_back(input_tensor_desc->GetDataType());
    int64_t input_size = 0;

    if (TensorUtils::GetTensorSizeInBytes(*input_tensor_desc, input_size) != SUCCESS) {
      GELOGW("[Save][OpExceptionInfo] Op [%s] get input size failed.", op->GetName().c_str());
      return;
    }
    GELOGD("[Save][OpExceptionInfo] Save dump op info, the input size is %ld", input_size);
    op_desc_info.input_size.emplace_back(input_size);
  }
  for (size_t j = 0; j < op->GetOutputsSize(); ++j) {
    GeTensorDescPtr output_tensor_desc = op->MutableOutputDesc(j);
    if (output_tensor_desc == nullptr) {
      continue;
    }
    op_desc_info.output_format.emplace_back(output_tensor_desc->GetFormat());
    op_desc_info.output_shape.emplace_back(output_tensor_desc->GetShape().GetDims());
    op_desc_info.output_data_type.emplace_back(output_tensor_desc->GetDataType());
    int64_t output_size = 0;
    if (TensorUtils::GetTensorSizeInBytes(*output_tensor_desc, output_size) != SUCCESS) {
      GELOGW("[Save][OpExceptionInfo] Op [%s] get output size failed.", op->GetName().c_str());
      return;
    }
    GELOGD("[Save][OpExceptionInfo] Save dump op info, the output size is %ld.", output_size);
    op_desc_info.output_size.emplace_back(output_size);
  }
}

Status ExceptionDumper::DumpExceptionInfo(const std::vector<rtExceptionInfo> &exception_infos) const {
  GELOGI("[Dump][Exception] Start to dump exception info");
  for (const rtExceptionInfo &iter : exception_infos) {
    OpDescInfo op_desc_info;
    if (GetOpDescInfo(iter.streamid, iter.taskid, op_desc_info)) {
      toolkit::dump::DumpData dump_data;
      SetDumpData(op_desc_info, dump_data);
      uint64_t now_time = GetNowTime();
      std::string op_name = op_desc_info.op_name;
      std::string op_type = op_desc_info.op_type;
      ReplaceStringElem(op_name);
      ReplaceStringElem(op_type);
      string dump_file_path =
        "./" + op_type + "." + op_name + "." + std::to_string(op_desc_info.task_id) + "." + std::to_string(now_time);
      GELOGI("[Dump][Exception] The exception dump file path is %s", dump_file_path.c_str());

      uint64_t proto_size = dump_data.ByteSizeLong();
      std::unique_ptr<char[]> proto_msg(new (std::nothrow) char[proto_size]);
      GE_CHECK_NOTNULL(proto_msg);
      bool ret = dump_data.SerializeToArray(proto_msg.get(), proto_size);
      if (!ret || proto_size == 0) {
        REPORT_INNER_ERROR("E19999", "Serialize proto to string fail");
        GELOGE(PARAM_INVALID, "[Dump][Exception] Dump data proto serialize failed");
        return PARAM_INVALID;
      }

      GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(dump_file_path.c_str(), &proto_size, sizeof(uint64_t)),
                        "Failed to dump proto size");
      GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(dump_file_path.c_str(), proto_msg.get(), proto_size),
                        "Failed to dump proto msg");
      if (DumpExceptionInput(op_desc_info, dump_file_path) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Dump][Exception] Dump exception input failed");
        return PARAM_INVALID;
      }

      if (DumpExceptionOutput(op_desc_info, dump_file_path) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Dump][Exception] Dump exception output failed");
        return PARAM_INVALID;
      }
      GELOGI("[Dump][Exception] Dump exception info SUCCESS");
    } else {
      GELOGE(PARAM_INVALID, "[Dump][Exception] Get op desc info failed,task id:%u,stream id:%u",
             iter.taskid, iter.streamid);
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

bool ExceptionDumper::GetOpDescInfo(uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info) const {
  GELOGI("[Get][OpDescInfo] There are %zu op need to dump.", op_desc_info_.size());
  for (size_t index = 0; index < op_desc_info_.size(); ++index) {
    OpDescInfo dump_op_info = op_desc_info_.at(index);
    if (dump_op_info.task_id == task_id && dump_op_info.stream_id == stream_id) {
      GELOGI("[Get][OpDescInfo] Find exception op [%s] of task_id: %u, stream_id: %u.",
             dump_op_info.op_name.c_str(), task_id, stream_id);
      op_desc_info = dump_op_info;
      return true;
    }
  }
  return false;
}

Status ExceptionDumper::DumpExceptionInput(const OpDescInfo &op_desc_info, const string &dump_file) const {
  GELOGI("[Dump][ExceptionInput] Start to dump exception input");
  for (size_t i = 0; i < op_desc_info.input_addrs.size(); i++) {
    if (Debug::DumpDevMem(dump_file.data(), op_desc_info.input_addrs.at(i), op_desc_info.input_size.at(i)) != SUCCESS) {
      GELOGE(PARAM_INVALID, "[Dump][ExceptionInput] Dump the %zu input data of op [%s] failed",
             i, op_desc_info.op_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status ExceptionDumper::DumpExceptionOutput(const OpDescInfo &op_desc_info, const string &dump_file) const {
  GELOGI("[Dump][ExceptionOutput] Start to dump exception output");
  for (size_t i = 0; i < op_desc_info.output_addrs.size(); i++) {
    if (Debug::DumpDevMem(dump_file.data(), op_desc_info.output_addrs.at(i), op_desc_info.output_size.at(i)) !=
        SUCCESS) {
      GELOGE(PARAM_INVALID, "[Dump][ExceptionInput] Dump the %zu input data of op [%s] failed",
             i, op_desc_info.op_name.c_str());
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

OpDescInfo *ExceptionDumper::MutableOpDescInfo(uint32_t task_id, uint32_t stream_id) {
  for (OpDescInfo &op_desc_info : op_desc_info_) {
    if (op_desc_info.task_id == task_id && op_desc_info.stream_id == stream_id) {
      return &op_desc_info;
    }
  }
  return nullptr;
}
}  // namespace ge