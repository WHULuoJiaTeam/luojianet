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

#include "graph/load/model_manager/task_info/memcpy_async_task_info.h"

#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/davinci_model.h"

namespace ge {
Status MemcpyAsyncTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("MemcpyAsyncTaskInfo Init Start");
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;

  Status ret = SetStream(task_def.stream_id(), davinci_model_->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  const domi::MemcpyAsyncDef &memcpy_async = task_def.memcpy_async();
  count_ = memcpy_async.count();
  kind_ = memcpy_async.kind();
  dst_max_ = memcpy_async.dst_max();
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(memcpy_async.op_index());
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", memcpy_async.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range", memcpy_async.op_index());
    return INTERNAL_ERROR;
  }

  if (davinci_model_->IsKnownNode()) {
    src_ = reinterpret_cast<uint8_t *>(davinci_model_->GetCurrentArgsAddr(args_offset_));
    dst_ = reinterpret_cast<uint8_t *>(reinterpret_cast<uintptr_t>(src_) + sizeof(void *));
    // for zero copy
    kind_ = RT_MEMCPY_ADDR_DEVICE_TO_DEVICE;
    GE_CHK_STATUS_RET(SetIoAddrs(op_desc, memcpy_async), "[Set][Addrs] failed, op:%s", op_desc->GetName().c_str());
    GELOGI("MemcpyAsyncTaskInfo op name %s, src_ %p, dst_ %p, args_offset %u.",
           op_desc->GetName().c_str(), src_, dst_, args_offset_);
    return SUCCESS;
  }

  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  ret = ModelUtils::GetRtAddress(rts_param, memcpy_async.src(), src_);
  if (ret != SUCCESS) {
    return ret;
  }

  // dst_ needs different address for different chips
  vector<int64_t> memory_type_list;
  (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, memory_type_list);
  if (!memory_type_list.empty() && memory_type_list[0] == RT_MEMORY_TS_4G) {  // TS Feature, Just one.
    uint64_t mem_offset = memcpy_async.dst() - rts_param.logic_mem_base;
    dst_ = static_cast<uint8_t *>(rts_param.ts_mem_mall->Acquire(mem_offset, memcpy_async.dst_max()));
    if (dst_ == nullptr) {
      return FAILED;
    }
  } else {
    ret = ModelUtils::GetRtAddress(rts_param, memcpy_async.dst(), dst_);
    if (ret != SUCCESS) {
      return ret;
    }
  }

  davinci_model_->DisableZeroCopy(src_);
  davinci_model_->DisableZeroCopy(dst_);
  GE_CHK_STATUS_RET(SetIoAddrs(op_desc, memcpy_async), "[Set][Addrs] failed, op:%s", op_desc->GetName().c_str());
  GELOGI("MemcpyAsyncTaskInfo Init Success, logic[0x%lx, 0x%lx], src:%p, dst:%p, max:%lu, count:%lu",
         memcpy_async.src(), memcpy_async.dst(), src_, dst_, dst_max_, count_);
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::Distribute() {
  GELOGI("MemcpyAsyncTaskInfo Distribute Start. dst_max:%lu, count:%lu, kind:%u", dst_max_, count_, kind_);

  rtError_t rt_ret = rtMemcpyAsync(dst_, dst_max_, src_, count_, static_cast<rtMemcpyKind_t>(kind_), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpyAsync failed, size:%lu, ret:0x%X", dst_max_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpyAsync] failed, size:%lu, ret:0x%X", dst_max_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("MemcpyAsyncTaskInfo Distribute Success");
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GE_CHECK_NOTNULL(davinci_model);
  OpDescPtr op_desc = davinci_model->GetOpByIndex(task_def.memcpy_async().op_index());
  // the num of src and dst size is 2
  uint32_t args_size = sizeof(void *) * 2;
  args_offset_ = davinci_model->GetTotalArgsSize();
  davinci_model->SetTotalArgsSize(args_size);
  davinci_model_ = davinci_model;
  GELOGI("MemcpyAsyncTaskInfo kernel args_size %u, args_offset %u", args_size, args_offset_);
  string peer_input_name;
  if (AttrUtils::GetStr(op_desc, ATTR_DYNAMIC_SHAPE_FIXED_ADDR, peer_input_name) && !peer_input_name.empty()) {
    uint32_t output_index = davinci_model->GetFixedAddrOutputIndex(peer_input_name);
    fixed_addr_offset_ = davinci_model->GetFixedAddrsSize(peer_input_name);
    auto tensor_desc = op_desc->GetOutputDesc(output_index);
    int64_t tensor_size = 0;
    GE_CHK_STATUS(TensorUtils::GetSize(tensor_desc, tensor_size));
    davinci_model->SetTotalFixedAddrsSize(peer_input_name, tensor_size);
  }
  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::SetIoAddrs(const OpDescPtr &op_desc, const domi::MemcpyAsyncDef &memcpy_async) {
  uint8_t *src = nullptr;
  Status ret = ModelUtils::GetRtAddress(davinci_model_->GetRuntimeParam(), memcpy_async.src(), src);
  if (ret != SUCCESS) {
    return ret;
  }
  io_addrs_.emplace_back(reinterpret_cast<void *>(src));

  if (op_desc->HasAttr(ATTR_DYNAMIC_SHAPE_FIXED_ADDR)) {
    void *fixed_addr = davinci_model_->GetCurrentFixedAddr(fixed_addr_offset_);
    io_addrs_.emplace_back(fixed_addr);
  } else {
    uint8_t *dst = nullptr;
    ret = ModelUtils::GetRtAddress(davinci_model_->GetRuntimeParam(), memcpy_async.dst(), dst);
    if (ret != SUCCESS) {
      return ret;
    }
    io_addrs_.emplace_back(reinterpret_cast<void *>(dst));
  }

  return SUCCESS;
}

Status MemcpyAsyncTaskInfo::UpdateArgs() {
  GELOGI("MemcpyAsyncTaskInfo::UpdateArgs in.");
  GE_CHECK_NOTNULL(davinci_model_);
  davinci_model_->SetTotalIOAddrs(io_addrs_);
  GELOGI("MemcpyAsyncTaskInfo::UpdateArgs success.");
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_MEMCPY_ASYNC, MemcpyAsyncTaskInfo);
}  // namespace ge
