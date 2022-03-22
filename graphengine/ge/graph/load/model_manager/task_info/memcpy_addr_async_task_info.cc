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

#include "graph/load/model_manager/task_info/memcpy_addr_async_task_info.h"

#include "framework/common/debug/ge_log.h"
#include "graph/load/model_manager/davinci_model.h"

namespace {
const uint32_t kAlignBytes = 64;
}

namespace ge {
Status MemcpyAddrAsyncTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("MemcpyAddrAsyncTaskInfo Init Start");
  GE_CHECK_NOTNULL(davinci_model);

  Status ret = SetStream(task_def.stream_id(), davinci_model->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  const auto &memcpy_async = task_def.memcpy_async();
  OpDescPtr op_desc = davinci_model->GetOpByIndex(memcpy_async.op_index());
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", memcpy_async.op_index());
    GELOGE(INTERNAL_ERROR, "[Get][Op] Task op index:%u out of range", memcpy_async.op_index());
    return INTERNAL_ERROR;
  }

  const RuntimeParam &rts_param = davinci_model->GetRuntimeParam();
  ret = ModelUtils::GetRtAddress(rts_param, memcpy_async.src(), src_);
  if (ret != SUCCESS) {
    return ret;
  }

  ret = ModelUtils::GetRtAddress(rts_param, memcpy_async.dst(), dst_);
  if (ret != SUCCESS) {
    return ret;
  }

  vector<void *> io_addrs;
  io_addrs.emplace_back(src_);
  io_addrs.emplace_back(dst_);

  // malloc args memory
  size_t args_size = sizeof(void *) * io_addrs.size();
  rtMemType_t memory_type = op_desc->HasAttr(ATTR_NAME_MEMORY_TYPE_RANGE) ? RT_MEMORY_TS_4G : RT_MEMORY_HBM;
  GELOGI("memory_type: %u", memory_type);
  rtError_t rt_ret = rtMalloc(&args_, args_size + kAlignBytes, memory_type);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed for op:%s(%s), size:%lu, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                      args_size + kAlignBytes, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed for op:%s(%s), size:%lu, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size + kAlignBytes, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  args_align_ = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(args_) / kAlignBytes + 1) * kAlignBytes);
  // copy orign src/dst
  GELOGI("src_args:%p, destMax:%zu, src_:%p, dst_args:%p, dst_:%p, count=%zu", args_align_, args_size, src_,
         static_cast<uint8_t *>(args_align_) + args_size, dst_, io_addrs.size());
  rt_ret = rtMemcpy(args_align_, args_size, io_addrs.data(), args_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed for op:%s(%s), size:%zu, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] for src failed for op:%s(%s), size:%zu, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  count_ = memcpy_async.count();
  kind_ = memcpy_async.kind();
  dst_max_ = memcpy_async.dst_max();
  GELOGI("InitMemcpyAddrAsyncTaskInfo, logic[0x%lx, 0x%lx], src:%p, dst:%p, max:%lu, count:%lu, args:%p, size:%zu",
         memcpy_async.src(), memcpy_async.dst(), src_, dst_, dst_max_, count_, args_align_, args_size);

  davinci_model->SetZeroCopyAddr(op_desc, io_addrs, io_addrs.data(), args_align_, args_size, 0);
  return SUCCESS;
}

Status MemcpyAddrAsyncTaskInfo::Distribute() {
  GELOGI("MemcpyAddrAsyncTaskInfo Distribute Start, dst_max:%lu, count:%lu, kind:%u", dst_max_, count_, kind_);

  rtError_t rt_ret = rtMemcpyAsync(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(args_align_) + sizeof(void *)),
                                   dst_max_, args_align_, count_, static_cast<rtMemcpyKind_t>(kind_), stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpyAsync failed, size:%lu, ret:0x%X", dst_max_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpyAsync] failed, size:%lu, ret:0x%X", dst_max_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_MEMCPY_ADDR_ASYNC, MemcpyAddrAsyncTaskInfo);
}  // namespace ge
