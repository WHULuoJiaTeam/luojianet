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

#include "common/dump/opdebug_register.h"

namespace {
const size_t kOpDebugMemorySize = 2048UL;
const size_t kDebugP2pSize = 8UL;
}  // namespace
namespace ge {
OpdebugRegister::~OpdebugRegister() {}

Status OpdebugRegister::RegisterDebugForModel(rtModel_t model_handle, uint32_t op_debug_mode, DataDumper &data_dumper) {
  GELOGD("Start to register debug for model in overflow");
  auto ret = MallocMemForOpdebug();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Malloc][MemForOpdebug]Failed when debug for model overflow, ret:0x%X", ret);
    REPORT_CALL_ERROR("E19999",  "Malloc memory for opdebug failed when debug "
                      "for model in overflow, ret 0x%X", ret);
    return ret;
  }
  uint32_t debug_stream_id = 0;
  uint32_t debug_task_id = 0;
  auto rt_ret = rtDebugRegister(model_handle, op_debug_mode, op_debug_addr_, &debug_stream_id, &debug_task_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "rtDebugRegister error, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGD("debug_task_id:%u, debug_stream_id:%u in model overflow", debug_task_id, debug_stream_id);
  data_dumper.SaveOpDebugId(debug_task_id, debug_stream_id, p2p_debug_addr_, true);
  return SUCCESS;
}

void OpdebugRegister::UnregisterDebugForModel(rtModel_t model_handle) {
  rtError_t rt_ret = RT_ERROR_NONE;
  if (model_handle != nullptr) {
    GELOGD("start to call rtDebugUnRegister in model overflow.");
    rt_ret = rtDebugUnRegister(model_handle);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtDebugUnRegister failed, ret: 0x%X", rt_ret);
    }
  }

  if (op_debug_addr_ != nullptr) {
    rt_ret = rtFree(op_debug_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree failed, ret: 0x%X", rt_ret);
    }
    op_debug_addr_ = nullptr;
  }

  if (p2p_debug_addr_ != nullptr) {
    rt_ret = rtFree(p2p_debug_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree failed, ret: 0x%X", rt_ret);
    }
    p2p_debug_addr_ = nullptr;
  }
  return;
}

Status OpdebugRegister::RegisterDebugForStream(rtStream_t stream, uint32_t op_debug_mode, DataDumper &data_dumper) {
  GELOGD("Start to register debug for stream in stream overflow");
  auto ret = MallocMemForOpdebug();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Malloc][MemForOpdebug]Failed when debug for stream in overflow, ret:0x%X", ret);
    REPORT_CALL_ERROR("E19999", "Malloc memory for opdebug failed when debug "
                      "for stream in overflow, ret:0x%X", ret);
    return ret;
  }

  uint32_t debug_stream_id = 0;
  uint32_t debug_task_id = 0;
  auto rt_ret = rtDebugRegisterForStream(stream, op_debug_mode, op_debug_addr_, &debug_stream_id, &debug_task_id);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtDebugRegisterForStream]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtDebugRegisterForStream failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGD("debug_task_id:%u, debug_stream_id:%u in stream overflow.", debug_task_id, debug_stream_id);
  data_dumper.SaveOpDebugId(debug_task_id, debug_stream_id, p2p_debug_addr_, true);
  return SUCCESS;
}

void OpdebugRegister::UnregisterDebugForStream(rtStream_t stream) {
  rtError_t rt_ret = RT_ERROR_NONE;
  if (stream != nullptr) {
    GELOGD("start call rtDebugUnRegisterForStream in unknown shape over flow.");
    rt_ret = rtDebugUnRegisterForStream(stream);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtDebugUnRegisterForStream failed, ret: 0x%X", rt_ret);
    }
  }

  if (op_debug_addr_ != nullptr) {
    rt_ret = rtFree(op_debug_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree failed, ret: 0x%X", rt_ret);
    }
    op_debug_addr_ = nullptr;
  }

  if (p2p_debug_addr_ != nullptr) {
    rt_ret = rtFree(p2p_debug_addr_);
    if (rt_ret != RT_ERROR_NONE) {
      GELOGW("rtFree failed, ret: 0x%X", rt_ret);
    }
    p2p_debug_addr_ = nullptr;
  }
  return;
}

Status OpdebugRegister::MallocMemForOpdebug() {
  rtError_t rt_ret = rtMalloc(&op_debug_addr_, kOpDebugMemorySize, RT_MEMORY_DDR);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtMalloc]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  uint64_t debug_addrs_tmp = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(op_debug_addr_));
  // For data dump, aicpu needs the pointer to pointer that save the real debug address.
  rt_ret = rtMalloc(&p2p_debug_addr_, kDebugP2pSize, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtMalloc]Failed, ret 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, ret 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtMemcpy(p2p_debug_addr_, sizeof(uint64_t), &debug_addrs_tmp, sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Call][rtMemcpy]To p2p_addr error 0x%X", rt_ret);
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy to p2p_addr error 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  return SUCCESS;
}

}  // namespace ge
