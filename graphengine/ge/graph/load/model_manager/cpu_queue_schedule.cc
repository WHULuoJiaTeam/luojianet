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

#include "graph/load/model_manager/cpu_queue_schedule.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"

namespace {
const uint32_t kCoreDim = 1;  // for rtCpuKernelLaunch
const char *const kCpuTaskModelEnqueue = "modelEnqueue";
const char *const kCpuTaskWaitEndGraph = "modelWaitEndGraph";
const char *const kCpuTaskPrepareOutput = "bufferPrepareOutput";
const char *const kCpuTaskModelDequeue = "modelDequeue";
const char *const kCpuTaskModelRepeat = "modelRepeat";
const char *const kCpuTaskZeroCopy = "zeroCpy";
}  // namespace

namespace ge {
CpuTaskInfo::CpuTaskInfo(rtStream_t stream) : args_(nullptr), args_size_(0) { stream_ = stream; }

CpuTaskInfo::~CpuTaskInfo() {
  if (args_ == nullptr) {
    return;
  }

  rtError_t status = rtFree(args_);
  if (status != RT_ERROR_NONE) {
    GELOGW("Call rt free failed, status: 0x%x", status);
  }
  args_ = nullptr;
}
///
/// @ingroup ge
/// @brief definiteness queue schedule, bind input queue to task.
/// @param [in] queue_id: input queue id from user.
/// @param [out] in_mbuf: input mbuf addr for input data.
/// @return: 0 for success / others for failed
///
Status CpuTaskModelDequeue::Init(uint32_t queue_id, uintptr_t &in_mbuf) {
  if ((args_ != nullptr) || (args_size_ > 0)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = sizeof(MbufQueueInfo) + sizeof(uintptr_t);  // sizeof(uintptr_t) for save in_mbuf.
  rtError_t status = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }
  in_mbuf = reinterpret_cast<uintptr_t>(args_) + sizeof(MbufQueueInfo);
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_)

  MbufQueueInfo queue_info;
  queue_info.queue_id = queue_id;
  queue_info.in_mbuf = in_mbuf;  // Placeholder, input mbuf addr will save to this place.
  status = rtMemcpy(args_, args_size_, &queue_info, sizeof(MbufQueueInfo), RT_MEMCPY_HOST_TO_DEVICE);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  return SUCCESS;
}

Status CpuTaskModelDequeue::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0) || (stream_ == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  rtError_t status = rtCpuKernelLaunch(nullptr, kCpuTaskModelDequeue, kCoreDim, args_, args_size_, nullptr, stream_);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCpuKernelLaunch failed, ret:0x%X", status);
    GELOGE(RT_FAILED, "[Call][RtCpuKernelLaunch] failed, ret:0x%X", status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  GELOGI("Cpu kernel launch model dequeue task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, zero copy.
/// @param [in] mbuf_list: input/output mbuf addr list for input/output data.
/// @param [in] outside_addrs: model input/output memory addr
/// @return: 0 for success / others for failed
///
Status CpuTaskZeroCopy::Init(std::vector<uintptr_t> &mbuf_list, const map<uint32_t, ZeroCopyOffset> &outside_addrs) {
  if ((args_ != nullptr) || (args_size_ > 0)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = sizeof(AddrMapInfo);
  GE_CHK_RT_RET(rtMalloc(&args_, args_size_, RT_MEMORY_HBM));
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_)

  AddrMapInfo addr_map_info;
  // init src_addrs/dst_addrs
  vector<uint64_t> src_addrs;
  vector<uint64_t> dst_addrs;
  for (const auto &addrs : outside_addrs) {
    const auto &addrs_mapping_list = addrs.second.GetOutsideAddrs();
    GE_CHK_BOOL_EXEC(!addrs_mapping_list.empty(), return PARAM_INVALID, "[Check][Param] not set outside_addrs");
    std::map<const void *, std::vector<void *>> virtual_args_addrs = addrs_mapping_list[0];
    for (const auto &virtual_args_addr : virtual_args_addrs) {
      addr_map_info.addr_num += virtual_args_addr.second.size();
      for (size_t i = 0; i < virtual_args_addr.second.size(); ++i) {
        src_addrs.emplace_back(mbuf_list.at(addrs.first));
        dst_addrs.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(virtual_args_addr.second.at(i))));
      }
    }
  }
  GELOGI("addr_map_info.addr_num is %u", addr_map_info.addr_num);

  // malloc mem for src_addrs/dst_addrs, and copy data of src_addrs/dst_addrs
  GE_CHK_RT_RET(rtMalloc(&src_addr_, src_addrs.size() * sizeof(uint64_t), RT_MEMORY_HBM));
  rtError_t status = rtMemcpy(src_addr_, src_addrs.size() * sizeof(uint64_t), src_addrs.data(),
                              src_addrs.size() * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
  GE_IF_BOOL_EXEC(status != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%lu, ret:0x%X",
                                    src_addrs.size() * sizeof(uint64_t), status);
                  GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%lu, ret:0x%X",
                         src_addrs.size() * sizeof(uint64_t), status);
                  return RT_ERROR_TO_GE_STATUS(status);)

  GE_CHK_RT_RET(rtMalloc(&dst_addr_, dst_addrs.size() * sizeof(uint64_t), RT_MEMORY_HBM));
  status = rtMemcpy(dst_addr_, dst_addrs.size() * sizeof(uint64_t), dst_addrs.data(),
                    dst_addrs.size() * sizeof(uint64_t), RT_MEMCPY_HOST_TO_DEVICE);
  GE_IF_BOOL_EXEC(status != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%lu, ret:0x%X",
                                    dst_addrs.size() * sizeof(uint64_t), status);
                  GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%lu, ret:0x%X",
                         dst_addrs.size() * sizeof(uint64_t), status);
                  return RT_ERROR_TO_GE_STATUS(status);)

  // src_addr_list is init to src_addr, which is the point to src_addrs
  if (!src_addrs.empty() && !dst_addrs.empty()) {
    addr_map_info.src_addr_list = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_addr_));
    addr_map_info.dst_addr_list = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst_addr_));
    GELOGI("src_addr_list is %lu, dst_addr_list is %lu", addr_map_info.src_addr_list, addr_map_info.dst_addr_list);
  }

  status = rtMemcpy(args_, args_size_, &addr_map_info, sizeof(AddrMapInfo), RT_MEMCPY_HOST_TO_DEVICE);
  GE_IF_BOOL_EXEC(status != RT_ERROR_NONE,
                  REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_, status);
                  GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_, status);
                  return RT_ERROR_TO_GE_STATUS(status);)
  return SUCCESS;
}

Status CpuTaskZeroCopy::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0) || (stream_ == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  rtError_t status = rtCpuKernelLaunch(nullptr, kCpuTaskZeroCopy, kCoreDim, args_, args_size_, nullptr, stream_);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCpuKernelLaunch failed, ret:0x%X", status);
    GELOGE(RT_FAILED, "[Call][RtCpuKernelLaunch] failed, ret:0x%X", status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  GELOGI("Cpu kernel launch zero copy task success.");
  return SUCCESS;
}

CpuTaskZeroCopy::~CpuTaskZeroCopy() {
  if (src_addr_ == nullptr && dst_addr_ == nullptr) {
    return;
  }
  if (src_addr_ != nullptr) {
    rtError_t status = rtFree(src_addr_);
    if (status != RT_ERROR_NONE) {
      GELOGW("Call rt free failed, status: 0x%x", status);
    }
  }
  if (dst_addr_ != nullptr) {
    rtError_t status = rtFree(dst_addr_);
    if (status != RT_ERROR_NONE) {
      GELOGW("Call rt free failed, status: 0x%x", status);
    }
  }
  src_addr_ = nullptr;
  dst_addr_ = nullptr;
}
///
/// @ingroup ge
/// @brief definiteness queue schedule, bind output queue to task.
/// @param [in] addr: NetOutput Op input tensor address.
/// @param [in] size: NetOutput Op input tensor size.
/// @param [in] in_mbuf: input mbuf addr for input data.
/// @param [out] out_mbuf: output mbuf addr for output data.
/// @return: 0 for success / others for failed
///
Status CpuTaskPrepareOutput::Init(uintptr_t addr, uint32_t size, uintptr_t in_mbuf, uintptr_t &out_mbuf) {
  if ((args_ != nullptr) || (args_size_ > 0)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = sizeof(PrepareOutputInfo) + sizeof(uintptr_t);  // sizeof(uintptr_t) for save out_mbuf.
  rtError_t status = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }
  out_mbuf = reinterpret_cast<uintptr_t>(args_) + sizeof(PrepareOutputInfo);
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_)

  // Get NetOutput Input address and bind to queue.
  PrepareOutputInfo prepare;
  prepare.data_size = size;
  prepare.data_addr = addr;
  prepare.in_mbuf = in_mbuf;
  prepare.out_mbuf = out_mbuf;  // Placeholder, output mbuf addr will save to this place.
  status = rtMemcpy(args_, args_size_, &prepare, sizeof(PrepareOutputInfo), RT_MEMCPY_HOST_TO_DEVICE);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  return SUCCESS;
}

Status CpuTaskPrepareOutput::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0) || (stream_ == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  rtError_t status = rtCpuKernelLaunch(nullptr, kCpuTaskPrepareOutput, kCoreDim, args_, args_size_, nullptr, stream_);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCpuKernelLaunch failed, ret:0x%X", status);
    GELOGE(RT_FAILED, "[Call][RtCpuKernelLaunch] failed, ret:0x%X", status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  GELOGI("Cpu kernel launch prepare output task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, bind output queue to task.
/// @param [in] queue_id: output queue id from user.
/// @param [in] out_mbuf: mbuf for output data.
/// @return: 0 for success / others for failed
///
Status CpuTaskModelEnqueue::Init(uint32_t queue_id, uintptr_t out_mbuf) {
  if ((args_ != nullptr) || (args_size_ > 0)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  // Get NetOutput Input address and bind to queue.
  args_size_ = sizeof(MbufQueueInfo);
  rtError_t status = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_)

  MbufQueueInfo queue_info;
  queue_info.queue_id = queue_id;
  queue_info.in_mbuf = out_mbuf;
  status = rtMemcpy(args_, args_size_, &queue_info, args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  return SUCCESS;
}

Status CpuTaskModelEnqueue::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0) || (stream_ == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is nullptr or args_size_ is 0 or stream_ is nullptr, arg_size:%u,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  rtError_t status = rtCpuKernelLaunch(nullptr, kCpuTaskModelEnqueue, kCoreDim, args_, args_size_, nullptr, stream_);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCpuKernelLaunch failed, ret:0x%X", status);
    GELOGE(RT_FAILED, "[Call][RtCpuKernelLaunch] failed, ret:0x%X", status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  GELOGI("Cpu kernel launch model enqueue task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, active entry stream.
/// @param [in] stream: stream to be active.
/// @return: 0 for success / others for failed
///
Status CpuTaskActiveEntry::Init(rtStream_t stream) {
  if (stream == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param stream is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] Task active stream not valid");
    return FAILED;
  }

  active_stream_ = stream;
  return SUCCESS;
}

Status CpuTaskActiveEntry::Distribute() {
  if ((active_stream_ == nullptr) || (stream_ == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Param stream is nullptr or active_stream_ is nullptr, check invalid");
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  rtError_t ret = rtStreamActive(active_stream_, stream_);
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamActive failed, ret:0x%X", ret);
    GELOGE(RT_FAILED, "[Call][RtStreamActive] failed, ret:0x%X", ret);
    return RT_ERROR_TO_GE_STATUS(ret);
  }

  GELOGI("Cpu kernel launch active entry task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, wait for end graph.
/// @param [in] model_id: model id for wait end graph.
/// @return: 0 for success / others for failed
///
Status CpuTaskWaitEndGraph::Init(uint32_t model_id) {
  if ((args_ != nullptr) || (args_size_ > 0)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = sizeof(model_id);
  rtError_t status = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_)

  status = rtMemcpy(args_, args_size_, &model_id, args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  return SUCCESS;
}

Status CpuTaskWaitEndGraph::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0) || (stream_ == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  rtError_t status = rtCpuKernelLaunch(nullptr, kCpuTaskWaitEndGraph, kCoreDim, args_, args_size_, nullptr, stream_);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCpuKernelLaunch failed, ret:0x%X", status);
    GELOGE(RT_FAILED, "[Call][RtCpuKernelLaunch] failed, ret:0x%X", status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  GELOGI("Cpu kernel launch wait end task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, repeat run model.
/// @param [in] model_id: model id for repeat run.
/// @return: 0 for success / others for failed
///
Status CpuTaskModelRepeat::Init(uint32_t model_id) {
  if ((args_ != nullptr) || (args_size_ > 0)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is not nullptr or args_size_:%u > 0, check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task already initialized, size:%u", args_size_);
    return FAILED;
  }

  args_size_ = sizeof(model_id);
  rtError_t status = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "args data.", args_size_)

  status = rtMemcpy(args_, args_size_, &model_id, args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_, status);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  return SUCCESS;
}

Status CpuTaskModelRepeat::Distribute() {
  if ((args_ == nullptr) || (args_size_ == 0) || (stream_ == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Param args_ is nullptr or args_size_:%u is 0 or stream_ is nullptr,"
                       "check invalid", args_size_);
    GELOGE(FAILED, "[Check][Param] Task not initialized, distribute failed, size:%u", args_size_);
    return FAILED;
  }

  rtError_t status = rtCpuKernelLaunch(nullptr, kCpuTaskModelRepeat, kCoreDim, args_, args_size_, nullptr, stream_);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCpuKernelLaunch failed, ret:0x%X", status);
    GELOGE(RT_FAILED, "[Call][RtCpuKernelLaunch] failed, ret:0x%X", status);
    return RT_ERROR_TO_GE_STATUS(status);
  }

  GELOGI("Cpu kernel launch repeat task success.");
  return SUCCESS;
}
}  // namespace ge
