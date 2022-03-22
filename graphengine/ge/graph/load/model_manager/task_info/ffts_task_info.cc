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

#include "graph/load/model_manager/task_info/ffts_task_info.h"

#include <vector>

#include "graph/load/model_manager/davinci_model.h"

namespace {
constexpr uint32_t kAddrLen = sizeof(void *);
}
namespace ge {
FftsTaskInfo::~FftsTaskInfo() {
  GE_FREE_RT_LOG(args_);
}

Status FftsTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GELOGI("FftsTaskInfo Init Start.");
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model_->GetStreamList()));

  const domi::FftsTaskDef &ffts_task_def = task_def.ffts_task();
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(ffts_task_def.op_index());
  GE_CHECK_NOTNULL(op_desc);

  if ((ffts_task_def.sub_task_size() > static_cast<int>(RT_FFTS_MAX_SUB_TASK_NUM)) ||
      (ffts_task_def.ticket_cache_size() > static_cast<int>(RT_FFTS_MAX_TICKET_CACHE_NUM))) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] failed. Node: %s, sub task desc size: %d, ticket cache size: %d",
           op_desc->GetName().c_str(), ffts_task_def.sub_task_size(), ffts_task_def.ticket_cache_size());
    return INTERNAL_ERROR;
  }
  args_size_ = kAddrLen * ffts_task_def.addr_size();
  GE_CHK_RT_RET(rtMalloc(&args_, args_size_, RT_MEMORY_HBM));
  InitFftsDescInfo(ffts_task_def.ffts_desc(), sub_task_info_.fftsDesc);

  sub_task_info_.fftsType = static_cast<rtFftsType_t>(ffts_task_def.ffts_type());
  sub_task_info_.subTaskNum = ffts_task_def.sub_task_size();
  for (int idx = 0; idx < ffts_task_def.sub_task_size(); ++idx) {
    GE_CHK_STATUS_RET_NOLOG(InitSubTaskInfo(ffts_task_def.sub_task(idx), sub_task_info_.subTask[idx]));
  }

  sub_task_info_.tickCacheNum = ffts_task_def.ticket_cache_size();
  for (int idx = 0; idx < ffts_task_def.ticket_cache_size(); ++idx) {
    GE_CHK_STATUS_RET_NOLOG(InitTicketCache(ffts_task_def.ticket_cache(idx), sub_task_info_.ticketCache[idx]));
  }

  size_t data_size = kAddrLen * io_addrs_.size();
  GE_CHK_RT_RET(rtMemcpy(args_, args_size_, io_addrs_.data(), data_size, RT_MEMCPY_HOST_TO_DEVICE));
  GELOGI("FftsTaskInfo::Init Success. Node: %s, input/output size: %zu", op_desc->GetName().c_str(), io_addrs_.size());
  return SUCCESS;
}

void FftsTaskInfo::InitFftsDescInfo(const domi::FftsDescInfoDef &ffts_desc_def, rtFftsDescInfo_t &ffts_desc) {
  ffts_desc.tm = static_cast<uint8_t>(ffts_desc_def.tm());
  ffts_desc.di = static_cast<uint8_t>(ffts_desc_def.di());
  ffts_desc.dw = static_cast<uint8_t>(ffts_desc_def.dw());
  ffts_desc.df = static_cast<uint8_t>(ffts_desc_def.df());
  ffts_desc.dataSplitUnit = static_cast<uint8_t>(ffts_desc_def.data_split_unit());
  ffts_desc.prefetchOstNum = static_cast<uint8_t>(ffts_desc_def.prefetch_ost_num());
  ffts_desc.cacheMaintainOstNum = static_cast<uint8_t>(ffts_desc_def.cache_maintain_ost_num());
  ffts_desc.aicPrefetchUpper = static_cast<uint8_t>(ffts_desc_def.aic_prefetch_upper());
  ffts_desc.aicPrefetchLower = static_cast<uint8_t>(ffts_desc_def.aic_prefetch_lower());
  ffts_desc.aivPrefetchUpper = static_cast<uint8_t>(ffts_desc_def.aiv_prefetch_upper());
  ffts_desc.aivPrefetchLower = static_cast<uint8_t>(ffts_desc_def.aiv_prefetch_lower());
}

Status FftsTaskInfo::InitSubTaskInfo(const domi::FftsSubTaskDef &sub_task_def, rtFftsSubTaskInfo_t &sub_task_desc) {
  if ((sub_task_def.dst_tick_cache_id_size() > static_cast<int>(RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK)) ||
      (sub_task_def.src_tick_cache_id_size() > static_cast<int>(RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK))) {
    GELOGE(FAILED, "[Check][Param] Invalid FftsSubTaskInfo, dst tick cache id size: %d, src tick cache id size: %d",
           sub_task_def.dst_tick_cache_id_size(), sub_task_def.src_tick_cache_id_size());
    return FAILED;
  }

  if (sub_task_def.has_auto_thread_aic_aiv() == sub_task_def.has_manual_thread_aic_aiv()) {
    GELOGE(FAILED, "[Check][Param] Invalid FftsSubTaskInfo, auto thread aic/aiv: %d, manual thread aic/aiv: %d",
           sub_task_def.has_auto_thread_aic_aiv(), sub_task_def.has_manual_thread_aic_aiv());
    return FAILED;
  }

  thread_dim_ = sub_task_def.thread_dim();
  GE_CHK_BOOL_RET_STATUS(thread_dim_ != 0, FAILED, "[Get][thread_dim] failed, Invalid thread dim: %u!", thread_dim_);

  sub_task_desc.subTaskType = static_cast<rtFftsSubTaskType_t>(sub_task_def.sub_task_type());
  sub_task_desc.threadDim = sub_task_def.thread_dim();

  sub_task_desc.dstTickCacheVldBitmap = sub_task_def.dst_tick_cache_vld_bitmap();
  sub_task_desc.srcTickCacheVldBitmap = sub_task_def.src_tick_cache_vld_bitmap();
  sub_task_desc.srcDataOutOfSubGraphBitmap = sub_task_def.src_data_out_of_subgraph_bitmap();

  for (int idx = 0; idx < sub_task_def.dst_tick_cache_id_size(); ++idx) {
    sub_task_desc.dstTickCacheID[idx] = sub_task_def.dst_tick_cache_id(idx);
  }

  for (int idx = 0; idx < sub_task_def.src_tick_cache_id_size(); ++idx) {
    sub_task_desc.srcTickCacheID[idx] = sub_task_def.src_tick_cache_id(idx);
  }

  if (sub_task_def.has_auto_thread_aic_aiv()) {
    GE_CHK_STATUS_RET_NOLOG(InitAutoAicAiv(sub_task_def.auto_thread_aic_aiv(), sub_task_desc.custom.autoThreadAicAiv));
  }

  if (sub_task_def.has_manual_thread_aic_aiv()) {
    GE_CHK_STATUS_RET_NOLOG(
        InitManualAicAiv(sub_task_def.manual_thread_aic_aiv(), sub_task_desc.custom.manualThreadAicAiv));
  }

  if (sub_task_def.has_manual_thread_nop()) {
    GE_CHK_STATUS_RET_NOLOG(InitManualNop(sub_task_def.manual_thread_nop(), sub_task_desc.custom.manualThreadNop));
  }

  return SUCCESS;
}

Status FftsTaskInfo::InitTicketCache(const domi::TicketCacheDef &ticket_cache_def, rtTicketCache_t &ticket_cache) {
  if (ticket_cache_def.has_auto_thread_cache() == ticket_cache_def.has_manual_thread_cache()) {
    GELOGE(FAILED, "[Check][Param] Invalid TicketCacheDef, has auto thread cache: %d, has manual thread cache: %d",
           ticket_cache_def.has_auto_thread_cache(), ticket_cache_def.has_manual_thread_cache());
    return FAILED;
  }

  ticket_cache.cacheOption = static_cast<rtCacheOp_t>(ticket_cache_def.cache_option());
  ticket_cache.ticketCacheWindow = ticket_cache_def.ticket_cache_window();

  if (ticket_cache_def.has_auto_thread_cache()) {
    InitAutoCacheInfo(ticket_cache_def.auto_thread_cache(), ticket_cache.custom.autoThreadCache);
  }
  if (ticket_cache_def.has_manual_thread_cache()) {
    GE_CHK_STATUS_RET_NOLOG(
        InitManualCacheInfo(ticket_cache_def.manual_thread_cache(), ticket_cache.custom.manualThreadCache));
  }

  return SUCCESS;
}

// task_addr = {0,200,700,1000,2000, 3500}
// task_addr_offset = {20,40,2,100,200}
template <typename T>
Status FftsTaskInfo::InitIoAddrs(const RuntimeParam &rts_param, const T &aic_aiv_def, uint32_t thread_dim,
                                 uint32_t addr_count) {
  for (uint32_t i = 0; i < addr_count; ++i) {
    uintptr_t logic_addr = aic_aiv_def.task_addr(i)  + thread_dim * aic_aiv_def.task_addr_offset(i);
    uint8_t *io_addr = nullptr;
    if (ModelUtils::GetRtAddress(rts_param, logic_addr, io_addr) != SUCCESS) {
      GELOGE(INTERNAL_ERROR, "[Check][GetRtAddress]GetRtAddress failed.");
      return INTERNAL_ERROR;
    }
    GELOGD("aic_aiv_def task base addr is %ld, offset is %ld, thread is %d, logic addrs is 0x%lx, io addr is %p",
           aic_aiv_def.task_addr(i), aic_aiv_def.task_addr_offset(i), thread_dim, logic_addr, io_addr);
    io_addrs_.emplace_back(io_addr);
  }
  return SUCCESS;
}

Status FftsTaskInfo::InitAutoAicAiv(const domi::AutoThreadAicAivDef &aic_aiv_def, rtAutoThreadAicAivInfo_t &aic_aiv) {
  if (aic_aiv_def.src_prefetch_size() > static_cast<int>(RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK)) {
    GELOGE(FAILED, "[Check][Param] Invalid AutoThreadAicAivInfo, prefetch size: %d", aic_aiv_def.src_prefetch_size());
    return FAILED;
  }

  aic_aiv.taskParamAddr = reinterpret_cast<uintptr_t>(args_) + kAddrLen * io_addrs_.size();
  GELOGD("AutoThreadAicAivDef: task param addr is %lu.", aic_aiv.taskParamAddr);
  const auto &rts_param = davinci_model_->GetRuntimeParam();
  for (uint32_t i = 0; i < thread_dim_ - 1; ++i) {
    GE_CHK_STATUS_RET_NOLOG(InitIoAddrs(rts_param, aic_aiv_def, i,
                                        static_cast<uint32_t>(aic_aiv_def.task_addr_offset_size())));
  }
  GE_CHK_STATUS_RET_NOLOG(InitIoAddrs(rts_param, aic_aiv_def, thread_dim_ - 1, aic_aiv_def.input_output_count()));
  int last_thread_workspace_size = aic_aiv_def.task_addr_size() - aic_aiv_def.task_addr_offset_size();
  for (int k = 0; k < last_thread_workspace_size; ++k) {
    uintptr_t logic_addr = aic_aiv_def.task_addr(aic_aiv_def.task_addr_offset_size() + k);
    uint8_t *io_addr = nullptr;
    GE_CHK_STATUS_RET_NOLOG(ModelUtils::GetRtAddress(rts_param, logic_addr, io_addr));
    GELOGD("logic addr is 0x%lx, io addr is %p.", logic_addr, io_addr);
    io_addrs_.emplace_back(io_addr);
  }

  aic_aiv.taskParamOffset = aic_aiv_def.task_param_offset();
  GELOGD("args_: %p, io_addrs size: %zu, task param offset: %u.", args_, io_addrs_.size(), aic_aiv.taskParamOffset);
  aic_aiv.satMode = aic_aiv_def.sat_mode();
  aic_aiv.scheduleMode = aic_aiv_def.schedule_mode();
  aic_aiv.iCachePrefetchCnt = aic_aiv_def.cache_prefetch_cnt();

  aic_aiv.prefetchEnableBitmap = aic_aiv_def.prefetch_enable_bitmap();
  aic_aiv.prefetchOnceBitmap = aic_aiv_def.prefetch_once_bitmap();

  aic_aiv.tailBlkDim = aic_aiv_def.tail_blk_dim();
  aic_aiv.nonTailBlkDim = aic_aiv_def.non_tail_blk_dim();

  aic_aiv.nonTailTaskFuncStub = davinci_model_->GetRegisterStub(aic_aiv_def.non_tail_task_func_stub(), "");
  aic_aiv.tailTaskFuncStub = davinci_model_->GetRegisterStub(aic_aiv_def.tail_task_func_stub(), "");

  GELOGI("Set func name[%s][%s] succ.", aic_aiv.nonTailTaskFuncStub, aic_aiv.tailTaskFuncStub);
  for (int idx = 0; idx < aic_aiv_def.src_prefetch_size(); ++idx) {
    InitAutoPrefetch(aic_aiv_def.src_prefetch(idx), aic_aiv.srcPrefetch[idx]);
  }

  return SUCCESS;
}

void FftsTaskInfo::InitAutoCacheInfo(const domi::AutoThreadCacheDef &cache_def, rtAutoThreadCacheInfo_t &cache) {
  cache.dataAddr = cache_def.data_addr();
  cache.dataAddrOffset = cache_def.data_addr_offset();
  cache.nonTailDataLen = cache_def.non_tail_data_len();
  cache.tailDataLen = cache_def.tail_data_len();
  cache.ticketCacheRefCnt = cache_def.ticket_cache_ref_cnt();
}

void FftsTaskInfo::InitAutoPrefetch(const domi::AutoThreadPrefetchDef &prefetch_def, rtAutoThreadPrefetch_t &prefetch) {
  prefetch.dataAddr = prefetch_def.data_addr();
  prefetch.dataAddrOffset = prefetch_def.data_addr_offset();
  prefetch.nonTailDataLen = prefetch_def.non_tail_data_len();
  prefetch.tailDataLen = prefetch_def.tail_data_len();
}

Status FftsTaskInfo::InitManualAicAiv(const domi::ManualThreadAicAivDef &aic_aiv_def,
                                      rtManualThreadAicAivInfo_t &aic_aiv) {
  if ((aic_aiv_def.thread_prefetch_dmu_idx_size() > static_cast<int>(RT_FFTS_MAX_MANUAL_THREAD_NUM)) ||
      (aic_aiv_def.thread_blk_dim_size() > static_cast<int>(RT_FFTS_MAX_MANUAL_THREAD_NUM)) ||
      (aic_aiv_def.thread_task_func_stub_size() > static_cast<int>(RT_FFTS_MAX_MANUAL_THREAD_NUM)) ||
      (aic_aiv_def.src_dep_tbl_size() > static_cast<int>(RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK))) {
    GELOGE(FAILED, "[Check][Param] Invalid ManualThreadAicAivInfo, thread prefetch dmu desc size: %d, "
           "thread blk dim size: %d, thread task func stub size: %d, src dep tbl size: %d",
           aic_aiv_def.thread_prefetch_dmu_idx_size(), aic_aiv_def.thread_blk_dim_size(),
           aic_aiv_def.thread_task_func_stub_size(), aic_aiv_def.src_dep_tbl_size());
    return FAILED;
  }
  aic_aiv.taskParamAddr = reinterpret_cast<uintptr_t>(args_) + kAddrLen * io_addrs_.size();
  GELOGD("ManualThreadAicAivDef: task param addr is %lu.", aic_aiv.taskParamAddr);
  const auto &rts_param = davinci_model_->GetRuntimeParam();
  for (uint32_t i = 0; i < thread_dim_ - 1; ++i) {
    GE_CHK_STATUS_RET_NOLOG(InitIoAddrs(rts_param, aic_aiv_def, i,
                                        static_cast<uint32_t>(aic_aiv_def.task_addr_offset_size())));
  }
  GE_CHK_STATUS_RET_NOLOG(InitIoAddrs(rts_param, aic_aiv_def, thread_dim_ - 1, aic_aiv_def.input_output_count()));
  int last_thread_workspace_size = aic_aiv_def.task_addr_size() - aic_aiv_def.task_addr_offset_size();
  for (int k = 0; k < last_thread_workspace_size; ++k) {
    uintptr_t logic_addr = aic_aiv_def.task_addr(aic_aiv_def.task_addr_offset_size() + k);
    uint8_t *io_addr = nullptr;
    GE_CHK_STATUS_RET_NOLOG(ModelUtils::GetRtAddress(rts_param, logic_addr, io_addr));
    io_addrs_.emplace_back(io_addr);
  }
  aic_aiv.taskParamOffset = aic_aiv_def.task_param_offset();

  aic_aiv.satMode = aic_aiv_def.sat_mode();
  aic_aiv.scheduleMode = aic_aiv_def.schedule_mode();
  aic_aiv.iCachePrefetchCnt = aic_aiv_def.cache_prefetch_cnt();

  aic_aiv.prefetchEnableBitmap = aic_aiv_def.prefetch_enable_bitmap();    // 8 bit bitmap 1 0 1 0
  aic_aiv.prefetchOnceBitmap = aic_aiv_def.prefetch_once_bitmap();   // 8 bit bitmap 1 0 1 0
  aic_aiv.prefetchOnceDmuNum = aic_aiv_def.prefetch_once_dmu_num();

  for (int idx = 0; idx < aic_aiv_def.thread_prefetch_dmu_idx_size(); ++idx) {
    aic_aiv.threadPrefetchDmuIdx[idx] = aic_aiv_def.thread_prefetch_dmu_idx(idx);
  }
  for (int idx = 0; idx < aic_aiv_def.thread_blk_dim_size(); ++idx) {
    aic_aiv.threadBlkDim[idx] = aic_aiv_def.thread_blk_dim(idx);
  }
  for (int idx = 0; idx < aic_aiv_def.thread_task_func_stub_size(); ++idx) {
    aic_aiv.threadTaskFuncStub[idx] = aic_aiv_def.thread_task_func_stub(idx).c_str();
  }

  InitManualDmuInfo(aic_aiv_def, aic_aiv.prefetchList);
  for (int idx = 0; idx < aic_aiv_def.src_dep_tbl_size(); ++idx) {
    GE_CHK_STATUS_RET_NOLOG(InitManualDependency(aic_aiv_def.src_dep_tbl(idx), aic_aiv.srcDepTbl[idx]));
  }

  return SUCCESS;
}

Status FftsTaskInfo::InitManualCacheInfo(const domi::ManualThreadCacheDef &cache_def,
                                         rtManualThreadCacheInfo_t &cache_info) {
  if ((cache_def.slice_dmu_idx_size() > static_cast<int>(RT_FFTS_MAX_MANUAL_THREAD_NUM)) ||
      (cache_def.ticket_cache_ref_cnt_tbl_size() > static_cast<int>(RT_FFTS_MAX_MANUAL_THREAD_NUM))) {
    GELOGE(FAILED, "[Check][Param] Invalid ManualThreadCacheInfo slice dum desc index %d, ticket cache ref cnt %d",
           cache_def.slice_dmu_idx_size(), cache_def.ticket_cache_ref_cnt_tbl_size());
    return FAILED;
  }

  InitManualDmuInfo(cache_def, cache_info.dmuList);
  for (int idx = 0; idx < cache_def.slice_dmu_idx_size(); ++idx) {
    cache_info.sliceDmuIdx[idx] = cache_def.slice_dmu_idx(idx);
  }

  for (int idx = 0; idx < cache_def.ticket_cache_ref_cnt_tbl_size(); ++idx) {
    cache_info.ticketCacheRefCntTbl[idx] = cache_def.ticket_cache_ref_cnt_tbl(idx);
  }

  return SUCCESS;
}

Status FftsTaskInfo::InitManualDependency(const domi::ManualThreadDependencyDef &dependency_def,
                                          rtManualThreadDependency_t &dependency) {
  if (dependency_def.dependency_size() > static_cast<int>(RT_FFTS_MANUAL_SRC_DEPEND_TBL_LEN)) {
    GELOGE(FAILED, "[Check][Param] Invalid ManualThreadDependency size: %d", dependency_def.dependency_size());
    return FAILED;
  }

  for (int idx = 0; idx < dependency_def.dependency_size(); ++idx) {
    dependency.dependency[idx] = dependency_def.dependency(idx);
  }

  return SUCCESS;
}

Status FftsTaskInfo::InitManualNop(const domi::ManualThreadNopDef &nop_def, rtManualThreadNopInfo_t &nop_info) {
  if (nop_def.src_dep_tbl_size() > static_cast<int>(RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK)) {
    GELOGE(FAILED, "[Check][Param] Invalid ManualThreadNopInfo, src dep tbl size: %d", nop_def.src_dep_tbl_size());
    return FAILED;
  }

  for (int idx = 0; idx < nop_def.src_dep_tbl_size(); ++idx) {
    GE_CHK_STATUS_RET_NOLOG(InitManualDependency(nop_def.src_dep_tbl(idx), nop_info.srcDepTbl[idx]));
  }

  return SUCCESS;
}

void FftsTaskInfo::InitManualDmuInfo(const domi::ManualThreadAicAivDef &aic_aiv_def, rtManualThreadDmuInfo_t *&dmu) {
  if (aic_aiv_def.prefetch_list().empty()) {
    return;
  }

  std::vector<uint8_t> buffer(sizeof(rtManualThreadDmuInfo_t) * aic_aiv_def.prefetch_list_size());
  dmu = reinterpret_cast<rtManualThreadDmuInfo_t *>(buffer.data());
  for (int idx = 0; idx < aic_aiv_def.prefetch_list_size(); ++idx) {
    InitManualDmuInfo(aic_aiv_def.prefetch_list(idx), dmu[idx]);
  }
}

void FftsTaskInfo::InitManualDmuInfo(const domi::ManualThreadCacheDef &cache_def, rtManualThreadDmuInfo_t *&dmu) {
  if (cache_def.dmu_list().empty()) {
    return;
  }

  std::vector<uint8_t> buffer(sizeof(rtManualThreadDmuInfo_t) * cache_def.dmu_list_size());
  dmu = reinterpret_cast<rtManualThreadDmuInfo_t *>(buffer.data());
  for (int idx = 0; idx < cache_def.dmu_list_size(); ++idx) {
    InitManualDmuInfo(cache_def.dmu_list(idx), dmu[idx]);
  }
}

void FftsTaskInfo::InitManualDmuInfo(const domi::ManualThreadDmuDef &dmu_def, rtManualThreadDmuInfo_t &dmu) {
  dmu.dataAddr = dmu_def.data_addr();
  dmu.numOuter = dmu_def.num_outer();
  dmu.numInner = dmu_def.num_inner();
  dmu.strideOuter = dmu_def.stride_outer();
  dmu.lenInner = dmu_def.len_inner();
  dmu.strideInner = dmu_def.stride_inner();
}

Status FftsTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  return SUCCESS;
}

Status FftsTaskInfo::UpdateArgs() {
  GE_CHECK_NOTNULL(davinci_model_);
  std::vector<void *> io_addrs = io_addrs_;
  davinci_model_->UpdateKnownZeroCopyAddr(io_addrs);
  auto addr_size = kAddrLen * io_addrs.size();
  GE_CHK_RT_RET(rtMemcpy(args_, args_size_, io_addrs.data(), addr_size, RT_MEMCPY_HOST_TO_DEVICE));
  return SUCCESS;
}

Status FftsTaskInfo::Distribute() {
  GELOGI("FftsTaskInfo Distribute Start.");
  rtError_t rt_ret = rtFftsTaskLaunch(&sub_task_info_, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    GELOGE(RT_FAILED, "[Check][RT_ret] Call rtFftsTaskLaunch failed, ret: 0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  GELOGI("FftsTaskInfo Distribute Success.");
  return SUCCESS;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_FFTS_TASK, FftsTaskInfo);
}  // namespace ge
