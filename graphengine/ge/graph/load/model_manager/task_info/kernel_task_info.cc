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

#include "graph/load/model_manager/task_info/kernel_task_info.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "aicpu/common/aicpu_task_struct.h"
#include "common/ge/plugin_manager.h"
#include "common/properties_manager.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/l2_cache_optimize.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/model_utils.h"
#include "runtime/rt.h"
#include "graph/load/model_manager/task_info/super_kernel/super_kernel.h"
#include "graph/load/model_manager/task_info/super_kernel/super_kernel_factory.h"
#include "cce/aicpu_engine_struct.h"
#include "framework/common/debug/log.h"

namespace {
const uint8_t kL2LoadToDdr = 1;
const uint8_t kL2NotLoadToDdr = 0;
// for skt
constexpr int64_t kInvalidGroupKey = -1;
constexpr uint32_t kSKTSingleSize = 1;
const char *kIsLastNode = "is_last_node";
const char *kIsFirstNode = "is_first_node";
const char *const kAicpuAllshape = "_AllShape";
const int64_t kCloseSkt = 100;
const uint32_t kAddrLen = sizeof(void *);
const int kBaseInt = 10;
const int kStrtolFail = 0;
const int kArgsInputDesc = 0;
const int kArgsInputAddr = 1;
const int kArgsOutputDesc = 2;
const int kArgsOutputAddr = 3;
const int kArgsAttrHandle = 4;
}  // namespace

namespace ge {
Status KernelTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  is_l1_fusion_enable_ = davinci_model_->GetL1FusionEnableOption();
  GELOGD("KernelTaskInfo init start, ge.enableL1Fusion in davinci model is %d.", is_l1_fusion_enable_);

  Status ret = SetStream(task_def.stream_id(), davinci_model_->GetStreamList());
  if (ret != SUCCESS) {
    return ret;
  }

  const domi::KernelDef &kernel_def = task_def.kernel();
  block_dim_ = kernel_def.block_dim();
  args_size_ = kernel_def.args_size();
  // get opcontext stored in model
  const domi::KernelContext &context = kernel_def.context();
  // get kernel_type
  kernel_type_ = static_cast<ccKernelType>(context.kernel_type());
  // get opdesc
  op_desc_ = davinci_model_->GetOpByIndex(context.op_index());
  GE_CHECK_NOTNULL(op_desc_);
  (void)AttrUtils::GetBool(*op_desc_, ATTR_N_BATCH_SPILT, is_n_batch_spilt_);
  GELOGD("node[%s] is_n_batch_spilt %d", op_desc_->GetName().c_str(), is_n_batch_spilt_);
  (void)AttrUtils::GetInt(*op_desc_, ATTR_NAME_FUSION_GROUP_KEY, group_key_);
  has_group_key_ = (group_key_ != kInvalidGroupKey);
  GELOGD("node[%s] has_group_key_ %d, group key is [%ld]", op_desc_->GetName().c_str(), has_group_key_, group_key_);

  // fusion_op_info
  vector<std::string> original_op_names;
  bool result = AttrUtils::GetListStr(op_desc_, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);
  GE_IF_BOOL_EXEC(result, fusion_op_info_.stream_id = task_def.stream_id();
                  fusion_op_info_.op_index = context.op_index(); fusion_op_info_.original_op_names = original_op_names;
                  fusion_op_info_.op_name = op_desc_->GetName());

  // new aicpu kernel(rtCpuKernelLaunch) no need to check function
  if (kernel_type_ == ccKernelType::CCE_AI_CORE) {
    rtError_t rt_ret = rtGetFunctionByName(const_cast<char *>(kernel_def.stub_func().c_str()), &stub_func_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    REPORT_CALL_ERROR("E19999", "Call rtGetFunctionByName failed for op:%s(%s), "
                                      "bin_file_key:%s, ret:0x%X", op_desc_->GetName().c_str(),
                                      op_desc_->GetType().c_str(), kernel_def.stub_func().c_str(), rt_ret);
                    GELOGE(RT_FAILED, "[Execute][RtGetFunctionByName] failed for op:%s(%s). stub_func:%s",
                           op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), kernel_def.stub_func().c_str());
                    return RT_ERROR_TO_GE_STATUS(rt_ret););
  } else if (kernel_type_ == ccKernelType::TE) {
    // get bin_file_key
    string session_graph_model_id;
    davinci_model_->GetUniqueId(op_desc_, session_graph_model_id);
    const char *bin_file_key = davinci_model_->GetRegisterStub(op_desc_->GetName(), session_graph_model_id);
    rtError_t rt_ret = rtGetFunctionByName(bin_file_key, &stub_func_);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    REPORT_CALL_ERROR("E19999", "Call rtGetFunctionByName failed for op:%s(%s), "
                                      "bin_file_key:%s, ret:0x%X", op_desc_->GetName().c_str(),
                                      op_desc_->GetType().c_str(), bin_file_key, rt_ret);
                    GELOGE(RT_FAILED, "[Execute][RtGetFunctionByName] failed for op:%s(%s), bin_file_key:%s",
                           op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), bin_file_key);
                    return RT_ERROR_TO_GE_STATUS(rt_ret););
  }

  if (context.origin_op_index_size() > CC_FUSION_OP_MAX) {
    REPORT_INNER_ERROR("E19999", "context.origin_op_index_size():%d is more than CC_FUSION_OP_MAX(%d), op:%s(%s), "
                       "check invalid", context.origin_op_index_size(), CC_FUSION_OP_MAX,
                       op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] context.origin_op_index_size():%d is more than CC_FUSION_OP_MAX(%d), "
           "op:%s(%s)", context.origin_op_index_size(), CC_FUSION_OP_MAX,
           op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return PARAM_INVALID;
  }

  for (int32_t i = 0; i < context.origin_op_index_size(); ++i) {
    ctx_.opIndex2[i] = context.origin_op_index(i);
  }
  ctx_.opCount = context.origin_op_index_size();
  InitDumpFlag();
  if (kernel_type_ == ccKernelType::TE) {
    ctx_.opIndex = context.op_index();
    uint16_t *args_offset_tmp = reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data()));
    if (context.args_offset().size() / sizeof(uint16_t) < 1) {
      REPORT_INNER_ERROR("E19999", "context.args_offset().size():%zu / sizeof(uint16_t) less than 1, op:%s(%s), "
                         "check invalid", context.args_offset().size(),
                         op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] context.args_offset().size() / sizeof(uint16_t) less than 1, op:%s(%s)",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return FAILED;
    }

    io_addr_offset_ = args_offset_tmp[0];
    ret = InitTVMTask(io_addr_offset_, kernel_def);
  } else if (kernel_type_ == ccKernelType::CUSTOMIZED) {
    ret = InitAICPUCustomTask(context.op_index(), kernel_def);
  } else if (kernel_type_ == ccKernelType::AI_CPU || kernel_type_ == ccKernelType::CUST_AI_CPU) {
    ret = InitAicpuTask(context.op_index(), kernel_def);
  } else {
    if (kernel_def.args().empty() || args_size_ == 0) {
      REPORT_INNER_ERROR("E19999", "kernel_def.args() is empty, op:%s(%s), check invalid",
                         op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] args is empty, op:%s(%s)",
             op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
      return FAILED;
    }
    ret = InitCceTask(kernel_def);
  }

  SetIoAddrs(op_desc_);
  GELOGD("KernelTaskInfo init finish, result=%u.", ret);
  return ret;
}

Status KernelTaskInfo::SaveSKTDumpInfo() {
  GE_CHECK_NOTNULL(davinci_model_);
  if (skt_dump_flag_ == RT_KERNEL_DEFAULT) {
    GELOGD("no need save skt dump info");
    return SUCCESS;
  }
  // all op in super kernel share one taskid and streamid
  const SuperKernelTaskInfo &skt_info = davinci_model_->GetSuperKernelTaskInfo();
  for (size_t i = 0; i < skt_info.op_desc_list.size(); i++) {
    davinci_model_->SaveDumpTask(skt_info.last_task_id, skt_info.last_stream_id, skt_info.op_desc_list[i],
                                 skt_info.dump_args_list[i]);
  }
  return SUCCESS;
}

void KernelTaskInfo::UpdateSKTTaskId() {
  uint32_t task_id = 0;
  uint32_t stream_id = 0;
  if (davinci_model_ != nullptr) {
    rtError_t rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &task_id, &stream_id);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtModelGetTaskId failed, ret:0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtModelGetTaskId] failed, ret:0x%X", rt_ret);
      return;
    }
    SuperKernelTaskInfo &skt_info = davinci_model_->GetSuperKernelTaskInfo();
    skt_info.last_task_id = task_id;
    skt_info.last_stream_id = stream_id;
    skt_id_ = skt_info.last_task_id;

    GELOGI("UpdateTaskId:UpdateSKTTaskId [%u],stream id [%u]", task_id, stream_id);
  }
}

void KernelTaskInfo::UpdateTaskId() {
  uint32_t task_id = 0;
  uint32_t stream_id = 0;  //  for profiling
  if (davinci_model_ != nullptr) {
    rtError_t rt_ret = rtModelGetTaskId(davinci_model_->GetRtModelHandle(), &task_id, &stream_id);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtModelGetTaskId failed, ret:0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtModelGetTaskId] failed, ret:0x%X", rt_ret);
      return;
    }
    task_id_ = task_id;
    stream_id_ = stream_id;
    GELOGD("UpdateTaskId:UpdateTaskId [%u], stream id [%u]:", task_id, stream_id);
  }
}

Status KernelTaskInfo::SKTFinalize() {
  UpdateSKTTaskId();
  GE_CHK_STATUS_RET(SaveSKTDumpInfo(), "[Save][SKTDumpInfo] failed");
  GELOGI("SuperKernel Distribute [skt_id:%u]", skt_id_);
  SuperKernelTaskInfo &skt_info = davinci_model_->GetSuperKernelTaskInfo();
  skt_info.kernel_list.clear();
  skt_info.arg_list.clear();
  skt_info.dump_flag_list.clear();
  skt_info.op_desc_list.clear();
  skt_info.dump_args_list.clear();
  skt_info.last_stream = nullptr;
  skt_info.last_block_dim = 0;
  skt_info.last_sm_desc = sm_desc_;
  skt_info.last_group_key = kInvalidGroupKey;
  skt_info.last_dump_flag = RT_KERNEL_DEFAULT;
  skt_info.last_dump_args = 0;
  skt_info.last_op = nullptr;
  return SUCCESS;
}

uint32_t KernelTaskInfo::GetDumpFlag() {
  const SuperKernelTaskInfo &skt_info = davinci_model_->GetSuperKernelTaskInfo();
  for (auto flag : skt_info.dump_flag_list) {
    if (flag == RT_KERNEL_DUMPFLAG) {
      return RT_KERNEL_DUMPFLAG;
    }
  }
  return RT_KERNEL_DEFAULT;
}

Status KernelTaskInfo::SuperKernelLaunch() {
  const SuperKernelTaskInfo &skt_info = davinci_model_->GetSuperKernelTaskInfo();
  if (skt_info.kernel_list.empty()) {
    GELOGI("SuperKernelLaunch: Skt_kernel_list has no task, just return");
    return SUCCESS;
  }
  rtError_t rt_ret;
  auto &skt_kernel_list = skt_info.kernel_list;
  auto &skt_arg_list = skt_info.arg_list;
  GELOGI("SuperKernelLaunch: Skt_kernel_list size[%zu] skt_arg_list[%zu]", skt_kernel_list.size(), skt_arg_list.size());
  if (skt_kernel_list.size() == kSKTSingleSize && skt_arg_list.size() == kSKTSingleSize) {
    rt_ret = rtKernelLaunchWithFlag(skt_info.kernel_list[0], static_cast<uint32_t>(skt_info.last_block_dim),
                                    skt_info.arg_list[0], skt_info.last_args_size,
                                    static_cast<rtSmDesc_t *>(skt_info.last_sm_desc), skt_info.last_stream,
                                    skt_info.last_dump_flag);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtKernelLaunchWithFlag failed, ret:0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtKernelLaunchWithFlag] failed, ret:0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    call_save_dump_ = true;
    GE_CHK_STATUS_RET(SKTFinalize(), "[Call][SKTFinalize] failed");
    return SUCCESS;
  }
  // Create super kernel factory
  skt::SuperKernelFactory *factory = &skt::SuperKernelFactory::GetInstance();
  // Init super kernel factory
  Status ge_ret = factory->Init();
  if (ge_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call SuperKernelFactory init fail, ret:0x%X", ge_ret);
    GELOGE(ge_ret, "[Init][SuperKernelFactory] failed, ret:0x%X", ge_ret);
    return ge_ret;
  }
  // Call the fuse API
  std::unique_ptr<skt::SuperKernel> superKernel = nullptr;
  ge_ret = factory->FuseKernels(skt_kernel_list, skt_arg_list, skt_info.last_block_dim, superKernel);
  if (ge_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call SuperKernelFactory FuseKernels fail, ret:0x%X", ge_ret);
    GELOGE(ge_ret, "[Call][FuseKernels] failed, ret:0x%X", ge_ret);
    return ge_ret;
  }
  // Launch a super kernel
  skt_dump_flag_ = GetDumpFlag();
  ge_ret = superKernel->Launch(skt_info.last_stream, skt_dump_flag_);
  if (ge_ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call SuperKernelFactory Launch fail, ret:0x%X", ge_ret);
    GELOGE(ge_ret, "[Call][Launch] failed, ret:0x%X", ge_ret);
    return ge_ret;
  }
  GELOGI("SuperKernelLaunch: success[skt_kernel_list size[%zu] skt_arg_list[%zu]]", skt_kernel_list.size(),
         skt_arg_list.size());
  // record skt addr for release
  superkernel_dev_nav_table_ = superKernel->GetNavTablePtr();
  superkernel_device_args_addr_ = superKernel->GetDeviceArgsPtr();
  GE_CHK_STATUS_RET(SKTFinalize(), "[Call][SKTFinalize] failed");
  return SUCCESS;
}

Status KernelTaskInfo::SaveSuperKernelInfo() {
  SuperKernelTaskInfo &skt_info = davinci_model_->GetSuperKernelTaskInfo();
  skt_info.kernel_list.push_back(stub_func_);
  skt_info.arg_list.push_back(args_);
  skt_info.last_stream = stream_;
  skt_info.last_block_dim = block_dim_;
  skt_info.last_args_size = args_size_;
  skt_info.last_sm_desc = sm_desc_;
  skt_info.last_dump_flag = dump_flag_;
  skt_info.dump_flag_list.push_back(dump_flag_);
  skt_info.op_desc_list.push_back(op_desc_);
  skt_info.dump_args_list.push_back(reinterpret_cast<uintptr_t>(skt_dump_args_));
  skt_info.last_group_key = group_key_;
  skt_info.last_dump_args = reinterpret_cast<uintptr_t>(skt_dump_args_);
  skt_info.last_op = op_desc_;
  // last node in a stream, just launch
  if (IsMarkedLastNode()) {
    return SuperKernelLaunch();
  }

  GELOGI("Save Current task [block_dim:%u, size:%zu].", block_dim_, skt_info.kernel_list.size());
  return SUCCESS;
}

bool KernelTaskInfo::IsMarkedLastNode() {
  if (davinci_model_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr");
    GELOGE(PARAM_INVALID, "[Check][Param] davinci_model is null!");
    return false;
  }
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(ctx_.opIndex);
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", ctx_.opIndex);
    GELOGE(INTERNAL_ERROR, "[Get][Op] by index failed, index:%u is out of range!", ctx_.opIndex);
    return false;
  }
  bool is_last_node = false;
  (void)AttrUtils::GetBool(*op_desc, kIsLastNode, is_last_node);
  return is_last_node;
}

bool KernelTaskInfo::IsMarkedFirstNode() {
  if (davinci_model_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr");
    GELOGE(PARAM_INVALID, "[Check][Param] davinci_model is null!");
    return false;
  }
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(ctx_.opIndex);
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", ctx_.opIndex);
    GELOGE(INTERNAL_ERROR, "[Get][Op] by index failed, index:%u is out of range!", ctx_.opIndex);
    return false;
  }
  bool is_first_node = false;
  (void)AttrUtils::GetBool(*op_desc, kIsFirstNode, is_first_node);
  return is_first_node;
}
// current task 's block dim and stream and grouo key (if have) must same with last task,
// then may be saved to skt task list; else
// call skt launch those saved tasks before
bool KernelTaskInfo::FirstCallSKTLaunchCheck() {
  const SuperKernelTaskInfo &skt_info = davinci_model_->GetSuperKernelTaskInfo();
  return ((block_dim_ != skt_info.last_block_dim) || (stream_ != skt_info.last_stream) ||
          (has_group_key_ && (group_key_ != skt_info.last_group_key)));
}

// current task has group_id or has n ATTR_N_BATCH_SPLIT then save it to skt task list; else
// call skt launch those saved tasks and call rtlaunch for current task
bool KernelTaskInfo::DoubleCallSKTSaveCheck() { return (!is_n_batch_spilt_ && !has_group_key_); }

Status KernelTaskInfo::SuperKernelDistribute() {
  Status ret;
  if (FirstCallSKTLaunchCheck()) {
    ret = SuperKernelLaunch();
    if (ret != SUCCESS) {
      GELOGE(FAILED, "[Call][SuperKernelLaunch] failed, taskid:%u", task_id_);
      return FAILED;
    }
  }
  if (DoubleCallSKTSaveCheck()) {
    // 1.launch before
    ret = SuperKernelLaunch();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][SuperKernelLaunch] failed, taskid:%u", task_id_);
      return ret;
    }
    // 2.launch current
    rtError_t rt_ret = rtKernelLaunchWithFlag(stub_func_, block_dim_, args_, args_size_,
                                              static_cast<rtSmDesc_t *>(sm_desc_), stream_, dump_flag_);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtKernelLaunchWithFlag failed, ret:0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtKernelLaunchWithFlag] failed, ret:0x%X", rt_ret);
      return rt_ret;
    }
    call_save_dump_ = true;
    UpdateTaskId();
    GELOGI("Current Common Task Distribute [taskid:%u]", task_id_);
  } else {
    ret = SaveSuperKernelInfo();
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][SaveSuperKernelInfo] failed, taskid:%u", task_id_);
      return ret;
    }
  }
  return SUCCESS;
}

void KernelTaskInfo::SetArgs() {
  if (davinci_model_->IsKnownNode()) {
    if (kernel_type_ == ccKernelType::TE) {
      args_ = l2_buffer_on_ ? davinci_model_->GetCurrentHybridArgsAddr(hybrid_args_offset_)
                            : davinci_model_->GetCurrentArgsAddr(args_offset_);
    } else if (kernel_type_ == ccKernelType::AI_CPU || kernel_type_ == ccKernelType::CUST_AI_CPU) {
      args_ = davinci_model_->GetCurrentHybridArgsAddr(hybrid_args_offset_);
    }
    GELOGI("Known node %s args addr %p, offset %u.", op_desc_->GetName().c_str(), args_, args_offset_);
  }
}

Status KernelTaskInfo::Distribute() {
  GELOGD("KernelTaskInfo Distribute Start.");
  SetArgs();
  rtError_t rt_ret = RT_ERROR_NONE;
  char skt_enable_env[MMPA_MAX_PATH] = { 0x00 };
  INT32 res = mmGetEnv("SKT_ENABLE", skt_enable_env, MMPA_MAX_PATH);
  int64_t env_flag = (res == EN_OK) ? strtol(skt_enable_env, nullptr, kBaseInt) : kStrtolFail;
  bool call_skt = ((env_flag != 0) || is_l1_fusion_enable_);
  if (kernel_type_ == ccKernelType::AI_CPU || kernel_type_ == ccKernelType::CUST_AI_CPU) {
    if (topic_type_flag_ > 0) {
      // Use the fifth and sixth bits of dump_flag_ indicate the value of topic_type.
      // xxxxxxxx xxxxxxxx xxxxxxxx xx00xxxx: DEVICE_ONLY
      // xxxxxxxx xxxxxxxx xxxxxxxx xx01xxxx: DEVICE_FIRST
      // xxxxxxxx xxxxxxxx xxxxxxxx xx10xxxx: HOST_ONLY
      // xxxxxxxx xxxxxxxx xxxxxxxx xx11xxxx: HOST_FIRST
      dump_flag_ = dump_flag_ | static_cast<uint32_t>(topic_type_flag_);
    }
    GELOGI("distribute task info kernel_type %d, flag %d", kernel_type_, dump_flag_);
    // blockDim is reserved parameter, set to 1
    std::string op_name = op_desc_->GetName();
    rtKernelLaunchNames_t launch_name = {so_name_.c_str(), kernel_name_.c_str(), op_name.c_str()};
    rt_ret = rtAicpuKernelLaunchWithFlag(&launch_name, 1, args_, args_size_,
                                         nullptr, stream_, dump_flag_);
    call_save_dump_ = true;
  } else {
    /* default: not skt launch */
    const SuperKernelTaskInfo &skt_info = davinci_model_->GetSuperKernelTaskInfo();
    GELOGD(
        "KernelTaskInfo Distribute Start, sktenable:%d taskid:%u sktid:%u last_sktid:%u stubfunc_name:%s "
        "stubfunc:%p blockdim:%u stream:%p",
        call_skt, task_id_, skt_id_, skt_info.last_task_id, stub_func_name_.c_str(), stub_func_, block_dim_, stream_);
    // l1 fusion enable and env flag open (kCloseSkt for skt debug)
    bool open_dump = false;
    if (davinci_model_->ModelNeedDump()) {
      open_dump = true;
    }
    if (call_skt && (env_flag != kCloseSkt) && !open_dump) {
      GE_RETURN_IF_ERROR(SuperKernelDistribute());
    } else {
      // call rtKernelLaunch for current task
      rt_ret = rtKernelLaunchWithFlag(stub_func_, block_dim_, args_, args_size_, static_cast<rtSmDesc_t *>(sm_desc_),
                                      stream_, dump_flag_);
      call_save_dump_ = true;
    }
  }
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtKernelLaunchWithFlag or rtCpuKernelLaunchWithFlag failed, "
                      "ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtApi] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  // set for task_id_
  UpdateTaskId();
  if (is_blocking_aicpu_op_) {
    if (DistributeWaitTaskForAicpuBlockingOp() != SUCCESS) {
      GELOGE(FAILED, "[Call][DistributeWaitTaskForAicpuBlockingOp] Call DistributeWaitTaskForAicpuBlockingOp failed");
      return FAILED;
    }
  }
  GELOGD(
      "KernelTaskInfo Distribute Success. sktenable:%d taskid:%d sktid:%d stubfunc_name:%s stubfunc:%p "
      "blockdim:%d stream:%p",
      call_skt, task_id_, skt_id_, stub_func_name_.c_str(), stub_func_, block_dim_, stream_);
  op_desc_.reset(); // Not hold OpDesc after distribute.
  return SUCCESS;
}

Status KernelTaskInfo::CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support) {
  int32_t device_id = 0;
  auto rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtGetDevice failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtGetDevice] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  int32_t value = 0;
  rt_ret = rtGetDeviceCapability(device_id, FEATURE_TYPE_BLOCKING_OPERATOR, RT_MODULE_TYPE_AICPU, &value);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtGetDeviceCapability failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][rtGetDeviceCapability] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if (value != RT_AICPU_BLOCKING_OP_NOT_SUPPORT && value != RT_AICPU_BLOCKING_OP_SUPPORT) {
    REPORT_INNER_ERROR("E19999", "Value should be %d or %d but %d",
                       RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    GELOGE(FAILED, "[Check][Value] Value should be %d or %d but %d",
           RT_AICPU_BLOCKING_OP_NOT_SUPPORT, RT_AICPU_BLOCKING_OP_SUPPORT, value);
    return FAILED;
  }
  is_support = (value == RT_AICPU_BLOCKING_OP_SUPPORT ? true : false);
  return SUCCESS;
}

Status KernelTaskInfo::UpdateEventIdForAicpuBlockingOp(std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> &ext_handle) {
  if (is_blocking_aicpu_op_) {
    bool is_support = false;
    if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
      GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOpProcess failed");
      return FAILED;
    }
    if (!is_support) {
      GELOGD("Device not support blocking aicpu op process");
      return SUCCESS;
    }
    uint32_t event_id = 0;
    if (davinci_model_->GetEventIdForBlockingAicpuOp(op_desc_, stream_, event_id) != SUCCESS) {
      GELOGE(FAILED, "[Get][EventId] Get event id failed for op:%s(%s)", op_desc_->GetName().c_str(),
             op_desc_->GetType().c_str());
      return FAILED;
    }
    if (ext_handle->UpdateEventId(event_id) != SUCCESS) {
      GELOGE(FAILED, "[Update][EventId] Update event id failed for op:%s(%s)", op_desc_->GetName().c_str(),
             op_desc_->GetType().c_str());
      return FAILED;
    }
    GELOGI("Update event_id=%u success", event_id);
  }
  return SUCCESS;
}

Status KernelTaskInfo::DistributeWaitTaskForAicpuBlockingOp() {
  bool is_support = false;
  if (CheckDeviceSupportBlockingAicpuOpProcess(is_support) != SUCCESS) {
    GELOGE(FAILED, "[Call][CheckDeviceSupportBlockingAicpuOpProcess] Call CheckDeviceSupportBlockingAicpuOpProcess failed");
    return FAILED;
  }
  if (!is_support) {
    GELOGD("device not support blocking aicpu op process.");
    return SUCCESS;
  }
  GELOGD("Distribute wait task begin");
  rtEvent_t rt_event = nullptr;
  if (davinci_model_->GetEventByStream(stream_, rt_event) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call GetEventByStream failed");
    GELOGE(FAILED, "[Call][GetEventByStream] Call GetEventByStream failed");
    return FAILED;
  }
  auto rt_ret = rtStreamWaitEvent(stream_, rt_event);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamWaitEvent failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtApi] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtEventReset(rt_event, stream_);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtEventReset failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtApi] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

void KernelTaskInfo::SetIoAddrs(const OpDescPtr &op_desc) {
  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);

  io_addrs_.insert(io_addrs_.end(), input_data_addrs.begin(), input_data_addrs.end());
  io_addrs_.insert(io_addrs_.end(), output_data_addrs.begin(), output_data_addrs.end());
  if (kernel_type_ == ccKernelType::TE) {
    vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);
    io_addrs_.insert(io_addrs_.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());
  }
}

Status KernelTaskInfo::CopyNoncontinuousArgs(uint16_t offset) {
  GE_CHECK_NOTNULL(davinci_model_);
  // copy new io addrs
  vector<void *> io_addrs = io_addrs_;
  davinci_model_->UpdateKnownZeroCopyAddr(io_addrs);
  auto addr_size = kAddrLen * io_addrs.size();

  // copy io addr
  errno_t sec_ret = memcpy_s(args_addr.get() + offset, addr_size, io_addrs.data(), addr_size);
  if (sec_ret != EOK) {
    REPORT_CALL_ERROR("E19999", "Call memcpy_s fail, size:%zu, ret:0x%X", addr_size, sec_ret);
    GELOGE(FAILED, "[Call][Memcpy] failed, size:%zu, ret:%d", addr_size, sec_ret);
    return FAILED;
  }

  // copy args to device
  rtError_t rt_ret = rtMemcpy(args_, args_size_, args_addr.get(), args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GELOGD("Copy noncontinuous args success, kernel type %d.", kernel_type_);
  return SUCCESS;
}

Status KernelTaskInfo::UpdateArgs() {
  GELOGI("KernelTaskInfo::UpdateArgs in.");
  GE_CHECK_NOTNULL(davinci_model_);
  if (kernel_type_ == ccKernelType::TE) {
    if (l2_buffer_on_) {
      return CopyNoncontinuousArgs(io_addr_offset_);
    }
    davinci_model_->SetTotalIOAddrs(io_addrs_);
    davinci_model_->UpdateOpIOAddrs(task_id_, stream_id_, io_addrs_);
  } else if (kernel_type_ == ccKernelType::AI_CPU || kernel_type_ == ccKernelType::CUST_AI_CPU) {
    return CopyNoncontinuousArgs(sizeof(aicpu::AicpuParamHead));
  }
  return SUCCESS;
}

Status KernelTaskInfo::Release() {
  if (davinci_model_ != nullptr && davinci_model_->IsKnownNode()) {
    return SUCCESS;
  }
  rtContext_t ctx = nullptr;
  rtError_t ret = rtCtxGetCurrent(&ctx);

  if (ret == RT_ERROR_NONE) {
    FreeRtMem(&args_);
    FreeRtMem(&superkernel_device_args_addr_);
    FreeRtMem(&superkernel_dev_nav_table_);
    FreeRtMem(&flowtable_);
    FreeRtMem(&custom_info_.input_descs);
    FreeRtMem(&custom_info_.input_addrs);
    FreeRtMem(&custom_info_.output_descs);
    FreeRtMem(&custom_info_.output_addrs);
    FreeRtMem(&custom_info_.attr_handle);
    FreeRtMem(&aicpu_ext_info_addr_);
  }

  if (ctx_.argsOffset != nullptr) {
    delete[] ctx_.argsOffset;
    ctx_.argsOffset = nullptr;
  }

  ret = (sm_desc_ != nullptr) ? rtMemFreeManaged(sm_desc_) : RT_ERROR_NONE;
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemFreeManaged failed, ret:0x%X", ret);
    GELOGE(RT_FAILED, "[Call][RtMemFreeManaged] failed, ret:0x%X", static_cast<int>(ret));
    return RT_ERROR_TO_GE_STATUS(ret);
  }
  sm_desc_ = nullptr;

  return SUCCESS;
}

Status KernelTaskInfo::UpdateL2Data(const domi::KernelDef &kernel_def) {
  string sm_desc = kernel_def.sm_desc();
  if (sm_desc.empty()) {
    return SUCCESS;
  }

  char *sm_control = const_cast<char *>(sm_desc.data());
  rtL2Ctrl_t *l2_ctrl_info = reinterpret_cast<rtL2Ctrl_t *>(sm_control);
  uint64_t gen_base_addr = davinci_model_->GetRtBaseAddr();

  // There is no weight for te op now. Update L2_mirror_addr by data memory base.
  uint64_t data_base_addr = (uint64_t)(uintptr_t)davinci_model_->MemBase() - (uint64_t)gen_base_addr;
  const uint32_t l2_ctrl_info_data_count = 8;
  for (uint32_t data_index = 0; data_index < l2_ctrl_info_data_count; ++data_index) {
    if (l2_ctrl_info->data[data_index].L2_mirror_addr != 0) {
      l2_ctrl_info->data[data_index].L2_mirror_addr += data_base_addr;
      l2_ctrl_info->data[data_index].L2_load_to_ddr = IsL2CpToDDR(l2_ctrl_info->data[data_index].L2_load_to_ddr);
    }
  }

  rtError_t rt_ret = rtMemAllocManaged(&sm_desc_, sm_desc.size(), RT_MEMORY_SPM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemAllocManaged failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemAllocManaged] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtMemcpy(sm_desc_, sm_desc.size(), sm_desc.data(), sm_desc.size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", sm_desc.size(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", sm_desc.size(), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  return SUCCESS;
}

void KernelTaskInfo::SetContinuousArgs(uint32_t args_size, DavinciModel *davinci_model) {
  args_offset_ = davinci_model->GetTotalArgsSize();
  davinci_model->SetTotalArgsSize(args_size);
}

void KernelTaskInfo::SetNoncontinuousArgs(uint32_t args_size, DavinciModel *davinci_model) {
  hybrid_args_offset_ = davinci_model->GetHybridArgsSize();
  davinci_model->SetHybridArgsSize(args_size);
}

Status KernelTaskInfo::CalculateArgs(const domi::TaskDef &task_def, DavinciModel *davinci_model) {
  GE_CHECK_NOTNULL(davinci_model);
  const domi::KernelDef &kernel_def = task_def.kernel();
  const domi::KernelContext &context = kernel_def.context();
  kernel_type_ = static_cast<ccKernelType>(context.kernel_type());
  uint32_t args_size = kernel_def.args_size();
  if (kernel_type_ == ccKernelType::TE) {
    if (kernel_def.sm_desc().empty()) {
      SetContinuousArgs(args_size, davinci_model);
      return SUCCESS;
    }
    l2_buffer_on_ = true;
    SetNoncontinuousArgs(args_size, davinci_model);
  } else if (kernel_type_ == ccKernelType::AI_CPU || kernel_type_ == ccKernelType::CUST_AI_CPU) {
    SetNoncontinuousArgs(args_size, davinci_model);
  }
  return SUCCESS;
}

Status KernelTaskInfo::InitTVMTask(uint16_t offset, const domi::KernelDef &kernel_def) {
  GELOGD("Do InitTVMTask.");
  GE_CHECK_NOTNULL(davinci_model_);
  // get tvm op desc
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(ctx_.opIndex);
  GE_CHECK_NOTNULL(op_desc);

  args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_size_]);
  GE_CHECK_NOTNULL(args_addr);
  errno_t sec_ret = memcpy_s(args_addr.get(), args_size_, kernel_def.args().data(), args_size_);
  if (sec_ret != EOK) {
    REPORT_CALL_ERROR("E19999", "Call memcpy_s fail, size:%u, ret:0x%X", args_size_, sec_ret);
    GELOGE(FAILED, "[Call][Memcpy] failed, size:%u, ret:0x%X", args_size_, sec_ret);
    return FAILED;
  }

  Status ge_ret = UpdateL2Data(kernel_def);
  // update origin l2 data
  if (ge_ret != SUCCESS) {
    return ge_ret;
  }

  if (davinci_model_->IsKnownNode()) {
    args_ = l2_buffer_on_ ? davinci_model_->GetCurrentHybridArgsAddr(hybrid_args_offset_)
                          : davinci_model_->GetCurrentArgsAddr(args_offset_);
    InitDumpArgs(offset);
    return SUCCESS;
  }

  // Update Stub
  // When training, when the the second call to DavinciModel::init() comes here, stub_func_ is already valid,
  // and does not need to be modified.
  // When inferencing, stub_func_ is different from dynamic-registration to runtime, and needs to be modified.
  string session_graph_model_id;
  davinci_model_->GetUniqueId(op_desc, session_graph_model_id);
  const char *bin_file_key = davinci_model_->GetRegisterStub(op_desc->GetName(), session_graph_model_id);
  rtError_t rt_ret = rtQueryFunctionRegistered(const_cast<char *>(bin_file_key));
  if (rt_ret != RT_ERROR_NONE) {
    stub_func_ = const_cast<char *>(bin_file_key);
  }

  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  const vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  const vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  const vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);

  vector<void *> tensor_device_addrs;
  tensor_device_addrs.insert(tensor_device_addrs.end(), input_data_addrs.begin(), input_data_addrs.end());
  tensor_device_addrs.insert(tensor_device_addrs.end(), output_data_addrs.begin(), output_data_addrs.end());
  tensor_device_addrs.insert(tensor_device_addrs.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());

  // malloc args memory
  rt_ret = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X", args_size_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X", args_size_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  // copy orign args
  rt_ret = rtMemcpy(args_, args_size_, kernel_def.args().data(), args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if ((args_size_ <= offset) || (args_size_ - offset < kAddrLen * tensor_device_addrs.size())) {
    REPORT_INNER_ERROR("E19999", "offset:%u >= kernelInfo.argsSize:%u or copy content:%zu beyond applied memory:%u, "
                       "check invalid", offset, args_size_, kAddrLen * tensor_device_addrs.size(), args_size_ - offset);
    GELOGE(FAILED, "[Check][Param] offset:%u >= kernelInfo.argsSize:%u or copy content:%zu beyond applied memory:%u, "
           "check invalid", offset, args_size_, kAddrLen * tensor_device_addrs.size(), args_size_ - offset);
    return FAILED;
  }

  // copy args
  rt_ret = rtMemcpy(static_cast<char *>(args_) + offset, args_size_ - offset, tensor_device_addrs.data(),
                    kAddrLen * tensor_device_addrs.size(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", args_size_ - offset, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", args_size_ - offset, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  sec_ret = memcpy_s(args_addr.get() + offset, args_size_ - offset, tensor_device_addrs.data(),
                     kAddrLen * tensor_device_addrs.size());
  if (sec_ret != EOK) {
    REPORT_CALL_ERROR("E19999", "Call memcpy_s failed, size:%u, ret:0x%X", args_size_ - offset, sec_ret);
    GELOGE(FAILED, "[Call][Memcpy] failed, size:%u, ret:0x%X", args_size_ - offset, sec_ret);
    return FAILED;
  }
  skt_dump_args_ = static_cast<char *>(args_) + offset;
  InitDumpArgs(offset);

  vector<void *> virtual_io_addrs;  // use virtual address for zero copy key.
  virtual_io_addrs.insert(virtual_io_addrs.end(), input_data_addrs.begin(), input_data_addrs.end());
  virtual_io_addrs.insert(virtual_io_addrs.end(), output_data_addrs.begin(), output_data_addrs.end());
  if (op_desc->GetType() == ATOMICADDRCLEAN) {
    virtual_io_addrs.insert(virtual_io_addrs.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());
  }
  davinci_model_->SetZeroCopyAddr(op_desc, virtual_io_addrs, args_addr.get(), args_, args_size_, offset);

  GELOGD("Do InitTVMTask end");
  return SUCCESS;
}

bool KernelTaskInfo::IsL1FusionOp(const OpDescPtr &op_desc) {
  std::vector<int64_t> input_memory_type;
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, input_memory_type);
  for (size_t i = 0; i < input_memory_type.size(); ++i) {
    if (input_memory_type.at(i) == RT_MEMORY_L1) {
      return true;
    }
  }

  std::vector<int64_t> output_memory_type;
  (void)ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, output_memory_type);
  for (size_t i = 0; i < output_memory_type.size(); ++i) {
    if (output_memory_type.at(i) == RT_MEMORY_L1) {
      return true;
    }
  }
  return false;
}

Status KernelTaskInfo::InitAICPUCustomTask(uint32_t op_index, const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAICPUCustomTask");
  OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index);
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", op_index);
    GELOGE(INTERNAL_ERROR, "[Get][Op] index is out of range, index:%u", op_index);
    return INTERNAL_ERROR;
  }

  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();

  const domi::KernelContext &context = kernel_def.context();
  const uint32_t kCustomAicpuArgsLen = 5;
  ctx_.argsOffset = new (std::nothrow) uint16_t[kCustomAicpuArgsLen]();
  if (ctx_.argsOffset == nullptr) {
    REPORT_CALL_ERROR("E19999", "New ctx_.argsOffset fail, size:%u, op:%s(%s)",
                      kCustomAicpuArgsLen, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Malloc][Memory] ctx_.argsOffset is null, size:%u, op:%s(%s)",
           kCustomAicpuArgsLen, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  if (context.args_offset().size() / sizeof(uint16_t) < kCustomAicpuArgsLen) {
    REPORT_INNER_ERROR("E19999", "context.args_offset().size():%zu / sizeof(uint16_t) is less than "
                       "kCustomAicpuArgsLen:%u, op:%s(%s), check invalid", context.args_offset().size(),
                       kCustomAicpuArgsLen, op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] context.args_offset().size():%zu / sizeof(uint16_t) is less than "
           "kCustomAicpuArgsLen:%u, op:%s(%s)", context.args_offset().size(), kCustomAicpuArgsLen,
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return PARAM_INVALID;
  }

  for (uint32_t i = 0; i < kCustomAicpuArgsLen; ++i) {
    ctx_.argsOffset[i] = (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[i];
  }

  const std::vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  const std::vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  Status ret = StoreInputOutputTensor(input_data_addrs, output_data_addrs, ModelUtils::GetInputDescs(op_desc),
                                      ModelUtils::GetOutputDescs(op_desc));
  if (ret != SUCCESS) {
    GELOGE(ret, "[Store][InputOutputTensor] Failed, op:%s(%s)", op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return ret;
  }

  // attrHandle
  Buffer buffer;
  if (!AttrUtils::GetBytes(op_desc, ATTR_NAME_OPATTR, buffer)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail", ATTR_NAME_OPATTR.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(FAILED, "[Get][Attr] %s in op:%s(%s) fail", ATTR_NAME_OPATTR.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return FAILED;
  }

  uint32_t op_attr_size = buffer.GetSize();
  if (op_attr_size == 0) {
    REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s) size is 0, check invalid",
                       ATTR_NAME_OPATTR.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] param op_attr_size is out of range, op:%s", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }

  rtError_t rt_ret = rtMalloc(&custom_info_.attr_handle, op_attr_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed for op:%s(%s), size:%u, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_attr_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed for op:%s(%s), size:%u, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_attr_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtMemcpy(custom_info_.attr_handle, op_attr_size, buffer.GetData(), op_attr_size, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed for op:%s(%s), size:%u, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_attr_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed for op:%s(%s), size:%u, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), op_attr_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  // args
  char *args = const_cast<char *>(kernel_def.args().data());

  for (uint32_t i = 0; i < kCustomAicpuArgsLen; ++i) {
    if (kernel_def.args().size() < ((size_t)ctx_.argsOffset[i] + sizeof(uint64_t))) {
      REPORT_INNER_ERROR("E19999", "ctx.argsOffset[%u]: %u + sizeof(uint64_t): %zu >= kernelDef.args().size():%zu, "
                         "op:%s(%s) check invalid", i, (uint32_t)ctx_.argsOffset[i],
                         sizeof(uint64_t), kernel_def.args().size(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str());
      GELOGE(FAILED, "[Check][Param] ctx.argsOffset[%u]:%u + sizeof(uint64_t):%zu >= kernelDef.args().size():%zu", i,
             (uint32_t)ctx_.argsOffset[i], sizeof(uint64_t), kernel_def.args().size());
      return FAILED;
    }
  }
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[kArgsInputDesc])) =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.input_descs));  // arg 0
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[kArgsInputAddr])) =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.input_addrs));  // arg 1
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[kArgsOutputDesc])) =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.output_descs));  // arg 2
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[kArgsOutputAddr])) =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.output_addrs));  // arg 3
  *(reinterpret_cast<uint64_t *>(args + ctx_.argsOffset[kArgsAttrHandle])) =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(custom_info_.attr_handle));  // arg 4

  rt_ret = rtMalloc(&args_, args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed for op:%s(%s), size:%u, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed for op:%s(%s), size:%u, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtMemcpy(args_, kernel_def.args_size(), kernel_def.args().data(), kernel_def.args_size(),
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed for op:%s(%s), size:%u, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_def.args_size(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed for op:%s(%s), size:%u, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), kernel_def.args_size(), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  davinci_model_->SetZeroCopyAddr(op_desc, input_data_addrs, input_data_addrs.data(), custom_info_.input_addrs,
                                  input_data_addrs.size() * kAddrLen, 0);
  davinci_model_->SetZeroCopyAddr(op_desc, output_data_addrs, output_data_addrs.data(), custom_info_.output_addrs,
                                  output_data_addrs.size() * kAddrLen, 0);
  return SUCCESS;
}

Status KernelTaskInfo::InitCceTask(const domi::KernelDef &kernel_def) {
  GELOGI("Do InitCCETask");
  if (davinci_model_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param davinci_model nullptr");
    GELOGE(PARAM_INVALID, "[Check][Param] davinci_model is null!");
    return PARAM_INVALID;
  }
  Status ret = SetContext(kernel_def);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Context] Fail.");
    return ret;
  }

  string flowtable = kernel_def.flowtable();
  const domi::KernelContext &context = kernel_def.context();

  if (context.is_flowtable()) {
    if (flowtable.empty()) {
      REPORT_INNER_ERROR("E19999", "kernel_def.flowtable is empty, check invalid");
      GELOGE(FAILED, "[Check][Param] flowtable is null.");
      return FAILED;
    }
  }

  // get smDesc stored in model
  string sm_desc = kernel_def.sm_desc();
  uint64_t sm_contrl_size = sm_desc.empty() ? 0 : sizeof(rtSmDesc_t);

  // Passing the memory info when the offline-model-generated to the CCE, which uses this info for address refresh
  ctx_.genDataBaseAddr = davinci_model_->GetRtBaseAddr();
  ctx_.genDataBaseSize = davinci_model_->TotalMemSize();
  ctx_.genWeightBaseAddr = davinci_model_->GetRtWeightAddr();
  ctx_.genWeightBaseSize = davinci_model_->TotalWeightsMemSize();
  ctx_.genVariableBaseAddr = davinci_model_->GetRtVarAddr();
  ctx_.genVariableBaseSize = davinci_model_->TotalVarMemSize();
  ctx_.l2ctrlSize = sm_contrl_size;

  ret = UpdateCceArgs(sm_desc, flowtable, kernel_def);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Update][CceArgs] fail");
    return ret;
  }

  // flowtable
  ret = SetFlowtable(flowtable, kernel_def);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Flowtable] Fail");
    return ret;
  }

  // args
  rtError_t rt_ret = rtMalloc(&args_, kernel_def.args_size(), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X", kernel_def.args_size(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X", kernel_def.args_size(), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "cce task physical memory.", kernel_def.args_size())

  rt_ret = rtMemcpy(args_, kernel_def.args_size(), kernel_def.args().data(), kernel_def.args_size(),
                    RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", kernel_def.args_size(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", kernel_def.args_size(), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  // L2
  if (!sm_desc.empty()) {
    rt_ret = rtMemAllocManaged(&sm_desc_, sm_desc.size(), RT_MEMORY_SPM);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMemAllocManaged failed, ret:0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMemAllocManaged] failed, ret:0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    rt_ret = rtMemcpy(sm_desc_, sm_desc.size(), sm_desc.data(), sm_desc.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", sm_desc.size(), rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", sm_desc.size(), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }
  return SUCCESS;
}

Status KernelTaskInfo::InitAicpuTask(uint32_t op_index, const domi::KernelDef &kernel_def) {
  GELOGI("Do InitAicpuTask");
  so_name_ = kernel_def.so_name();
  kernel_name_ = kernel_def.kernel_name();

  OpDescPtr op_desc = davinci_model_->GetOpByIndex(op_index);
  if (op_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "Can't get op_desc from davinci_model by index:%u", op_index);
    GELOGE(INTERNAL_ERROR, "[Get][Op] index is out of range, index:%u", op_index);
    return INTERNAL_ERROR;
  }
  GELOGI("node[%s] test so name %s, kernel name %s", op_desc->GetName().c_str(), so_name_.c_str(),
         kernel_name_.c_str());

  if (kernel_type_ == ccKernelType::CUST_AI_CPU) {
    bool loaded = false;
    GE_CHK_STATUS_RET(ModelManager::GetInstance()->LoadCustAicpuSo(op_desc, so_name_, loaded),
                      "[Launch][CustAicpuSo] failed");
  }

  // copy args to new host memory
  args_addr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[args_size_]);
  GE_CHECK_NOTNULL(args_addr);
  GE_PRINT_DYNAMIC_MEMORY(new, "cce task physical memory.", sizeof(uint8_t) * args_size_)
  errno_t sec_ret = memcpy_s(args_addr.get(), args_size_, kernel_def.args().data(), args_size_);
  if (sec_ret != EOK) {
    REPORT_CALL_ERROR("E19999", "Call memcpy_s fail, size:%u, ret:0x%X", args_size_, sec_ret);
    GELOGE(FAILED, "[Call][Memcpy] failed, size:%u, ret:0x%X", args_size_, sec_ret);
    return FAILED;
  }

  auto aicpu_param_head = reinterpret_cast<aicpu::AicpuParamHead *>(args_addr.get());
  const auto &ext_info = kernel_def.kernel_ext_info();
  auto init_ret = InitAicpuTaskExtInfo(ext_info);
  if (init_ret != SUCCESS) {
    GELOGE(init_ret, "[Init][AicpuTaskExtInfo] failed, ext_info size=%zu", ext_info.size());
    return init_ret;
  }
  GELOGI("Node[%s] type[%s] kernel_ext_info size=%zu, aicpu_ext_info_addr_=%p", op_desc->GetName().c_str(),
         op_desc->GetType().c_str(), ext_info.size(), aicpu_ext_info_addr_);

  aicpu_param_head->extInfoAddr = reinterpret_cast<uintptr_t>(aicpu_ext_info_addr_);
  aicpu_param_head->extInfoLength = static_cast<uintptr_t>(ext_info.size());

  if (davinci_model_->IsKnownNode()) {
    args_ = davinci_model_->GetCurrentHybridArgsAddr(hybrid_args_offset_);
    InitDumpArgs(sizeof(aicpu::AicpuParamHead));
    return SUCCESS;
  }
  const RuntimeParam &rts_param = davinci_model_->GetRuntimeParam();
  vector<void *> input_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
  vector<void *> output_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
  vector<void *> io_addrs;
  io_addrs.insert(io_addrs.end(), input_addrs.begin(), input_addrs.end());
  io_addrs.insert(io_addrs.end(), output_addrs.begin(), output_addrs.end());
  if (!io_addrs.empty()) {
    // refresh io addrs
    uintptr_t io_addr = reinterpret_cast<uintptr_t>(args_addr.get()) + sizeof(aicpu::AicpuParamHead);
    auto addrs_size = sizeof(uint64_t) * io_addrs.size();
    sec_ret = memcpy_s(reinterpret_cast<void *>(io_addr), addrs_size, io_addrs.data(), addrs_size);
    if (sec_ret != EOK) {
      REPORT_CALL_ERROR("E19999", "Call memcpy_s fail, size:%lu, ret:0x%X", addrs_size, sec_ret);
      GELOGE(FAILED, "[Call][Memcpy] failed, size:%lu, ret:0x%X", addrs_size, sec_ret);
      return FAILED;
    }
  }

  // malloc device memory for args
  rtError_t rt_ret = rtMalloc(static_cast<void **>(&args_), args_size_, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed for op:%s(%s), size:%u, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed for op:%s(%s), size:%u, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "cce task physical memory.", args_size_)

  // copy args to device
  rt_ret = rtMemcpy(args_, args_size_, args_addr.get(), args_size_, RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed for op:%s(%s), size:%u, ret:0x%X",
                      op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size_, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed for op:%s(%s), size:%u, ret:0x%X",
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), args_size_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  InitDumpArgs(sizeof(aicpu::AicpuParamHead));

  davinci_model_->SetZeroCopyAddr(op_desc, io_addrs, args_addr.get(), args_, args_size_, sizeof(aicpu::AicpuParamHead));

  return SUCCESS;
}

void KernelTaskInfo::InitDumpFlag() {
  if (davinci_model_->OpNeedDump(op_desc_->GetName())) {
    GELOGD("Op %s init dump flag", op_desc_->GetName().c_str());
    if (IsL1FusionOp(op_desc_)) {
      dump_flag_ = RT_FUSION_KERNEL_DUMPFLAG;
    } else {
      dump_flag_ = RT_KERNEL_DUMPFLAG;
    }
  }
}

void KernelTaskInfo::InitDumpArgs(uint32_t offset) {
  if (davinci_model_->OpNeedDump(op_desc_->GetName())) {
    GELOGD("Op %s need dump in task info", op_desc_->GetName().c_str());
    dump_args_ = static_cast<char *>(args_) + offset;
  }
  if (davinci_model_->GetOpDugReg()) {
    GELOGD("Op debug is open in kernel task info");
    dump_args_ = static_cast<char *>(args_) + offset;
  }
  if (kernel_type_ == ccKernelType::CUST_AI_CPU) {
    dump_flag_ |= RT_KERNEL_CUSTOM_AICPU;
  }
}

Status KernelTaskInfo::InitAicpuTaskExtInfo(const std::string &ext_info) {
  if (ext_info.empty()) {
    return SUCCESS;
  }

  int32_t unknown_shape_type_val = 0;
  (void) AttrUtils::GetInt(op_desc_, ::ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  UnknowShapeOpType unknown_type = static_cast<UnknowShapeOpType>(unknown_shape_type_val);
  uint32_t num_inputs = op_desc_->GetInputsSize();
  uint32_t num_outputs = op_desc_->GetOutputsSize();
  std::shared_ptr<ge::hybrid::AicpuExtInfoHandler> ext_handle(
          new(std::nothrow) ::ge::hybrid::AicpuExtInfoHandler(op_desc_->GetName(),
                                                              num_inputs,
                                                              num_outputs,
                                                              unknown_type));
  GE_CHK_BOOL_RET_STATUS(ext_handle != nullptr, FAILED, "[Malloc][Memory] for aicpu_ext_handle failed!");
  GE_CHK_STATUS_RET(ext_handle->Parse(ext_info),
                    "[Parse][KernelExtInfo] failed, kernel_ext_info_size=%zu, op:%s.",
                    ext_info.size(), op_desc_->GetName().c_str());
  GE_CHK_STATUS_RET(ext_handle->UpdateSessionInfoSessionId(davinci_model_->GetSessionId()),
                    "[Update][SessionInfoSessionId] failed, op:%s", op_desc_->GetName().c_str());
  GELOGD("Update aicpu_task ext_info session_info session_id is %lu", davinci_model_->GetSessionId());
  GE_CHK_STATUS_RET(ext_handle->UpdateExecuteMode(true),
                    "[Update][ExecuteMode] failed, op:%s", op_desc_->GetName().c_str());
  GELOGD("Update aicpu_task ext_info bit_map execute mode to 1.");
  topic_type_flag_ = ext_handle->GetTopicTypeFlag();

  bool all_shape = false;
  (void)AttrUtils::GetBool(op_desc_, kAicpuAllshape, all_shape);
  if (all_shape) {
    GELOGD("Aicpu all_shape kernel need to update io shape.");
    for (uint32_t i = 0; i < num_inputs; i++) {
      auto input_desc = op_desc_->MutableInputDesc(i);
      GE_CHECK_NOTNULL(input_desc);
      GE_CHK_STATUS_RET(ext_handle->UpdateInputShapeAndType(i, *input_desc),
                        "[Call][UpdateInputShapeAndType] Input[%u] update input shape failed, op:%s.",
                        i, op_desc_->GetName().c_str());
    }
    for (uint32_t j = 0; j < num_outputs; j++) {
      auto output_desc = op_desc_->MutableOutputDesc(j);
      GE_CHECK_NOTNULL(output_desc);
      GE_CHK_STATUS_RET(ext_handle->UpdateOutputShapeAndType(j, *output_desc),
                        "[Call][UpdateOutputShapeAndType] Output[%u] update output shape failed, op:%s.",
                        j, op_desc_->GetName().c_str());
    }
  }

  AttrUtils::GetBool(op_desc_, ATTR_NAME_IS_BLOCKING_OP, is_blocking_aicpu_op_);
  GELOGD("Get op:%s attribute(is_blocking_op), value:%d", op_desc_->GetName().c_str(), is_blocking_aicpu_op_);

  if (UpdateEventIdForAicpuBlockingOp(ext_handle) != SUCCESS) {
    GELOGE(FAILED, "[Call][UpdateEventIdForAicpuBlockingOp] failed for op:%s(%s)",
           op_desc_->GetName().c_str(), op_desc_->GetType().c_str());
    return FAILED;
  }

  auto rt_ret = rtMalloc(&aicpu_ext_info_addr_, ext_handle->GetExtInfoLen(), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed for op:%s(%s), size:%zu, ret:0x%X",
                      op_desc_->GetName().c_str(), op_desc_->GetType().c_str(),
                      ext_handle->GetExtInfoLen(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed for op:%s(%s), size:%zu, ret:0x%X",
           op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ext_handle->GetExtInfoLen(), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtMemcpy(aicpu_ext_info_addr_, ext_handle->GetExtInfoLen(), ext_handle->GetExtInfo(),
                    ext_handle->GetExtInfoLen(), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed for op:%s(%s), size:%zu, ret:0x%X",
                      op_desc_->GetName().c_str(), op_desc_->GetType().c_str(),
                      ext_handle->GetExtInfoLen(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed for op:%s(%s), size:%zu, ret:0x%X",
           op_desc_->GetName().c_str(), op_desc_->GetType().c_str(), ext_handle->GetExtInfoLen(), rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  return SUCCESS;
}

Status KernelTaskInfo::StoreInputOutputTensor(const std::vector<void *> &input_data_addrs,
                                              const std::vector<void *> &output_data_addrs,
                                              const std::vector<::tagCcAICPUTensor> &input_descs,
                                              const std::vector<::tagCcAICPUTensor> &output_descs) {
  auto input_size = input_descs.size();
  auto output_size = output_descs.size();

  // inputDescs
  rtError_t rt_ret = rtMalloc(&custom_info_.input_descs, sizeof(opTensor_t) * input_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%zu, ret:0x%X", sizeof(opTensor_t) * input_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%zu, ret:0x%X", sizeof(opTensor_t) * input_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  for (std::size_t i = 0; i < input_size; ++i) {
    rt_ret = rtMemcpy(static_cast<opTensor_t *>(custom_info_.input_descs) + i, sizeof(opTensor_t),
                      const_cast<tagOpTensor *>(&input_descs[i]), sizeof(opTensor_t), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", sizeof(opTensor_t), rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", sizeof(opTensor_t), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }

  // inputAddrs
  rt_ret = rtMalloc(&custom_info_.input_addrs, sizeof(opTensor_t) * input_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%zu, ret:0x%X", sizeof(opTensor_t) * input_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%zu, ret:0x%X", sizeof(opTensor_t) * input_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if (!input_data_addrs.empty()) {
    rt_ret = rtMemcpy(custom_info_.input_addrs, kAddrLen * input_size, &input_data_addrs[0], kAddrLen * input_size,
                      RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", kAddrLen * input_size, rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", kAddrLen * input_size, rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }

  // outputDescs
  rt_ret = rtMalloc(&custom_info_.output_descs, sizeof(opTensor_t) * output_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%zu, ret:0x%X", sizeof(opTensor_t) * output_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%zu, ret:0x%X", sizeof(opTensor_t) * output_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  for (std::size_t i = 0; i < output_size; ++i) {
    rt_ret = rtMemcpy(static_cast<opTensor_t *>(custom_info_.output_descs) + i, sizeof(opTensor_t),
                      const_cast<tagOpTensor *>(&input_descs[i]), sizeof(opTensor_t), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", sizeof(opTensor_t), rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", sizeof(opTensor_t), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }

  // outputAddrs
  rt_ret = rtMalloc(&custom_info_.output_addrs, sizeof(opTensor_t) * output_size, RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%zu, ret:0x%X", sizeof(opTensor_t) * output_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%zu, ret:0x%X", sizeof(opTensor_t) * output_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  if (!output_data_addrs.empty()) {
    rt_ret = rtMemcpy(custom_info_.output_addrs, kAddrLen * output_size, &output_data_addrs[0], kAddrLen * output_size,
                      RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", kAddrLen * output_size, rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", kAddrLen * output_size, rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }

  return SUCCESS;
}

Status KernelTaskInfo::SetContext(const domi::KernelDef &kernel_def) {
  const domi::KernelContext &context = kernel_def.context();
  ctx_.kernelType = static_cast<ccKernelType>(context.kernel_type());
  ctx_.opId = context.op_id();
  ctx_.kernelFuncId = context.kernel_func_id();
  ctx_.isFlowtable = context.is_flowtable();
  ctx_.argsCount = context.args_count();
  if (ctx_.argsCount == 0) {
    REPORT_INNER_ERROR("E19999", "kernel_def.context.args_count is 0, check invalid");
    GELOGE(INTERNAL_ERROR, "[Check][Param] argsCount is %u.", ctx_.argsCount);
    return INTERNAL_ERROR;
  }

  if (context.args_offset().size() / sizeof(uint16_t) < ctx_.argsCount) {
    REPORT_INNER_ERROR("E19999", "param [context.args_offset().size():%zu / sizeof(uint16_t)] "
                       "is less than [ctx_.argsCount:%u], check invalid",
                       context.args_offset().size(), ctx_.argsCount);
    GELOGE(PARAM_INVALID, "[Check][Param] [context.args_offset().size():%zu / sizeof(uint16_t)] "
           "is less than [ctx_.argsCount:%u], check invalid", context.args_offset().size(), ctx_.argsCount);
    return PARAM_INVALID;
  }

  // ctx_.argsOffset stores the offset of the internal information of agrs_, equal to the ctx_.argsCount
  ctx_.argsOffset = new (std::nothrow) uint16_t[ctx_.argsCount]();
  if (ctx_.argsOffset == nullptr) {
    REPORT_CALL_ERROR("E19999", "New ctx_.argsOffset fail, size:%u", ctx_.argsCount);
    GELOGE(PARAM_INVALID, "[Malloc][Memory] failed, ctx_.argsOffset must not be null, size:%u", ctx_.argsCount);
    return PARAM_INVALID;
  }

  for (uint32_t i = 0; i < ctx_.argsCount; ++i) {
    ctx_.argsOffset[i] = (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[i];
  }

  return SUCCESS;
}

void KernelTaskInfo::FreeRtMem(void **ptr) {
  if (ptr == nullptr || *ptr == nullptr) {
    return;
  }
  rtError_t ret = rtFree(*ptr);
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtFree failed, ret:0x%X", ret);
    GELOGE(RT_FAILED, "[Call][RtFree] failed, ret:0x%X", ret);
  }

  *ptr = nullptr;
}

Status KernelTaskInfo::UpdateCceArgs(std::string &sm_desc, std::string &flowtable, const domi::KernelDef &kernel_def) {
  GE_CHECK_NOTNULL(davinci_model_);
  const domi::KernelContext &context = kernel_def.context();

  uint64_t data_base_addr = reinterpret_cast<uintptr_t>(davinci_model_->MemBase()) - davinci_model_->GetRtBaseAddr();
  uint64_t weight_base_addr =
      reinterpret_cast<uintptr_t>(davinci_model_->WeightsMemBase()) - davinci_model_->GetRtWeightAddr();
  uint64_t var_base_addr = reinterpret_cast<uintptr_t>(davinci_model_->VarMemBase()) - davinci_model_->GetRtVarAddr();

  Status status =
      CceUpdateKernelArgs(context, data_base_addr, weight_base_addr, var_base_addr, sm_desc, flowtable, kernel_def);
  if (status != SUCCESS) {
    GELOGE(status, "[Call][CceUpdateKernelArgs] failed, ret:%d", status);
    return status;
  }
  return SUCCESS;
}

Status KernelTaskInfo::CceUpdateKernelArgs(const domi::KernelContext &context, uint64_t &data_base_addr,
                                           uint64_t &weight_base_addr, uint64_t &var_base_addr, std::string &sm_desc,
                                           std::string &flowtable, const domi::KernelDef &kernel_def) {
  char *sm_contrl = nullptr;
  if (!sm_desc.empty()) {
    sm_contrl = const_cast<char *>(sm_desc.data());
  }

  std::string file_name = "libcce.so";
  std::string path = PluginManager::GetPath();
  path.append(file_name);
  string canonicalPath = RealPath(path.c_str());
  if (canonicalPath.empty()) {
    GELOGW("failed to get realpath of %s", path.c_str());
    return FAILED;
  }

  GELOGI("FileName:%s, Path:%s.", file_name.c_str(), canonicalPath.c_str());
  auto handle = mmDlopen(canonicalPath.c_str(), MMPA_RTLD_NOW | MMPA_RTLD_GLOBAL);
  const char *error = "";
  if (handle == nullptr) {
    error = mmDlerror();
    GE_IF_BOOL_EXEC(error == nullptr, error = "");
    REPORT_INNER_ERROR("E19999", "Failed in dlopen:%s, dlerror:%s", canonicalPath.c_str(), error);
    GELOGE(GE_PLGMGR_SO_NOT_EXIST, "[Open][File] %s failed, reason:%s! ", canonicalPath.c_str(), error);
    return FAILED;
  }
  ccStatus_t cc_ret;
  std::string update_kernel_args = "ccUpdateKernelArgs";
  auto cceUpdateKernelArgs = (ccStatus_t(*)(ccOpContext &, uint64_t, uint64_t,
      uint64_t, void *, uint64_t, void *))mmDlsym(handle, const_cast<char *>(update_kernel_args.c_str()));
  if (cceUpdateKernelArgs == nullptr) {
    REPORT_INNER_ERROR("E19999", "No symbol:%s in %s, check invalid",
                       update_kernel_args.c_str(), canonicalPath.c_str());
    GELOGE(FAILED, "[Invoke][Function] ccUpdateKernelArgs failed.");
    if (mmDlclose(handle) != 0) {
      error = mmDlerror();
      GE_IF_BOOL_EXEC(error == nullptr, error = "");
      GELOGW("Failed to close handle %s", error);
    }
    return FAILED;
  } else {
    GELOGI("Libcce.so has been opened");
    if (context.is_flowtable()) {
      cc_ret = cceUpdateKernelArgs(ctx_, data_base_addr, weight_base_addr, var_base_addr,
                                   const_cast<char *>(flowtable.data()), kernel_def.flowtable().size(), sm_contrl);
    } else {
      cc_ret = cceUpdateKernelArgs(ctx_, data_base_addr, weight_base_addr, var_base_addr,
                                   const_cast<char *>(kernel_def.args().data()), args_size_, sm_contrl);
    }
  }
  if (mmDlclose(handle) != 0) {
    error = mmDlerror();
    GE_IF_BOOL_EXEC(error == nullptr, error = "");
    GELOGW("Failed to close handle %s", error);
    return FAILED;
  }
  if (cc_ret != CC_STATUS_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call cceUpdateKernelArgs fail, ret:0x%X", cc_ret);
    GELOGE(CCE_FAILED, "[Call][CceUpdateKernelArgs] failed, ret:0x%X", cc_ret);
    return CCE_FAILED;
  }

  GELOGI("CceUpdateKernelArgs success!");
  return SUCCESS;
}

Status KernelTaskInfo::SetFlowtable(std::string &flowtable, const domi::KernelDef &kernel_def) {
  const domi::KernelContext &context = kernel_def.context();
  if (context.is_flowtable()) {
    rtError_t rt_ret = rtMalloc(&flowtable_, flowtable.size(), RT_MEMORY_HBM);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%zu, ret:0x%X", flowtable.size(), rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%zu, ret:0x%X", flowtable.size(), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    GE_PRINT_DYNAMIC_MEMORY(rtMalloc, "flowtable refresh of cce scence.", flowtable.size())

    rt_ret = rtMemcpy(flowtable_, flowtable.size(), flowtable.data(), flowtable.size(), RT_MEMCPY_HOST_TO_DEVICE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", flowtable.size(), rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", flowtable.size(), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    // modify flowtable addr in args
    char *args = const_cast<char *>(kernel_def.args().data());

    if (kernel_def.args().size() <
        ((reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0] + sizeof(uint64_t))) {
      REPORT_INNER_ERROR(
          "E19999", "(context.args_offset().data()))[0]:%u + sizeof(uint64_t):%zu > "
          "kernelDef.args().size():%zu, check invalid",
          (uint32_t)((reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0]),
          sizeof(uint64_t), kernel_def.args().size());
      GELOGE(FAILED, "[Check][Param] (context.args_offset().data()))[0]:%u + sizeof(uint64_t):%zu > "
             "kernelDef.args().size():%zu",
             (uint32_t)((reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0]),
             sizeof(uint64_t), kernel_def.args().size());
      return FAILED;
    }

    *(reinterpret_cast<uint64_t *>(
        args + (reinterpret_cast<uint16_t *>(const_cast<char *>(context.args_offset().data())))[0])) =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(flowtable_));
  }
  return SUCCESS;
}

uint8_t KernelTaskInfo::IsL2CpToDDR(uint8_t origain_L2_load_to_ddr) {
  if (origain_L2_load_to_ddr == kL2LoadToDdr) {
    return kL2LoadToDdr;
  }

  if (dump_flag_ == RT_KERNEL_DUMPFLAG) {
    return kL2LoadToDdr;
  }
  return kL2NotLoadToDdr;
}

REGISTER_TASK_INFO(RT_MODEL_TASK_KERNEL, KernelTaskInfo);
}  // namespace ge
