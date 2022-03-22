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

#include "graph/load/model_manager/davinci_model.h"

#include <graph/utils/node_utils.h>
#include <algorithm>
#include <map>
#include <utility>

#include "framework/common/debug/log.h"
#include "common/formats/formats.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/math/math_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "common/profiling/profiling_manager.h"
#include "common/properties_manager.h"
#include "framework/common/scope_guard.h"
#include "common/thread_pool.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/util.h"
#include "common/ge_call_wrapper.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "external/graph/graph.h"
#include "graph/load/model_manager/cpu_queue_schedule.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/tbe_handle_store.h"
#include "graph/manager/graph_mem_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/manager/trans_var_data_utils.h"
#include "graph/manager/util/debug.h"
#include "graph/model_serialize.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "init/gelib.h"
#include "mmpa/mmpa_api.h"
#include "runtime/base.h"
#include "runtime/dev.h"
#include "runtime/event.h"
#include "runtime/mem.h"
#include "runtime/rt_model.h"
#include "runtime/stream.h"
#include "securec.h"
#include "common/local_context.h"
#include "common/formats/utils/formats_trans_utils.h"
#include "common/omg_util.h"
#include "graph/build/memory/block_mem_assigner.h"
#include "graph/manager/session_scope_mem_allocator.h"
#include "framework/omg/omg_inner_types.h"

// create std::thread, catch exceptions using try/catch
#define CREATE_STD_THREAD(thread_id, func, args)                                                  \
  do {                                                                                            \
    try {                                                                                         \
      thread_id = std::thread(func, args);                                                        \
    } catch (const std::system_error &e) {                                                        \
      REPORT_CALL_ERROR("E19999", "Create thread fail, ecode:%d, emsg:%s",                        \
                        e.code().value(), e.what());                                              \
      GELOGE(FAILED, "[Create][Thread] Caught system_error with code:%d, meaning:%s",             \
             e.code().value(), e.what());                                                         \
      GELOGE(FAILED, "[Create][Thread] FAIL, Please check the left resource!");                   \
      return FAILED;                                                                              \
    }                                                                                             \
  } while (0)

namespace ge {
namespace {
const uint32_t kDataIndex = 0;
const uint32_t kTrueBranchStreamNum = 1;
const uint32_t kGetDynamicDimsCount = 1;
const uint32_t kThreadNum = 16;
const uint32_t kAddrLen = sizeof(void *);
const int kDecimal = 10;
const int kBytes = 8;
const uint32_t kDataMemAlignSizeCompare = 64;
const uint32_t kDumpL1FusionOpMByteSize = 2097152;   // 2 * 1024 * 1024
const uint32_t kDumpFlagOfL1Fusion = 0;
const char *const kDefaultBatchLable = "Batch_default";
const char *const kGetDynamicDimsName = "ascend_mbatch_get_dynamic_dims_node";
const char *const kMultiBatchNodePostfix = "_ascend_mbatch_batch_";
const int32_t kInvalidStream = -1;
const uint32_t kEndOfSequence = 0x0704000a;
const uint32_t kEndOfSequenceNew = 507005;
const int32_t kModelAbortNormal = 0x0704000e;
const int32_t kModelAbortNormalNew = 507024;
const uint32_t kInteval = 2;
const uint32_t kFftsTbeHandleElementSize = 2;
const uint32_t kNonTailBlock = 0;
const uint32_t kTailBlock = 1;
const char *const kModelName = "model_name";
const char *const kModeleId = "model_id";
const char *const kLoadStartTime = "load_start_time";
const char *const kLoadEndTime = "load_end_time";
const char *const kFusionOpInfo = "fusion_op_info";
const char *const kFusionOpName = "fusion_op_name";
const char *const kOriginalOpNum = "origin_op_num";
const char *const kOriginalOpName = "origin_op_name";
const char *const kStreamId = "stream_id";
const char *const kFusionOpMemoryInfo = "memory_info";
const char *const kInputSize = "input_size";
const char *const kOutputSize = "output_size";
const char *const kWeightSize = "weight_size";
const char *const kWorkSpaceSize = "workspace_size";
const char *const kTotalSize = "total_size";
const char *const kTaskCount = "task_count";
const char *const kTaskId = "task_id";
const char *const kRequestId = "request_id";
const char *const kThreadId = "thread_id";
const char *const kInputBeginTime = "input_begin_time";
const char *const kInputEndTime = "input_end_time";
const char *const kInferBeginTime = "infer_begin_time";
const char *const kInferEndTime = "infer_end_time";
const char *const kOutputBeginTime = "output_start_time";
const char *const kOutputEndTime = "output_end_time";
const char *const kStubFuncName = "_register_stub_func";
const uint32_t kStringHeadElems = 2;
const uint32_t kPlacementHostData = 0;
const size_t kAlignment = 64;

inline bool IsDataOp(const std::string &node_type) {
  return (node_type == DATA_TYPE) || (node_type == AIPP_DATA_TYPE) || (node_type == ANN_DATA_TYPE);
}

bool IsTbeTask(const OpDescPtr &op_desc) {
  uint32_t run_mode = static_cast<uint32_t>(domi::ImplyType::INVALID);
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_IMPLY_TYPE, run_mode)) {
    return false;
  }

  if (run_mode != static_cast<uint32_t>(domi::ImplyType::TVM)) {
    return false;
  }

  // Skip no_task operator, such as concat and split.
  bool attr_no_task = false;
  bool get_attr_no_task_flag = AttrUtils::GetBool(op_desc, ATTR_NAME_NOTASK, attr_no_task);
  if (get_attr_no_task_flag && attr_no_task) {
    GELOGI("Node[name:%s, type:%s] does not generate task, skip initialization.",
           op_desc->GetName().c_str(), op_desc->GetType().c_str());
    return false;
  }

  return true;
}

inline bool IsNoTaskAndDumpNeeded(const OpDescPtr &op_desc) {
  bool save_dump_info = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NO_TASK_AND_DUMP_NEEDED, save_dump_info);
  return save_dump_info;
}
}  // namespace

std::mutex DavinciModel::tvm_bin_mutex_;

DavinciModel::DavinciModel(int32_t priority, const std::shared_ptr<ModelListener> &listener)
    : weights_mem_base_(nullptr),
      var_mem_base_(nullptr),
      fixed_mem_base_(0),
      mem_base_(nullptr),
      is_inner_mem_base_(false),
      is_inner_weight_base_(false),
      data_inputer_(nullptr),
      load_begin_time_(0),
      load_end_time_(0),
      time_info_(),
      dataInputTid(0),
      is_weight_mem_has_inited_(false),
      is_feature_map_mem_has_inited_(false),
      model_id_(0),
      runtime_model_id_(0),
      version_(0),
      ge_model_(nullptr),
      listener_(listener),
      run_flg_(false),
      priority_(priority),
      rt_model_handle_(nullptr),
      rt_model_stream_(nullptr),
      is_inner_model_stream_(false),
      is_async_mode_(false),
      last_execute_mode_(INITIALIZATION),
      session_id_(0),
      device_id_(0),
      maxDumpOpNum_(0), data_dumper_(&runtime_param_),
      iterator_count_(0),
      is_l1_fusion_enable_(false),
      is_first_execute_(true) {
  op_list_.clear();
  skt_info_ = {0, 0, 0, 0, nullptr, nullptr, {}, {}, {}, {}, {}, RT_KERNEL_DEFAULT, -1, 0, nullptr};
}

DavinciModel::~DavinciModel() {
  try {
    GE_CHK_STATUS(ModelRunStop());

    Status ret = data_dumper_.UnloadDumpInfo();
    if (ret != SUCCESS) {
      GELOGW("UnloadDumpInfo failed, ret: %u.", ret);
    }

    ClearTaskAddrs();

    op_list_.clear();
    tensor_name_to_fixed_addr_size_.clear();
    tensor_name_to_peer_output_index_.clear();
    GE_DELETE_NEW_SINGLE(data_inputer_);
    // check rt ctx is exist. rt api call will cause error log when ctx not exist
    rtContext_t ctx = nullptr;
    rtError_t rt_ret = rtCtxGetCurrent(&ctx);
    if (rt_ret == RT_ERROR_NONE) {
      UnbindTaskSinkStream();
      for (size_t i = 0; i < label_list_.size(); ++i) {
        if (label_list_[i] != nullptr) {
          GE_LOGW_IF(rtLabelDestroy(label_list_[i]) != RT_ERROR_NONE, "Destroy label failed, index:%zu.", i);
        }
      }

      for (size_t i = 0; i < stream_list_.size(); ++i) {
        GE_LOGW_IF(rtStreamDestroy(stream_list_[i]) != RT_ERROR_NONE, "Destroy stream failed, index:%zu.", i);
      }

      for (size_t i = 0; i < event_list_.size(); ++i) {
        GE_LOGW_IF(rtEventDestroy(event_list_[i]) != RT_ERROR_NONE, "Destroy event failed, index: %zu", i);
      }

      for (const auto &it : stream_2_event_) {
        if (rtEventDestroy(it.second) != RT_ERROR_NONE) {
          GELOGW("Destroy event failed");
        }
      }

      FreeWeightsMem();

      FreeFeatureMapMem();

      FreeExMem();

      OpDebugUnRegister();

      if (l1_fusion_addr_ != nullptr) {
        GE_CHK_RT(rtFree(l1_fusion_addr_));
      }

      if (rt_model_handle_ != nullptr) {
        GE_CHK_RT(rtModelDestroy(rt_model_handle_));
        rt_model_handle_ = nullptr;
      }
    }

    ReleaseTask();
    CleanTbeHandle();

    var_mem_base_ = nullptr;
    if (known_node_) {
      if (args_ != nullptr) {
        GE_CHK_RT(rtFree(args_));
      }
      total_io_addrs_.clear();
      if (fixed_addrs_ != nullptr) {
        GE_CHK_RT(rtFree(fixed_addrs_));
      }
    }
  } catch (...) {
    GELOGW("DavinciModel::~DavinciModel: clear op_list catch exception.");
  }
}

void DavinciModel::ClearTaskAddrs() {
  for (const auto &op_and_addr : saved_task_addrs_) {
    auto addr = op_and_addr.second;
    if (addr != nullptr) {
      GE_CHK_RT(rtFree(addr));
    }
    addr = nullptr;
  }
  saved_task_addrs_.clear();
}

void DavinciModel::UnbindHcomStream() {
  if (!all_hccl_stream_list_.empty()) {
    for (size_t i = 0; i < all_hccl_stream_list_.size(); i++) {
      GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, all_hccl_stream_list_[i]) != RT_ERROR_NONE,
                 "Unbind hccl stream from model failed, Index: %zu", i);
      GE_LOGW_IF(rtStreamDestroy(all_hccl_stream_list_[i]) != RT_ERROR_NONE, "Destroy hccl stream for rt_model failed")
    }
  }
  return;
}

void DavinciModel::ReleaseTask() {
  for (const auto &task : cpu_task_list_) {
    if (task != nullptr) {
      GE_CHK_STATUS(task->Release(), "[Release][Task] failed, model id:%u.", model_id_);
    }
  }
  cpu_task_list_.clear();

  for (const auto &task : task_list_) {
    if (task != nullptr) {
      GE_CHK_STATUS(task->Release(), "[Release][Task] failed, model id:%u.", model_id_);
    }
  }

  for (auto &item : label_goto_args_) {
    GE_FREE_RT_LOG(item.second.first);
  }
  label_goto_args_.clear();
}

Status DavinciModel::Assign(const GeModelPtr &ge_model) {
  if (ge_model == nullptr) {
    GELOGI("can't assign null ge_model");
    return FAILED;
  }
  ge_model_ = ge_model;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Reduce memory usage after task sink.
/// @return: void
///
void DavinciModel::Shrink() {
  skt_info_ = {0, 0, 0, 0, nullptr, nullptr, {}, {}, {}, {}, {}, RT_KERNEL_DEFAULT, -1, 0, nullptr};
  DumperShrink();
  ge_model_.reset();  // delete object.
  op_list_.clear();
  ClearTaskAddrs();
}

Status DavinciModel::InitWeightMem(void *dev_ptr, void *weight_ptr, size_t weight_size) {
  if (is_weight_mem_has_inited_) {
    REPORT_INNER_ERROR("E19999", "Call InitWeightMem more than once, model_id:%u, check invalid", model_id_);
    GELOGE(FAILED, "[Check][Param] call InitWeightMem more than once, model id:%u.", model_id_);
    return FAILED;
  }
  is_weight_mem_has_inited_ = true;

  const Buffer &weights = ge_model_->GetWeight();
  std::size_t weights_size = weights.GetSize();
  GE_CHECK_LE(weights_size, ALLOC_MEMORY_MAX_SIZE);

  if ((weight_ptr != nullptr) && (weight_size < weights_size)) {
    REPORT_INNER_ERROR("E19999", "Param weight_ptr is nullptr or ge_model.weight.size:%zu < param weights_size:%zu, "
                       "model_id:%u, check invalid", weight_size, weights_size, model_id_);
    GELOGE(FAILED, "[Check][Param] Invalid mem param: weight_size=%zu totalsize=%zu, model_id:%u.",
           weight_size, weights_size, model_id_);
    return FAILED;
  }

  weights_mem_base_ = static_cast<uint8_t *>(dev_ptr);
  is_inner_weight_base_ = false;

  if (weights_size != 0) {
    weights_mem_base_ = static_cast<uint8_t *>(weight_ptr);
    is_inner_weight_base_ = false;
    if (weight_ptr == nullptr) {
      weights_mem_base_ = MallocWeightsMem(weights_size);
      if (weights_mem_base_ == nullptr) {
        REPORT_CALL_ERROR("E19999", "MallocWeightsMem fail, weights_size:%zu, model_id:%u, check invalid",
                          weights_size, model_id_);
        GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Alloc][Memory] for weight failed. size:%zu, model_id:%u",
               weights_size, model_id_);
        return ACL_ERROR_GE_MEMORY_ALLOCATION;
      }
      is_inner_weight_base_ = true;
    }
    GELOGI("[IMAS]InitWeightMem graph_%u MallocMemory type[W] memaddr[%p] mem_size[%zu]", runtime_param_.graph_id,
           weights_mem_base_, weights_size);
    GE_CHK_RT_RET(rtMemcpy(weights_mem_base_, weights_size, weights.GetData(), weights_size, RT_MEMCPY_HOST_TO_DEVICE));
    GELOGI("copy weights data to device");
  }

  runtime_param_.weight_base = weights_mem_base_;
  return SUCCESS;
}


Status DavinciModel::InitFeatureMapAndP2PMem(void *dev_ptr, size_t mem_size) {
  if (is_feature_map_mem_has_inited_) {
    REPORT_INNER_ERROR("E19999", "InitFeatureMapMem is called more than once, model_id:%u, check invalid", model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] InitFeatureMapMem is called more than once, model_id:%u", model_id_);
    return PARAM_INVALID;
  }
  is_feature_map_mem_has_inited_ = true;

  std::size_t data_size = TotalMemSize();

  if ((dev_ptr != nullptr) && (mem_size < TotalMemSize())) {
    REPORT_INNER_ERROR("E19999", "Param dev_ptr is nullptr or mem_size:%zu < ge_model.mem_size:%zu, "
                       "model_id:%u, check invalid", mem_size, TotalMemSize(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] Invalid mem param: mem_size=%zu totalsize=%zu, model_id:%u.",
           mem_size, TotalMemSize(), model_id_);
    return PARAM_INVALID;
  }

  mem_base_ = static_cast<uint8_t *>(dev_ptr);
  is_inner_mem_base_ = false;

  if (TotalMemSize() && mem_base_ == nullptr) {
    mem_base_ = MallocFeatureMapMem(data_size);
    if (mem_base_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "MallocFeatureMapMem fail, data_size:%zu, model_id:%u, check invalid",
                        data_size, model_id_);
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Alloc][Memory] for feature map failed. size:%zu, model_id:%u",
             data_size, model_id_);
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }
    GEEVENT("[IMAS]InitFeatureMapAndP2PMem graph_%u MallocMemory type[F] memaddr[%p] mem_size[%zu]",
            runtime_param_.graph_id, mem_base_, data_size);

    if (!is_inner_weight_base_) {
      weights_mem_base_ = mem_base_;
      is_inner_weight_base_ = true;
    }
    is_inner_mem_base_ = true;
  }

  if (!runtime_param_.memory_infos.empty()) {
    GE_CHK_STATUS_RET(MallocExMem(), "MallocExMem failed.");
  }

  GE_CHK_STATUS_RET(InitVariableMem(), "[Init][VariableMemory] failed, model_id:%u", model_id_);
  runtime_param_.mem_base = mem_base_;
  runtime_param_.weight_base = weights_mem_base_;
  return SUCCESS;
}

Status DavinciModel::InitVariableMem() {
  // malloc variable memory base
  var_mem_base_ = VarManager::Instance(session_id_)->GetVarMemoryBase(RT_MEMORY_HBM);
  if (TotalVarMemSize() && (var_mem_base_ == nullptr)) {
    Status ret = VarManager::Instance(session_id_)->MallocVarMemory(TotalVarMemSize());
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "MallocVarMemory fail, var_size:%zu, model_id:%u, check invalid",
                        TotalVarMemSize(), model_id_);
      GELOGE(ret, "[Malloc][VarMemory] failed, var_size:%zu, model_id:%u", TotalVarMemSize(), model_id_);
      return ret;
    }
    var_mem_base_ = VarManager::Instance(session_id_)->GetVarMemoryBase(RT_MEMORY_HBM);
    GEEVENT("[IMAS]InitVariableMem graph_%u MallocMemory type[V] memaddr[%p] mem_size[%zu]", runtime_param_.graph_id,
            var_mem_base_, TotalVarMemSize());
  }
  runtime_param_.var_base = var_mem_base_;
  return SUCCESS;
}

void DavinciModel::InitRuntimeParams() {
  int64_t value = 0;
  bool ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_MEMORY_SIZE, value);
  runtime_param_.mem_size = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_WEIGHT_SIZE, value);
  runtime_param_.weight_size = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_STREAM_NUM, value);
  runtime_param_.stream_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_EVENT_NUM, value);
  runtime_param_.event_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_LABEL_NUM, value);
  runtime_param_.label_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_BATCH_NUM, value);
  runtime_param_.batch_num = ret ? (uint32_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, MODEL_ATTR_TASK_GEN_BASE_ADDR, value);
  runtime_param_.logic_mem_base = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, value);
  runtime_param_.logic_weight_base = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ge::MODEL_ATTR_SESSION_ID, value);
  runtime_param_.session_id = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_TASK_GEN_VAR_ADDR, value);
  runtime_param_.logic_var_base = ret ? (uint64_t)value : 0;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_VAR_SIZE, value);
  runtime_param_.var_size = ret ? (uint64_t)value : 0;
  session_id_ = runtime_param_.session_id;
  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_P2P_MEMORY_SIZE, value);
  MemInfo p2p_mem_info;
  p2p_mem_info.memory_size = static_cast<size_t>(ret ? value : 0);
  p2p_mem_info.memory_type = RT_MEMORY_P2P_DDR;
  p2p_mem_info.memory_key = "_p";
  runtime_param_.memory_infos[RT_MEMORY_P2P_DDR] = std::move(p2p_mem_info);

  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE, value);
  MemInfo session_scope_mem_info;
  session_scope_mem_info.memory_size = static_cast<size_t>(ret ? value : 0);
  runtime_param_.memory_infos[kSessionScopeMemory | RT_MEMORY_HBM] = std::move(session_scope_mem_info);

  ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_ZERO_COPY_MEMORY_SIZE, value);
  runtime_param_.zero_copy_size = ret ? value : 0;
  GELOGI("InitRuntimeParams(), %s.", runtime_param_.ToString().c_str());
}

void DavinciModel::CheckHasHcomOp(const ComputeGraphPtr &compute_graph) {
  const set<string> hcom_opp_types({
      HCOMBROADCAST, HCOMALLGATHER, HCOMALLREDUCE, HCOMSEND, HCOMRECEIVE, HCOMREDUCESCATTER,
      HVDCALLBACKALLREDUCE, HVDCALLBACKALLGATHER, HVDCALLBACKBROADCAST, HVDWAIT, HCOMREDUCE
  });

  for (const auto &node : compute_graph->GetAllNodes()) {
    OpDescPtr op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, GELOGW("Node OpDesc is nullptr."); continue);
    if (hcom_opp_types.count(op_desc->GetType()) > 0) {
      uint32_t stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
      hcom_streams_.emplace(stream_id);
      GELOGD("hcom stream: %u.", stream_id);
    }
  }
}

///
/// @ingroup ge
/// @brief Make active stream list and bind to model.
/// @return: 0 for success / others for fail
///
Status DavinciModel::BindModelStream() {
  // Stream not in active_stream_indication_ is active stream.
  is_stream_list_bind_ = false;
  if ((!input_queue_ids_.empty() || !output_queue_ids_.empty()) || (deploy_type_ == AICPU_DEPLOY_CROSS_THREAD)) {
    for (size_t i = 0; i < stream_list_.size(); ++i) {
      if (active_stream_indication_.count(i) == 0) {
        active_stream_list_.push_back(stream_list_[i]);
        active_stream_indication_.insert(i);  // deactive all model stream.
      }
    }
  }

  for (size_t i = 0; i < stream_list_.size(); ++i) {
    if (active_stream_indication_.count(i) > 0) {
      GELOGI("rtModelBindStream[%zu]", i);
      GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, stream_list_[i], RT_INVALID_FLAG));
    } else {
      // bind rt_model_handel to all streams that relates to op
      GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, stream_list_[i], RT_HEAD_STREAM));
    }
  }
  is_stream_list_bind_ = true;
  return SUCCESS;
}

Status DavinciModel::DoTaskSink() {
  // task sink is supported as model_task_def is set
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  if (model_task_def == nullptr) {
    return SUCCESS;
  }

  GE_CHK_RT_RET(rtGetAicpuDeploy(&deploy_type_));
  GELOGI("Do task sink. AiCpu deploy type is: %x.", deploy_type_);

  GE_CHK_STATUS_RET(BindModelStream(), "[Bind][ModelStream] failed, model_id:%u.", model_id_);

  if (known_node_) {
    GE_CHK_STATUS_RET(MallocKnownArgs(), "[Malloc][KnownArgs] failed, model_id:%u.", model_id_);
  }

  GE_CHK_STATUS_RET(InitTaskInfo(*model_task_def.get()), "[Init][TaskInfo] failed, model_id:%u.", model_id_);

  GE_CHK_STATUS_RET(ModelManager::GetInstance()->LaunchCustAicpuSo(),
                    "[Launch][CustAicpuSo] failed, model_id:%u.", model_id_);

  GE_CHK_STATUS_RET(ModelManager::GetInstance()->CheckAicpuOpList(ge_model_),
                    "[Check][AicpuOpList] failed, model_id:%u.", model_id_);

  GE_CHK_STATUS_RET(InitEntryTask(), "[Init][EntryTask] failed, model_id:%u.", model_id_);

  GE_CHK_STATUS_RET(InitL1DataDumperArgs(), "[Init][L1DataDumperArgs] failed, model_id:%u.", model_id_);

  GE_CHK_STATUS_RET(DistributeTask(), "[Distribute][Task] failed, model_id:%u.", model_id_);

  GE_CHK_RT_RET(rtModelLoadComplete(rt_model_handle_));

  SetCopyOnlyOutput();
  return SUCCESS;
}

// set device use aicore(0) or vectorcore(1)
Status DavinciModel::SetTSDevice() {
  int64_t value = 0;
  bool ret = ge::AttrUtils::GetInt(ge_model_, ATTR_MODEL_CORE_TYPE, value);
  uint32_t core_type = ret ? static_cast<uint32_t>(value) : 0;
  GELOGD("Set TSDevice: %u.", core_type);
  rtError_t rt_ret = rtSetTSDevice(core_type);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtSetTSDevice failed, core_type:%u, model_id:%u", core_type, model_id_);
    GELOGE(RT_FAILED, "[Set][TSDevice] failed, core_type:%u, model_id:%u, ret: 0x%X", core_type, model_id_, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

Status DavinciModel::OpDebugRegister() {
  if (GetDumpProperties().IsOpDebugOpen()) {
    uint32_t op_debug_mode = GetDumpProperties().GetOpDebugMode();
    auto ret = opdebug_register_.RegisterDebugForModel(rt_model_handle_, op_debug_mode, data_dumper_);
    if (ret != SUCCESS) {
      GELOGE(ret,"[Call][RegisterDebugForModel] Register known shape op debug failed, ret: 0x%X", ret);
      return ret;
    }
    is_op_debug_reg_ = true;
  }
  return SUCCESS;
}

void DavinciModel::OpDebugUnRegister() {
  if (is_op_debug_reg_) {
    opdebug_register_.UnregisterDebugForModel(rt_model_handle_);
    is_op_debug_reg_ = false;
  }
  return;
}

// initialize op sequence and call initialization function of each op respectively
Status DavinciModel::Init(void *dev_ptr, size_t mem_size, void *weight_ptr, size_t weight_size) {
  // validating params
  GELOGI("Priority is %d.", priority_);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(priority_ < 0 || priority_ > 7, return PARAM_INVALID,
                                 "[Check][Param] Priority must between 0-7, now is %d.", priority_);
  GE_CHK_BOOL_RET_STATUS(ge_model_ != nullptr, PARAM_INVALID, "[Check][Param] GeModel is null.");
  Graph graph = ge_model_->GetGraph();
  ComputeGraphPtr compute_graph = GraphUtils::GetComputeGraph(graph);
  GE_CHK_BOOL_RET_STATUS(compute_graph != nullptr, INTERNAL_ERROR, "[Get][ComputeGraph] failed, ret is nullptr.");

  // Initializing runtime_param_
  InitRuntimeParams();

  // RTS set aicore or vectorcore
  GE_CHK_STATUS_RET(SetTSDevice(), "[Set][TSDevice] failed, graph:%s.", compute_graph->GetName().c_str());

  version_ = ge_model_->GetVersion();
  name_ = ge_model_->GetName();
  (void)ge::AttrUtils::GetBool(ge_model_, ATTR_NAME_SWITCH_FOR_L1_FUSION, is_l1_fusion_enable_);
  GELOGD("The value of ge.l1Fusion in ge_model is %d.", is_l1_fusion_enable_);
  CheckHasHcomOp(compute_graph);

  vector<int64_t> huge_stream_list;
  (void)ge::AttrUtils::GetListInt(ge_model_, ATTR_MODEL_HUGE_STREAM_LIST, huge_stream_list);
  std::set<int64_t> huge_streams(huge_stream_list.begin(), huge_stream_list.end());

  for (uint32_t i = 0; i < StreamNum(); i++) {
    rtStream_t stream = nullptr;
    GE_MAKE_GUARD_RTSTREAM(stream);

    uint32_t stream_flags = RT_STREAM_PERSISTENT;
    if (huge_streams.find(i) != huge_streams.end()) {
      GELOGI("Stream %u is huge stream.", i);
      stream_flags |= RT_STREAM_HUGE;
    }

    if (hcom_streams_.find(i) != hcom_streams_.end()) {
      GE_CHK_RT_RET(rtStreamCreateWithFlags(&stream, priority_, stream_flags | RT_STREAM_FORCE_COPY));
    } else {
      GE_CHK_RT_RET(rtStreamCreateWithFlags(&stream, priority_, stream_flags));
    }

    GE_DISMISS_GUARD(stream);
    stream_list_.push_back(stream);
    int32_t rt_stream_id = kInvalidStream;
    (void)rtGetStreamId(stream, &rt_stream_id);
    GELOGI("Logical stream index:%u, stream:%p, rtstream: %d.", i, stream, rt_stream_id);
  }

  uint32_t event_num = EventNum();
  uint32_t create_flag = static_cast<uint32_t>((event_num > kEventReuseThreshold) ? RT_EVENT_WITH_FLAG :
                                                                                    RT_EVENT_DEFAULT);
  for (uint32_t i = 0; i < event_num; ++i) {
    rtEvent_t rt_event = nullptr;
    GE_CHK_RT_RET(rtEventCreateWithFlag(&rt_event, create_flag));
    event_list_.push_back(rt_event);
  }

  label_list_.resize(LabelNum(), nullptr);

  // create model_handle to load model
  GE_CHK_RT_RET(rtModelCreate(&rt_model_handle_, 0));
  GE_CHK_RT_RET(rtModelGetId(rt_model_handle_, &runtime_model_id_));

  // inference will use default graph_id 0;
  runtime_param_.graph_id = compute_graph->GetGraphID();

  // op debug register
  GE_CHK_STATUS_RET(OpDebugRegister(), "[Call][OpDebugRegister] failed, model_id:%u.", model_id_);

  GE_TIMESTAMP_START(TransAllVarData);
  GE_CHK_STATUS_RET(TransAllVarData(compute_graph, runtime_param_.graph_id),
                    "[Call][TransAllVarData] failed, graph:%s, graph_id:%u.",
                    compute_graph->GetName().c_str(), runtime_param_.graph_id);
  GE_TIMESTAMP_END(TransAllVarData, "GraphLoader::TransAllVarData");
  GE_CHK_STATUS_RET(TransVarDataUtils::CopyVarData(compute_graph, session_id_, device_id_),
                    "[Copy][VarData] failed, graph:%s, session_id:%lu, device_id:%u",
                    compute_graph->GetName().c_str(), session_id_, device_id_);

  GE_TIMESTAMP_START(InitModelMem);
  GELOGD("Known node is %d.", known_node_);
  GE_CHK_STATUS_RET_NOLOG(InitWeightMem(dev_ptr, weight_ptr, weight_size));
  if (!known_node_) {
    GE_CHK_STATUS_RET_NOLOG(InitFeatureMapAndP2PMem(dev_ptr, mem_size));
    data_inputer_ = new (std::nothrow) DataInputer();
    GE_CHK_BOOL_RET_STATUS(data_inputer_ != nullptr, MEMALLOC_FAILED,
                           "[Create][DataInputer] data_inputer_ is nullptr");
  }
  fixed_mem_base_ = reinterpret_cast<uintptr_t>(mem_base_);
  GE_TIMESTAMP_END(InitModelMem, "GraphLoader::InitModelMem");

  for (const ge::NodePtr &node : compute_graph->GetDirectNode()) {
    auto op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
    GE_IF_BOOL_EXEC(op_desc->GetType() != VARIABLE, continue);
    GE_IF_BOOL_EXEC(IsBroadCastOpData(node),
                    (void)ge::AttrUtils::SetStr(op_desc, VAR_ATTR_VAR_IS_BROADCAST, "var_is_restore"););
  }

  GE_CHK_STATUS_RET(InitNodes(compute_graph), "[Init][Nodes] failed, graph:%s.", compute_graph->GetName().c_str());

  GE_TIMESTAMP_START(DoTaskSink);
  GE_CHK_STATUS_RET(DoTaskSink(), "[Call][DoTaskSink] failed, model_id:%u.", model_id_);
  GE_TIMESTAMP_END(DoTaskSink, "GraphLoader::DoTaskSink");

  /// In zero copy model, if a aicpu operator is connected to the first or last layer, before model execution,
  /// the aicpu opertor needs to destroy history record, and update operator memory address.
  /// The model with specified aicpu operators is only marked here, and destruction is in ModelManager::ExecuteModel().
  need_destroy_aicpu_kernel_ = IsAicpuKernelConnectSpecifiedLayer();

  string fp_ceiling_mode;
  if (ge::AttrUtils::GetStr(ge_model_, ATTR_FP_CEILING_MODE, fp_ceiling_mode)) {
    GELOGI("Get attr ATTR_FP_CEILING_MODE from model, value is %s.", fp_ceiling_mode.c_str());
    // mode 0: Do not perform saturation processing. By default, IEEE754 is used.
    GE_CHK_RT_RET(rtSetCtxINFMode((fp_ceiling_mode != "0")));
  }

  SetProfileTime(MODEL_LOAD_END);
  // collect profiling for ge
  auto &profiling_manager = ProfilingManager::Instance();
  if (profiling_manager.ProfilingModelLoadOn()) {
    GE_CHK_STATUS_RET(InitModelProfile(), "[Init][ModelProfile] failed, model_id:%u.", model_id_);
    Status p_ret = ReportProfilingData();
    if (p_ret != SUCCESS) {
      GELOGE(p_ret, "[Report][ProfilingData] failed, ret:%d, model_id:%u.", p_ret, model_id_);
      return p_ret;
    }
  }

  Shrink();
  return SUCCESS;
}

// save specify attr values of op, such as ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES
// it will save more attr values in the future
void DavinciModel::SaveSpecifyAttrValues(const OpDescPtr &op_desc) {
  std::vector<std::string> value;
  if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, value)) {
    std::map<std::string, std::vector<std::string>> attr_name_to_value;
    attr_name_to_value[ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES]= value;
    op_name_to_attrs_[op_desc->GetName()] = attr_name_to_value;
    GELOGD("Get op:%s attr:%s success.", op_desc->GetName().c_str(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES.c_str());
  }
  return;
}

Status DavinciModel::ReportProfilingData() {
  bool is_train = domi::GetContext().train_flag;
  auto model_id = model_id_;
  auto &profiling_manager = ProfilingManager::Instance();
  auto graph_id = runtime_param_.graph_id;
  if (is_train) {
    GELOGD("Replace model_id:%u with graph_id:%u, when training.", model_id, graph_id);
    model_id = graph_id;
  }
  profiling_manager.ReportProfilingData(model_id, GetTaskDescInfo());
  GE_CHK_STATUS(SinkModelProfile(), "[Sink][ModelProfile] failed, model_id:%u.", model_id);

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Travel all nodes and determine if destruction is required.
/// @return bool
///
bool DavinciModel::IsAicpuKernelConnectSpecifiedLayer() {
  Graph graph = ge_model_->GetGraph();
  ComputeGraphPtr compute_graph = GraphUtils::GetComputeGraph(graph);
  auto all_nodes = compute_graph->GetAllNodes();
  for (auto &node : all_nodes) {
    GE_IF_BOOL_EXEC(node == nullptr, continue);
    OpDescPtr op_desc = node->GetOpDesc();
    GE_IF_BOOL_EXEC(op_desc == nullptr, continue);

    int64_t imply_type = -1;
    (void)ge::AttrUtils::GetInt(op_desc, ATTR_NAME_IMPLY_TYPE, imply_type);
    if (imply_type != static_cast<int64_t>(domi::ImplyType::AI_CPU)) {
      continue;
    }
    GELOGD("Current operator imply type is %ld, name is %s.", imply_type, op_desc->GetName().c_str());

    for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
      GE_IF_BOOL_EXEC(in_data_anchor == nullptr, continue);
      auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      GE_IF_BOOL_EXEC(peer_out_data_anchor == nullptr, continue);
      auto peer_node = peer_out_data_anchor->GetOwnerNode();
      GE_IF_BOOL_EXEC(peer_node == nullptr, continue);
      auto peer_op_desc = peer_node->GetOpDesc();
      GE_IF_BOOL_EXEC(peer_op_desc == nullptr, continue);
      if (IsDataOp(peer_op_desc->GetType())) {
        GELOGI("Mark specified aicpu operator connected to data.");
        return true;
      }
    }
    for (auto &out_data_anchor : node->GetAllOutDataAnchors()) {
      GE_IF_BOOL_EXEC(out_data_anchor == nullptr, continue);
      auto peer_in_data_anchors = out_data_anchor->GetPeerInDataAnchors();
      for (auto &peer_in_data_anchor : peer_in_data_anchors) {
        GE_IF_BOOL_EXEC(peer_in_data_anchor == nullptr, continue);
        auto peer_node = peer_in_data_anchor->GetOwnerNode();
        GE_IF_BOOL_EXEC(peer_node == nullptr, continue);
        auto peer_op_desc = peer_node->GetOpDesc();
        GE_IF_BOOL_EXEC(peer_op_desc == nullptr, continue);
        if (peer_op_desc->GetType() == NETOUTPUT) {
          GELOGI("Mark specified aicpu operator connected to netoutput.");
          return true;
        }
      }
    }
  }

  return false;
}

Status DavinciModel::UpdateSessionId(uint64_t session_id) {
  GE_CHECK_NOTNULL(ge_model_);
  if (!AttrUtils::SetInt(ge_model_, MODEL_ATTR_SESSION_ID, static_cast<int64_t>(session_id))) {
    GELOGW("Set attr[%s] failed in updating session_id.", MODEL_ATTR_SESSION_ID.c_str());
  }

  GELOGD("Update session id: %lu.", session_id);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Travel all nodes and do some init.
/// @param [in] compute_graph: ComputeGraph to load.
/// @return Status
///
Status DavinciModel::InitNodes(const ComputeGraphPtr &compute_graph) {
  uint32_t data_op_index = 0;
  GE_TIMESTAMP_CALLNUM_START(LoadTBEKernelBinToOpDesc);
  GE_TIMESTAMP_CALLNUM_START(InitTbeHandle);

  typedef Status (DavinciModel::*OpDescCall)(const OpDescPtr &);
  static std::map<std::string, OpDescCall> op_desc_handle = {
      {CONSTANTOP, &DavinciModel::InitConstant},
      {STREAMACTIVE, &DavinciModel::InitStreamActive},
      {STREAMSWITCH, &DavinciModel::InitStreamSwitch},
      {STREAMSWITCHN, &DavinciModel::InitStreamSwitchN},
      {LABELSET, &DavinciModel::InitLabelSet},
      {CASE, &DavinciModel::InitCase},
  };

  vector<OpDescPtr> output_op_list;
  set<const void *> input_outside_addrs;
  set<const void *> output_outside_addrs;
  map<uint32_t, OpDescPtr> data_by_index;
  map<string, OpDescPtr> variable_by_name;
  auto nodes = compute_graph->GetAllNodes();
  const CustAICPUKernelStore &aicpu_kernel_store = ge_model_->GetCustAICPUKernelStore();
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto &node = nodes.at(i);
    const auto &op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    SaveSpecifyAttrValues(op_desc);
    op_list_[op_desc->GetId()] = op_desc;

    GE_TIMESTAMP_RESTART(LoadTBEKernelBinToOpDesc);
    aicpu_kernel_store.LoadCustAICPUKernelBinToOpDesc(op_desc);
    GE_TIMESTAMP_ADD(LoadTBEKernelBinToOpDesc);

    if (IsDataOp(op_desc->GetType())) {
      if (InitDataOp(compute_graph, node, data_op_index, data_by_index, input_outside_addrs) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Init][DataOp] failed, Name:%s", op_desc->GetName().c_str());
        return PARAM_INVALID;
      }
      data_dumper_.SaveDumpInput(node);
      continue;
    }

    if (op_desc->GetType() == NETOUTPUT) {
      if (InitNetOutput(compute_graph, node, output_op_list, output_outside_addrs) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Init][NetOutput] failed, Name:%s", op_desc->GetName().c_str());
        return PARAM_INVALID;
      }
      if (InitRealSizeAndShapeInfo(compute_graph, node) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Init][RealSizeAndShapeInfo] failed, Name:%s", op_desc->GetName().c_str());
        return PARAM_INVALID;
      }
      continue;
    }

    if (op_desc->GetType() == VARIABLE) {
      if (InitVariable(op_desc, variable_by_name) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Init][Variable] failed, Name:%s", op_desc->GetName().c_str());
        return PARAM_INVALID;
      }
      continue;
    }

    // for dynamic shape with control flow
    SetLabelForDynamic(node);
    auto it = op_desc_handle.find(op_desc->GetType());
    if (it != op_desc_handle.end()) {
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG((this->*it->second)(op_desc) != SUCCESS, return PARAM_INVALID,
                                     "[Init][Node] failed, Name:%s", op_desc->GetName().c_str());
      continue;
    }

    if (IsNoTaskAndDumpNeeded(op_desc)) {
      GELOGD("node[%s] without task, and save op_desc and addr for dump", op_desc->GetName().c_str());
      const RuntimeParam &rts_param = GetRuntimeParam();
      const vector<void *> input_data_addrs = ModelUtils::GetInputDataAddrs(rts_param, op_desc);
      const vector<void *> output_data_addrs = ModelUtils::GetOutputDataAddrs(rts_param, op_desc);
      const vector<void *> workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrs(rts_param, op_desc);
      vector<void *> tensor_device_addrs;
      tensor_device_addrs.insert(tensor_device_addrs.end(), input_data_addrs.begin(), input_data_addrs.end());
      tensor_device_addrs.insert(tensor_device_addrs.end(), output_data_addrs.begin(), output_data_addrs.end());
      tensor_device_addrs.insert(tensor_device_addrs.end(), workspace_data_addrs.begin(), workspace_data_addrs.end());
      void *addr = nullptr;
      auto size = kAddrLen * tensor_device_addrs.size();
      GE_CHK_RT_RET(rtMalloc(&addr, size, RT_MEMORY_HBM));

      rtError_t rt_ret = rtMemcpy(addr, size, tensor_device_addrs.data(), size, RT_MEMCPY_HOST_TO_DEVICE);
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", size, rt_ret);
        GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", size, rt_ret);
        GE_CHK_RT(rtFree(addr));
        return RT_ERROR_TO_GE_STATUS(rt_ret);
      }
      saved_task_addrs_.emplace(op_desc, addr);
    }

    GE_TIMESTAMP_RESTART(InitTbeHandle);
    if (IsTbeTask(op_desc)) {
      Status status =
          op_desc->HasAttr(ATTR_NAME_THREAD_SCOPE_ID) ? InitTbeHandleWithFfts(op_desc) : InitTbeHandle(op_desc);
      if (status != SUCCESS) {
        GELOGE(status, "[Init][TbeHandle] failed. op:%s", op_desc->GetName().c_str());
        return status;
      }
    }
    GE_TIMESTAMP_ADD(InitTbeHandle);
  }

  SetDataDumperArgs(compute_graph, variable_by_name);
  GE_TIMESTAMP_CALLNUM_END(LoadTBEKernelBinToOpDesc, "GraphLoader::LoadTBEKernelBinToOpDesc.");
  GE_TIMESTAMP_CALLNUM_END(InitTbeHandle, "GraphLoader::InitTbeHandle.");
  return GenInputOutputInfo(data_by_index, output_op_list);
}

void DavinciModel::SetLabelForDynamic(const NodePtr &node) {
  if (known_node_ && (node->GetType() == LABELSWITCHBYINDEX || node->GetType() == STREAMSWITCH)) {
    for (auto &in_data_anchor : node->GetAllInDataAnchors()) {
      auto peer_out_data_anchor = in_data_anchor->GetPeerOutAnchor();
      if (peer_out_data_anchor != nullptr) {
        // name+index as the label of switch input
        string tensor_name = node->GetName() + std::to_string(in_data_anchor->GetIdx());
        auto peer_node = peer_out_data_anchor->GetOwnerNode();
        (void)AttrUtils::SetStr(peer_node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_FIXED_ADDR, tensor_name);
        (void)AttrUtils::SetInt(peer_node->GetOpDesc(), ATTR_DYNAMIC_SHAPE_FIXED_ADDR_INDEX, 0);
        tensor_name_to_peer_output_index_[tensor_name] = 0;
      }
    }
  }
}

///
/// @ingroup ge
/// @brief Data Op Initialize.
/// @param [in] ComputeGraphPtr: root graph of the model.
/// @param [in] NodePtr: Data Op.
/// @param [in/out] data_op_index: index of courrent count.
/// @param [in/out] data_by_index: Data ordered by index.
/// @return Status
///
Status DavinciModel::InitDataOp(const ComputeGraphPtr &graph, const NodePtr &node, uint32_t &data_op_index,
                                map<uint32_t, OpDescPtr> &data_by_index, set<const void *> &input_outside_addrs) {
  // op_desc Checked by Init: Data, valid.
  auto op_desc = node->GetOpDesc();
  if (node->GetOwnerComputeGraph() != graph) {
    GELOGI("Skip Data node: %s in subgraph.", op_desc->GetName().c_str());
    return SUCCESS;
  }

  auto data_index = data_op_index++;
  const auto &index_attr = GraphUtils::FindRootGraph(graph) == graph ? ATTR_NAME_INDEX : ATTR_NAME_PARENT_NODE_INDEX;
  if (AttrUtils::GetInt(op_desc, index_attr, data_index)) {
    GELOGD("Get new index %u, old %u", data_index, data_op_index - 1);
  }
  GELOGI("Init data node: %s, index: %u.", op_desc->GetName().c_str(), data_index);

  data_by_index[data_index] = op_desc;
  if (known_node_) {
    return SUCCESS;
  }

  // Make information for copy input data.
  const vector<int64_t> output_size_list = ModelUtils::GetOutputSize(op_desc);
  const vector<void *> virtual_addr_list = ModelUtils::GetOutputDataAddrs(runtime_param_, op_desc);
  const vector<int64_t> output_offset_list = op_desc->GetOutputOffset();
  if (output_size_list.empty() || virtual_addr_list.empty() || (output_size_list.size() != virtual_addr_list.size()) ||
      (output_offset_list.size() != virtual_addr_list.size())) {
    REPORT_INNER_ERROR(
        "E19999", "Check data fail in op:%s(%s), output_desc size:%zu output addr size:%zu output offset size:%zu "
        "not equal or has empty, model_id:%u",
        op_desc->GetName().c_str(), op_desc->GetType().c_str(),
        output_size_list.size(), virtual_addr_list.size(), output_offset_list.size(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] Data[%s] init failed: output size is %zu, "
           "virtual_addr size is %zu, offset size is %zu.", op_desc->GetName().c_str(), output_size_list.size(),
           virtual_addr_list.size(), output_offset_list.size());
    return PARAM_INVALID;
  }

  bool fusion_flag = false;
  ZeroCopyOffset zero_copy_offset;
  int64_t data_size = output_size_list[kDataIndex];
  void *virtual_addr = virtual_addr_list[kDataIndex];
  Status ret = zero_copy_offset.InitInputDataInfo(data_size, virtual_addr, op_desc, fusion_flag);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Init][DataInfo] of input_info %s failed.", op_desc->GetName().c_str());
    return PARAM_INVALID;
  }
  if (input_outside_addrs.count(virtual_addr) == 0) {
    int64_t output_offset = output_offset_list.at(kDataIndex);
    zero_copy_offset.SetInputOutsideAddrs(output_offset, virtual_addr, fusion_flag, real_virtual_addrs_);
    input_outside_addrs.insert(virtual_addr);
  }
  input_data_info_[data_index] = zero_copy_offset;

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Sort Data op list by index.
/// @param [in] data_by_index: map of Data Op.
/// @param [in] output_op_list: list of NetOutput op.
/// @return Status
///
Status DavinciModel::GenInputOutputInfo(const map<uint32_t, OpDescPtr> &data_by_index,
                                        const vector<OpDescPtr> &output_op_list) {
  GELOGD("Data node size: %zu, NetOutput node size: %zu", data_by_index.size(), output_op_list.size());
  for (auto &item : data_by_index) {
    const auto output_addrs = ModelUtils::GetOutputDataAddrs(runtime_param_, item.second);
    GELOGD("Data node is: %s, output addr size: %zu", item.second->GetName().c_str(), output_addrs.size());
    input_addrs_list_.emplace_back(output_addrs);

    GE_CHK_STATUS_RET(InitAippInfo(item.first, item.second),
                      "[Init][AippInfo] failed, node:%s", item.second->GetName().c_str());
    GE_CHK_STATUS_RET(InitAippType(item.first, item.second, data_by_index),
                      "[Init][AippType] failed, node:%s", item.second->GetName().c_str());
    GE_CHK_STATUS_RET(InitOrigInputInfo(item.first, item.second),
                      "[Init][OrigInputInfo] failed, node:%s", item.second->GetName().c_str());
    GE_CHK_STATUS_RET(InitAippInputOutputDims(item.first, item.second),
                      "[Init][AippInputOutputDims] failed, node:%s", item.second->GetName().c_str());
    GE_CHK_STATUS_RET(InitInputDescInfo(item.second),
                      "[Init][InputDescInfo] failed, node:%s", item.second->GetName().c_str());
    if (item.second->GetType() == AIPP_DATA_TYPE) {
      GELOGI("This is dynamic aipp model, Node: %s", item.second->GetName().c_str());
      is_dynamic_aipp_ = true;
    }
  }

  vector<string> out_node_name;
  (void)AttrUtils::GetListStr(ge_model_, ATTR_MODEL_OUT_NODES_NAME, out_node_name);
  GELOGD("Output node size: %zu, out nodes name is: %zu", output_op_list.size(), out_node_name.size());
  for (const auto &op_desc : output_op_list) {
    const auto input_addrs = ModelUtils::GetInputDataAddrs(runtime_param_, op_desc);
    GELOGD("NetOutput node is: %s, input addr size: %zu", op_desc->GetName().c_str(), input_addrs.size());
    output_addrs_list_.emplace_back(input_addrs);

    bool getnext_sink_dynamic = false;
    if (AttrUtils::GetBool(op_desc, ATTR_GETNEXT_SINK_DYNMAIC, getnext_sink_dynamic) && getnext_sink_dynamic) {
      GELOGI("ATTR_GETNEXT_SINK_DYNMAIC has been set and is true, node: %s", op_desc->GetName().c_str());
      is_getnext_sink_dynamic_ = true;
    }

    vector<string> shape_info;
    if (AttrUtils::GetListStr(op_desc, ATTR_NAME_DYNAMIC_OUTPUT_DIMS, shape_info)) {
      dynamic_output_shape_info_.insert(dynamic_output_shape_info_.end(), shape_info.begin(), shape_info.end());
    }

    if (InitOutputTensorInfo(op_desc) != SUCCESS) {
      return INTERNAL_ERROR;
    }

    GE_CHK_STATUS_RET(InitOutputDescInfo(op_desc, out_node_name),
                      "[Init][OutputDescInfo] failed, node:%s", op_desc->GetName().c_str());
  }

  return SUCCESS;
}

bool DavinciModel::IsGetNextSinkDynamic(const OpDescPtr &op_desc) {
  bool getnext_sink_dynamic = false;
  if (ge::AttrUtils::GetBool(op_desc, ATTR_GETNEXT_SINK_DYNMAIC, getnext_sink_dynamic) && getnext_sink_dynamic) {
    GELOGI("ATTR_GETNEXT_SINK_DYNMAIC has been set and is true.");
    return true;
  }
  return false;
}

/// @ingroup ge
/// @brief NetOutput Op Initialize.
/// @param [in] ComputeGraphPtr: root graph of the model.
/// @param [in] NodePtr: NetOutput Op.
/// @param [in/out] vector<OpDescPtr>: All NetOutput node in model.
/// @return Status
Status DavinciModel::InitNetOutput(const ComputeGraphPtr &graph, const NodePtr &node,
                                   vector<OpDescPtr> &output_op_list, set<const void *> &output_outside_addrs) {
  // node->GetOpDesc Checked by Init: NetOutput, valid.
  auto op_desc = node->GetOpDesc();
  // excludes the function op sub graph, e.g. case,if
  if (node->GetOwnerComputeGraph() != graph) {
    GELOGI("Skip subgraph NetOutput node: %s.", op_desc->GetName().c_str());
    op_list_.erase(op_desc->GetId());
    return SUCCESS;
  }

  GELOGI("Init NetOutput node: %s.", op_desc->GetName().c_str());
  output_op_list.push_back(op_desc);
  has_output_node_ = true;
  if (known_node_) {
    return SUCCESS;
  }

  // Make information for copy output data.
  const vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
  const vector<void *> virtual_addr_list = ModelUtils::GetInputDataAddrs(runtime_param_, op_desc);
  const vector<int64_t> input_offset_list = op_desc->GetInputOffset();
  GE_IF_BOOL_EXEC(input_offset_list.size() != virtual_addr_list.size(),
                  REPORT_INNER_ERROR("E19999", "Check data fail in op:%s(%s), input addr size:%zu "
                                     "input offset size:%zu not equal, model_id:%u", op_desc->GetName().c_str(),
                                     op_desc->GetType().c_str(), virtual_addr_list.size(), input_offset_list.size(),
                                     model_id_);
                  GELOGE(PARAM_INVALID, "[Check][Param] virtual_addr size:%zu should be equal to offset size:%zu, "
                         "op:%s(%s), model id:%u", virtual_addr_list.size(), input_offset_list.size(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
                  return PARAM_INVALID;);
  if (input_size_list.empty() && virtual_addr_list.empty()) {
    GELOGI("NetOutput[%s] is empty.", op_desc->GetName().c_str());
    return SUCCESS;
  }
  if (input_size_list.empty() || input_size_list.size() != virtual_addr_list.size()) {
    REPORT_INNER_ERROR("E19999", "Check data fail in op:%s(%s), input_desc size:%zu input addr size:%zu "
                       "not equal or has empty, model_id:%u", op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       input_size_list.size(), virtual_addr_list.size(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] NetOutput[%s] init failed: Input size is %zu, Input addr is %zu",
           op_desc->GetName().c_str(), input_size_list.size(), virtual_addr_list.size());
    return PARAM_INVALID;
  }

  size_t num = output_data_info_.size();

  size_t input_count = input_size_list.size();
  is_getnext_sink_dynamic_ = false;
  if (IsGetNextSinkDynamic(op_desc)) {
    input_count = input_size_list.size() - kGetDynamicDimsCount;
    is_getnext_sink_dynamic_ = true;
  }
  for (size_t idx = 0; idx < input_count; ++idx) {
    ZeroCopyOffset zero_copy_offset;
    bool fusion_flag = false;
    Status ret = zero_copy_offset.InitOutputDataInfo(input_size_list, virtual_addr_list, op_desc, idx, fusion_flag);
    GE_IF_BOOL_EXEC(ret != SUCCESS,
                    GELOGE(PARAM_INVALID, "[Init][DataInfo] of input_info %s failed.", op_desc->GetName().c_str());
                    return PARAM_INVALID;);
    void *addr = virtual_addr_list.at(idx);
    int64_t input_offset = input_offset_list.at(idx);
    if (output_outside_addrs.count(addr) == 0) {
      vector<void *> tensor_addrs;
      zero_copy_offset.SetOutputOutsideAddrs(input_offset, fusion_flag, addr, tensor_addrs);
      output_outside_addrs.insert(addr);
      for (size_t i = 0; i < tensor_addrs.size(); ++i) {
        void *real_addr = tensor_addrs.at(i);
        DisableZeroCopy(real_addr);
        real_virtual_addrs_.insert(real_addr);
      }
    } else {
      GELOGI("same output_tensor_addr %p to different input_tensor of %s", addr, op_desc->GetName().c_str());
      DisableZeroCopy(addr);
    }
    output_data_info_[num + idx] = zero_copy_offset;
  }
  return SUCCESS;
}

Status DavinciModel::InitRealSizeAndShapeInfo(const ComputeGraphPtr &compute_graph, const NodePtr &node) {
  if (node->GetName().find(kMultiBatchNodePostfix) != string::npos) {
    GELOGD("No need to get size and shape of netoutput in subgraph.");
    return SUCCESS;
  }
  GELOGD("Start to initialize real size and shape info of %s.", node->GetName().c_str());
  GetAllGearsInfo(node);
  if (is_getnext_sink_dynamic_) {
    GE_IF_BOOL_EXEC(GetGetDynamicDimsNodeInfo(node) != SUCCESS,
                    GELOGE(PARAM_INVALID, "[Get][Info] of getdynamicdims node:%s failed.", node->GetName().c_str());
                    return PARAM_INVALID;);
  }
  if (is_online_infer_dynamic_) {
    GE_IF_BOOL_EXEC(GetGearAndRealOutSizeInfo(compute_graph, node) != SUCCESS,
                    GELOGE(PARAM_INVALID, "[Call][GetGearAndRealOutSizeInfo] failed, node:%s.",
                           node->GetName().c_str());
                    return PARAM_INVALID;);
    GE_IF_BOOL_EXEC(GetGearAndRealOutShapeInfo(compute_graph, node) != SUCCESS,
                    GELOGE(PARAM_INVALID, "[Call][GetGearAndRealOutShapeInfo] failed, node:%s.",
                           node->GetName().c_str());
                    return PARAM_INVALID;);
  }

  return SUCCESS;
}

void DavinciModel::GetAllGearsInfo(const NodePtr &node) {
  is_online_infer_dynamic_ = false;
  all_gears_info_.clear();
  std::string shapes;
  (void) AttrUtils::GetStr(node->GetOpDesc(), ATTR_ALL_GEARS_INFO, shapes);
  if (!shapes.empty()) {
    is_online_infer_dynamic_ = true;
    std::vector<std::string> shape_strs = ge::StringUtils::Split(shapes, ';');
    for (const auto &shape_str : shape_strs) {
      if (shape_str.empty()) {
        continue;
      }
      std::vector<int32_t> gear_info;
      std::vector<std::string> dims = ge::StringUtils::Split(shape_str, ',');
      for (const auto &dim : dims) {
        if (dim.empty()) {
          continue;
        }
        gear_info.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimal));
      }
      if (!gear_info.empty()) {
        all_gears_info_.emplace_back(gear_info);
        GELOGD("Init all gears info from %s, gear info is %s", node->GetName().c_str(),
               formats::JoinToString(gear_info).c_str());
      }
    }
  }
}

Status DavinciModel::GetGetDynamicDimsNodeInfo(const NodePtr &node) {
  GE_CHECK_NOTNULL(node->GetOpDesc());
  size_t input_count = node->GetAllInDataAnchors().size();
  GELOGI("input_anchor count of %s is %zu.", node->GetName().c_str(), input_count);
  size_t get_dynamic_dims_index = input_count - kGetDynamicDimsCount;
  auto in_anchor = node->GetAllInDataAnchors().at(get_dynamic_dims_index);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "In anchor index:%zu in op:%s(%s) peer anchor is nullptr, model_id:%u, check invalid",
                       get_dynamic_dims_index, node->GetName().c_str(), node->GetType().c_str(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] In anchor index:%zu in op:%s(%s) peer anchor is nullptr, model_id:%u.",
           get_dynamic_dims_index, node->GetName().c_str(), node->GetType().c_str(), model_id_);
    return PARAM_INVALID;
  }
  auto peer_node = peer_out_anchor->GetOwnerNode();
  auto op_desc = peer_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  if (op_desc->GetName() == kGetDynamicDimsName && op_desc->GetType() == GETDYNAMICDIMS) {
    GELOGD("Start get info of %s.", op_desc->GetName().c_str());
    auto input_addr = ModelUtils::GetInputDataAddrs(runtime_param_, node->GetOpDesc());
    auto input_size = ModelUtils::GetInputSize(node->GetOpDesc());
    if (input_addr.empty() || input_size.empty()) {
      REPORT_INNER_ERROR("E19999", "input_addr size:%zu or input_length size:%zu in op:%s(%s) has empty, model_id:%u "
                         "check invalid", input_addr.size(), input_size.size(),
                         node->GetName().c_str(), node->GetType().c_str(), model_id_);
      GELOGE(PARAM_INVALID, "[Check][Param] input_addr size:%zu or input_length size:%zu in op:%s(%s) is empty, "
             "model_id:%u", input_addr.size(), input_size.size(),
             node->GetName().c_str(), node->GetType().c_str(), model_id_);
      return PARAM_INVALID;
    }
    auto input_desc = node->GetOpDesc()->GetInputDescPtr(get_dynamic_dims_index);
    GE_CHECK_NOTNULL(input_desc);
    if (input_desc->GetShape().GetDims().empty()) {
      REPORT_INNER_ERROR("E19999", "input_desc_index:%zu in op:%s(%s) shape dim is empty, model_id:%u, check invalid",
                         get_dynamic_dims_index, node->GetName().c_str(), node->GetType().c_str(), model_id_);
      GELOGE(PARAM_INVALID, "[Check][Param] input_desc_index:%zu in op:%s(%s) shape dim is empty, model_id:%u",
             get_dynamic_dims_index, node->GetName().c_str(), node->GetType().c_str(), model_id_);
      return PARAM_INVALID;
    }
    netoutput_last_input_addr_ = input_addr[get_dynamic_dims_index];
    netoutput_last_input_size_ = input_size[get_dynamic_dims_index];
    shape_of_cur_dynamic_dims_ = input_desc->GetShape().GetDims().at(0);
    GELOGD("Shape of cur dynamic dims is %zu, size is %ld, addr is %p.", shape_of_cur_dynamic_dims_,
           netoutput_last_input_size_, netoutput_last_input_addr_);
  }
  return SUCCESS;
}

Status DavinciModel::GetGearAndRealOutSizeInfo(const ComputeGraphPtr &graph, const NodePtr &node) {
  GELOGD("Start get gear and real output size info of %s.", node->GetName().c_str());
  merge_nodes_gear_and_real_out_size_info_.clear();
  size_t idx = 0;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto peer_node = peer_out_anchor->GetOwnerNode();
    auto op_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if ((peer_node->GetType() == CASE) && (op_desc->HasAttr(ATTR_INSERT_BY_MBATCH))) {
      if (GetRealOutputSizeOfCase(graph, idx, peer_node) != SUCCESS) {
        GELOGE(PARAM_INVALID, "[Get][RealOutputSizeOfCase] %s failed.", peer_node->GetName().c_str());
        return PARAM_INVALID;
      }
    }
    idx++;
  }
  return SUCCESS;
}

Status DavinciModel::GetRealOutputSizeOfCase(const ComputeGraphPtr &graph, size_t input_index,
                                             const NodePtr &case_node) {
  GELOGD("Start to get output size of %s, which is %zu input to netoutput", case_node->GetName().c_str(), input_index);
  const auto &func_desc = case_node->GetOpDesc();
  GE_CHECK_NOTNULL(func_desc);
  std::map<vector<int32_t>, int64_t> gear_and_real_out_size_info;
  for (const auto &name : func_desc->GetSubgraphInstanceNames()) {
    const auto &subgraph = graph->GetSubgraph(name);
    if (subgraph == nullptr) {
      REPORT_INNER_ERROR("E19999", "Get name:%s subgraph in graph:%s fail, model_id:%u, check invalid",
                         name.c_str(), graph->GetName().c_str(), model_id_);
      GELOGE(GE_GRAPH_EMPTY_SUBGRAPH, "[Get][Subgraph] %s in graph:%s failed, model_id:%u.",
             name.c_str(), graph->GetName().c_str(), model_id_);
      return GE_GRAPH_EMPTY_SUBGRAPH;
    }
    for (auto &node : subgraph->GetDirectNode()) {
      if (node->GetType() == NETOUTPUT) {
        auto op_desc = node->GetOpDesc();
        GE_CHECK_NOTNULL(op_desc);
        string batch_label;
        if (AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label)) {
          size_t batch_index = static_cast<size_t>(stoi(batch_label.substr(batch_label.rfind('_') + 1)));
          GELOGD("Batch index of %s is %zu.", op_desc->GetName().c_str(), batch_index);
          if (batch_index > all_gears_info_.size()) {
            REPORT_INNER_ERROR("E19999", "Batch_index:%zu in op:%s(%s) > all_gears_info.size:%zu, model_id:%u, "
                               "check invalid", batch_index,
                               op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                               all_gears_info_.size(), model_id_);
            GELOGE(PARAM_INVALID, "[Check][Param] Batch_index:%zu in op:%s(%s) > all_gears_info.size:%zu, "
                   "model_id:%u.", batch_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                   all_gears_info_.size(), model_id_);
            return PARAM_INVALID;
          }

          const vector<int64_t> input_size_list = ModelUtils::GetInputSize(op_desc);
          auto tensor_desc = op_desc->GetInputDescPtr(input_index);
          GE_CHECK_NOTNULL(tensor_desc);
          int64_t data_size = 0;
          if (TensorUtils::GetTensorSizeInBytes(*tensor_desc, data_size) != GRAPH_SUCCESS) {
            REPORT_INNER_ERROR("E19999", "Get input TensorSize in op:%s(%s) failed, input_index:%zu, model_id:%u",
                               op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                               input_index, model_id_);
            GELOGE(FAILED, "[Get][TensorSize] in op:%s(%s) failed, input_index:%zu, model_id:%u",
                   op_desc->GetName().c_str(), op_desc->GetType().c_str(), input_index, model_id_);
            return FAILED;
          }
          gear_and_real_out_size_info[all_gears_info_[batch_index]] = data_size;
          GELOGD("Get real gear index is: %zu, gear info is %s, size is %ld, tensor size is %ld",
                 batch_index, formats::JoinToString(all_gears_info_[batch_index]).c_str(),
                 input_size_list[input_index], data_size);
        }
        break;
      }
    }
  }
  merge_nodes_gear_and_real_out_size_info_[input_index] = gear_and_real_out_size_info;
  return SUCCESS;
}

Status DavinciModel::GetGearAndRealOutShapeInfo(const ComputeGraphPtr &graph, const NodePtr &node) {
  GELOGD("Start to get dynamic output dims of %s", node->GetName().c_str());
  merge_nodes_gear_and_real_out_shape_info_.clear();
  size_t idx = 0;
  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto peer_node = peer_out_anchor->GetOwnerNode();
    auto op_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    if ((peer_node->GetType() == CASE) && (op_desc->HasAttr(ATTR_INSERT_BY_MBATCH))) {
      std::vector<std::string> dynamic_output_shape_info;
      if (!AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_DYNAMIC_OUTPUT_DIMS, dynamic_output_shape_info)) {
        GELOGD("Can not get dynamic output dims attr from %s", node->GetName().c_str());
        return SUCCESS;
      }
      GELOGI("Dynamic output shape info is %s", formats::JoinToString(dynamic_output_shape_info).c_str());
      std::vector<vector<int64_t>> dynamic_output_shape;
      ParseDynamicOutShape(dynamic_output_shape_info, dynamic_output_shape);
      std::map<vector<int32_t>, vector<int64_t>> gear_and_real_out_shape_info;
      for (auto &it : dynamic_output_shape) {
        auto gear_index = static_cast<size_t>(it[0]);
        if (gear_index > all_gears_info_.size()) {
          REPORT_INNER_ERROR("E19999", "gear index:%zu in op:%s(%s) > all_gears_info.size:%zu in model:%u "
                             "check invalid", gear_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                             all_gears_info_.size(), model_id_);
          GELOGE(PARAM_INVALID, "[Check][Param] gear index:%zu in op:%s(%s) > all_gears_info.size:%zu in model:%u.",
                 gear_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), all_gears_info_.size(), model_id_);
          return PARAM_INVALID;
        }

        if (static_cast<size_t>(it[1]) == idx) {
          vector<int64_t> output_shape;
          for (size_t i = 2; i < it.size(); ++i) {
            output_shape.emplace_back(it[i]);
          }
          gear_and_real_out_shape_info[all_gears_info_[gear_index]] = output_shape;
          GELOGD("Get real gear index is: %zu, gear info is %s, output shape is %s",
                 gear_index, formats::JoinToString(all_gears_info_[gear_index]).c_str(),
                 formats::JoinToString(output_shape).c_str());
        }
      }
      merge_nodes_gear_and_real_out_shape_info_[idx] = gear_and_real_out_shape_info;
    }
    idx++;
  }
  return SUCCESS;
}

void DavinciModel::ParseDynamicOutShape(const std::vector<std::string> &str_info,
                                        std::vector<vector<int64_t>> &vec_info) {
  for (size_t i = 0; i < str_info.size(); ++i) {
    std::vector<int64_t> shape;
    std::vector<std::string> dims = ge::StringUtils::Split(str_info[i], ',');
    for (const auto &dim : dims) {
      if (dim.empty()) {
        continue;
      }
      shape.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimal));
    }
    GELOGI("Shape from attr is %s", formats::JoinToString(shape).c_str());
    vec_info.emplace_back(shape);
  }
}

Status DavinciModel::GetLabelGotoAddr(uint32_t label_index, rtMemType_t mem_type, void *&arg_addr, uint32_t &arg_size) {
  std::lock_guard<std::mutex> lock(label_args_mutex_);
  auto it = label_goto_args_.find(label_index);
  if (it != label_goto_args_.end()) {
    arg_addr = it->second.first;
    arg_size = it->second.second;
    return SUCCESS;
  }

  if (label_index >= label_list_.size()) {
    REPORT_INNER_ERROR("E19999", "Param label index:%u >= label_list_.size:%zu in model:%u, check invalid",
                       label_index, label_list_.size(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Param label index:%u >= label_list_.size:%zu in model:%u",
           label_index, label_list_.size(), model_id_);
    return INTERNAL_ERROR;
  }
  GE_CHECK_NOTNULL(label_list_[label_index]);
  vector<rtLabel_t> label_used = { label_list_[label_index] };

  arg_size = label_used.size() * sizeof(rtLabelDevInfo);
  rtError_t rt_ret = rtMalloc(&arg_addr, arg_size, mem_type);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret:0x%X", arg_size, rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret:0x%X", arg_size, rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  label_goto_args_[label_index] = { arg_addr, arg_size };
  rt_ret = rtLabelListCpy(label_used.data(), label_used.size(), arg_addr, arg_size);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtLabelListCpy failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtLabelListCpy] failed, ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  return SUCCESS;
}

void DavinciModel::SetGlobalStep(void *global_step, uint64_t global_step_size) {
  global_step_addr_ = global_step;
  global_step_size_ = global_step_size;
}

/// @ingroup ge
/// @brief LabelSet Op Initialize.
/// @param [in] op_desc: LabelSet Op descriptor.
/// @return Status
Status DavinciModel::InitLabelSet(const OpDescPtr &op_desc) {
  uint32_t label_index = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, label_index)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s in op:%s(%s) fail, model_id:%u, check invalid",
                       ATTR_NAME_LABEL_SWITCH_INDEX.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s in op:%s(%s) fail, model_id:%u",
           ATTR_NAME_LABEL_SWITCH_INDEX.c_str(), op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return INTERNAL_ERROR;
  }
  if (label_index >= LabelNum()) {
    REPORT_INNER_ERROR("E19999", "label_switch_index:%u in op:%s(%s) >= label_num:%u in model:%u, check invalid",
                       label_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       LabelNum(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] label_switch_index:%u in op:%s(%s) >= label_num:%u in model:%u",
           label_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), LabelNum(), model_id_);
    return INTERNAL_ERROR;
  }
  if (label_id_indication_.count(label_index) > 0) {
    REPORT_INNER_ERROR("E19999", "label_switch_index:%u in op:%s(%s) is already used  in model:%u, check invalid",
                       label_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] label_switch_index:%u in op:%s(%s) is already used  in model:%u",
           label_index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return INTERNAL_ERROR;
  }

  rtStream_t stream = nullptr;
  uint32_t stream_id = static_cast<uint32_t>(op_desc->GetStreamId());
  if (stream_list_.size() == 1) {
    stream = stream_list_[0];
  } else if (stream_list_.size() > stream_id) {
    stream = stream_list_[stream_id];
  } else {
    REPORT_INNER_ERROR("E19999", "stream_id:%u in op:%s(%s) >= stream size:%zu in model:%u, check invalid",
                       stream_id, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       stream_list_.size(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] stream_id:%u in op:%s(%s) >= stream size:%zu in model:%u",
           stream_id, op_desc->GetName().c_str(), op_desc->GetType().c_str(), stream_list_.size(), model_id_);
    return INTERNAL_ERROR;
  }

  rtLabel_t rt_label = nullptr;
  rtError_t rt_error = rtLabelCreateExV2(&rt_label, rt_model_handle_, stream);
  if (rt_error != RT_ERROR_NONE || rt_label == nullptr) {
    REPORT_CALL_ERROR("E19999", "Call rtLabelCreateExV2 failed, ret:0x%X", rt_error);
    GELOGE(INTERNAL_ERROR, "[Call][RtLabelCreateExV2] InitLabelSet: %s create label failed, ret:0x%x.",
           op_desc->GetName().c_str(), rt_error);
    return INTERNAL_ERROR;
  }

  GELOGI("InitLabelSet: label[%u]=%p stream[%u]=%p", label_index, rt_label, stream_id, stream);
  label_id_indication_.insert(label_index);
  label_list_[label_index] = rt_label;
  return SUCCESS;
}

Status DavinciModel::InitVariable(const OpDescPtr &op_desc, map<string, OpDescPtr> &variable_by_name) {
  if (!known_node_) {
    if (op_desc->GetName() == NODE_NAME_GLOBAL_STEP) {
      const auto output_sizes = ModelUtils::GetOutputSize(op_desc);
      if (!output_sizes.empty()) {
        global_step_size_ = output_sizes[0];
      }
      const auto output_addrs = ModelUtils::GetOutputDataAddrs(runtime_param_, op_desc);
      if (!output_addrs.empty()) {
        global_step_addr_ = output_addrs[0];
      }
    }
  }

  if (op_desc->HasAttr(VAR_ATTR_VAR_IS_BROADCAST)) {
    broadcast_variable_[op_desc->GetName()] = op_desc->GetOutputDesc(0);
  }

  variable_by_name[op_desc->GetName()] = op_desc;
  return SUCCESS;
}

/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [in] input_queue_ids: input queue ids from user, nums equal Data Op.
/// @param [in] output_queue_ids: input queue ids from user, nums equal NetOutput Op.
/// @return: 0 for success / others for failed
Status DavinciModel::SetQueIds(const std::vector<uint32_t> &input_queue_ids,
                               const std::vector<uint32_t> &output_queue_ids) {
  if (input_queue_ids.empty() && output_queue_ids.empty()) {
    REPORT_INNER_ERROR("E19999", "Param input_queue_ids.size:%zu and output_queue_ids.size:%zu is empty, model_id:%u,"
                       "check invalid", input_queue_ids.size(), output_queue_ids.size(),
                       model_id_);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID, "[Check][Param] Param is empty, model_id:%u", model_id_);
    return ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  input_queue_ids_ = input_queue_ids;
  output_queue_ids_ = output_queue_ids;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [in] input_que_ids: input queue ids from user, nums equal Data Op.
/// @param [in] output_que_ids: input queue ids from user, nums equal NetOutput Op.
/// @return: 0 for success / others for failed
///
Status DavinciModel::LoadWithQueue() {
  if (input_queue_ids_.empty() && output_queue_ids_.empty()) {
    return SUCCESS;
  }

  if (input_queue_ids_.size() != input_data_info_.size()) {
    REPORT_INNER_ERROR("E19999", "Param input_queue_ids_.size:%zu != input_data_info_.size:%zu, model_id:%u,"
                       "check invalid", input_queue_ids_.size(), input_data_info_.size(),
                       model_id_);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID, "[Check][Param] Input queue ids not match model: "
           "input_queue=%zu input_data=%zu, model_id:%u", input_queue_ids_.size(), input_data_info_.size(), model_id_);
    return ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  if (output_queue_ids_.size() != output_data_info_.size()) {
    REPORT_INNER_ERROR("E19999", "Param output_queue_ids_.size:%zu != output_data_info_.size:%zu, model_id:%u,"
                       "check invalid", output_queue_ids_.size(), output_data_info_.size(), model_id_);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID,
           "[Check][Param] Output queue ids not match model: output_queue=%zu output_data=%zu, model_id:%u",
           output_queue_ids_.size(), output_data_info_.size(), model_id_);
    return ACL_ERROR_GE_EXEC_MODEL_QUEUE_ID_INVALID;
  }

  GE_CHK_STATUS_RET(AddHeadStream(), "[Add][HeadStream] failed, model_id:%u", model_id_);
  // Binding input_queue and Data Op.
  GE_CHK_STATUS_RET(BindInputQueue(), "[Bind][InputQueue] failed, model_id:%u", model_id_);
  GE_CHK_STATUS_RET(CpuTaskModelZeroCopy(input_mbuf_list_, input_data_info_),
                    "[Call][CpuTaskModelZeroCopy] failed, model_id:%u", model_id_);

  // Binding output_queue and NetOutput Op.
  GE_CHK_STATUS_RET(BindOutputQueue(), "[Bind][OutputQueue] failed, model_id:%u", model_id_);
  GE_CHK_STATUS_RET(CpuTaskModelZeroCopy(output_mbuf_list_, output_data_info_),
                    "[Call][CpuTaskModelZeroCopy] failed, model_id:%u", model_id_);

  GE_CHK_STATUS_RET(CpuActiveStream(), "[Call][CpuActiveStream] failed, model_id:%u", model_id_);
  GE_CHK_STATUS_RET(CpuWaitEndGraph(), "[Call][CpuWaitEndGraph] failed, model_id:%u", model_id_);
  GE_CHK_STATUS_RET(BindEnqueue(), "[Call][BindEnqueue] failed, model_id:%u", model_id_);
  GE_CHK_STATUS_RET(CpuModelRepeat(), "[Call][CpuModelRepeat] failed, model_id:%u", model_id_);

  return SUCCESS;
}

/// @ingroup ge
/// @brief queue schedule, Bind  input queue to Data output address.
/// @return: 0 for success / others for failed
Status DavinciModel::BindInputQueue() {
  // Caller checked: input_queue_ids_.size() == input_size_list_.size() != input_addr_list_.size()
  for (size_t i = 0; i < input_queue_ids_.size(); ++i) {
    auto it = input_data_info_.find(i);
    if (it == input_data_info_.end()) {
      GELOGE(FAILED, "[Check][Param] Input not match: tensor num=%zu, Queue id index=%zu", input_data_info_.size(), i);
      return FAILED;
    }

    uint32_t queue_id = input_queue_ids_[i];
    if (it->second.GetDataInfo().empty()) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] the %zu input_queue not set data_info.", i);
      return INTERNAL_ERROR;
    }
    uint32_t data_size = static_cast<uint32_t>(it->second.GetDataInfo().at(0).first);
    uintptr_t data_addr = reinterpret_cast<uintptr_t>(it->second.GetDataInfo().at(0).second);
    GELOGI("BindInputToQueue: graph_%u index[%zu] queue id[%u] output addr[0x%lx] output size[%u]",
           runtime_param_.graph_id, i, queue_id, data_addr, data_size);

    rtError_t rt_ret = rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_INPUT_QUEUE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtModelBindQueue failed, ret: 0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Call][RtModelBindQueue] failed, ret: 0x%X", rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    if (CpuModelDequeue(queue_id) != SUCCESS) {
      return INTERNAL_ERROR;
    }
  }

  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, bind input queue to task.
/// @param [in] queue_id: input queue id from user.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelDequeue(uint32_t queue_id) {
  GELOGI("Set CpuKernel model dequeue task enter.");
  std::shared_ptr<CpuTaskModelDequeue> dequeue_task = MakeShared<CpuTaskModelDequeue>(rt_entry_stream_);
  if (dequeue_task == nullptr) {
    REPORT_CALL_ERROR("E19999", "New CpuTaskModelDequeue failed, model_id:%u", model_id_);
    GELOGE(MEMALLOC_FAILED, "[New][CpuTaskModelDequeue] task failed, model_id:%u", model_id_);
    return MEMALLOC_FAILED;
  }

  // Get DataOp Output address and bind to queue.
  uintptr_t in_mbuf = 0;
  Status status = dequeue_task->Init(queue_id, in_mbuf);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(dequeue_task);
  input_mbuf_list_.push_back(in_mbuf);
  GELOGI("Set CpuKernel model dequeue task success.");
  return SUCCESS;
}

Status DavinciModel::CpuTaskModelZeroCopy(std::vector<uintptr_t> &mbuf_list,
                                          const map<uint32_t, ZeroCopyOffset> &outside_addrs) {
  GELOGI("Set CpuKernel model zero_copy task enter.");
  std::shared_ptr<CpuTaskZeroCopy> zero_copy = MakeShared<CpuTaskZeroCopy>(rt_entry_stream_);
  if (zero_copy == nullptr) {
    REPORT_CALL_ERROR("E19999", "New CpuTaskZeroCopy failed, model_id:%u", model_id_);
    GELOGE(MEMALLOC_FAILED, "[New][CpuTaskZeroCopy] failed, model_id:%u", model_id_);
    return MEMALLOC_FAILED;
  }

  // mdc zero_copy not support l2 fusion
  Status status = zero_copy->Init(mbuf_list, outside_addrs);
  if (status != SUCCESS) {
    return status;
  }
  cpu_task_list_.push_back(zero_copy);
  GELOGI("Set CpuKernel model zero_copy task success.");
  return SUCCESS;
}

/// @ingroup ge
/// @brief queue schedule, bind output queue to NetOutput input address.
/// @return: 0 for success / others for failed
Status DavinciModel::BindOutputQueue() {
  // Caller checked: input_queue_ids_.size() == input_size_list_.size() != input_addr_list_.size()
  for (size_t i = 0; i < output_queue_ids_.size(); ++i) {
    auto it = output_data_info_.find(i);
    if (it == output_data_info_.end()) {
      REPORT_INNER_ERROR("E19999", "Index:%zu can't find in output_data_info_ size:%zu in model_id:%u, check invalid",
                         i, output_data_info_.size(), model_id_);
      GELOGE(FAILED, "[Check][Param] Index:%zu can't find in output_data_info_ size:%zu in model_id:%u",
             i, output_data_info_.size(), model_id_);
      return FAILED;
    }

    uint32_t queue_id = output_queue_ids_[i];
    if (it->second.GetDataInfo().empty()) {
      REPORT_INNER_ERROR("E19999", "Index:%zu out_data_info in model:%u is empty, check invalid", i, model_id_);
      GELOGE(INTERNAL_ERROR, "[Check][Param] Index:%zu out_data_info in model:%u is empty, check invalid",
             i, model_id_);
      return INTERNAL_ERROR;
    }
    uint32_t data_size = static_cast<uint32_t>(it->second.GetDataInfo().at(0).first);
    uintptr_t data_addr = reinterpret_cast<uintptr_t>(it->second.GetDataInfo().at(0).second);
    GELOGI("BindOutputToQueue: graph_%u index[%zu] queue id[%u] input addr[0x%lx] input size[%u]",
           runtime_param_.graph_id, i, queue_id, data_addr, data_size);

    rtError_t rt_ret = rtModelBindQueue(rt_model_handle_, queue_id, RT_MODEL_OUTPUT_QUEUE);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtModelBindQueue failed, queue_id:%u, ret:0x%X", queue_id, rt_ret);
      GELOGE(RT_FAILED, "[Call][RtModelBindQueue] failed, queue_id:%u, ret:0x%X", queue_id, rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }

    Status status = CpuModelPrepareOutput(data_addr, data_size);
    if (status != SUCCESS) {
      return status;
    }
  }

  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, bind output queue to task.
/// @param [in] addr: NetOutput Op input tensor address.
/// @param [in] size: NetOutput Op input tensor size.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelPrepareOutput(uintptr_t addr, uint32_t size) {
  GELOGI("Set CpuKernel model enqueue task enter.");
  if (input_mbuf_list_.empty()) {
    REPORT_INNER_ERROR("E19999", "input_mbuf_list_ is empty, model_id:%u, check invalid", model_id_);
    GELOGE(FAILED, "[Check][Param] input_mbuf_list_ is empty, model_id:%u", model_id_);
    return FAILED;
  }

  std::shared_ptr<CpuTaskPrepareOutput> prepare_output = MakeShared<CpuTaskPrepareOutput>(rt_entry_stream_);
  if (prepare_output == nullptr) {
    REPORT_CALL_ERROR("E19999", "New CpuTaskPrepareOutput failed, model_id:%u", model_id_);
    GELOGE(MEMALLOC_FAILED, "[New][CpuTaskPrepareOutput] failed, model_id:%u", model_id_);
    return MEMALLOC_FAILED;
  }

  uintptr_t out_mbuf = 0;
  if (prepare_output->Init(addr, size, input_mbuf_list_.back(), out_mbuf) != SUCCESS) {
    return FAILED;
  }

  cpu_task_list_.push_back(prepare_output);
  output_mbuf_list_.push_back(out_mbuf);
  GELOGI("Set CpuKernel model enqueue task success.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief definiteness queue schedule, active original model stream.
/// @return: 0 for success / others for failed
///
Status DavinciModel::CpuActiveStream() {
  GELOGI("Set CpuKernel active stream task enter.");
  std::shared_ptr<CpuTaskActiveEntry> active_entry = MakeShared<CpuTaskActiveEntry>(rt_entry_stream_);
  if (active_entry == nullptr) {
    REPORT_CALL_ERROR("E19999", "New CpuTaskActiveEntry failed, model_id:%u", model_id_);
    GELOGE(MEMALLOC_FAILED, "[New][CpuTaskActiveEntry] failed, model_id:%u", model_id_);
    return MEMALLOC_FAILED;
  }

  Status status = active_entry->Init(rt_head_stream_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(active_entry);
  GELOGI("Set CpuKernel active stream task success.");
  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, wait for end graph.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuWaitEndGraph() {
  GELOGI("Set CpuKernel wait end graph task enter.");
  std::shared_ptr<CpuTaskWaitEndGraph> wait_endgraph = MakeShared<CpuTaskWaitEndGraph>(rt_entry_stream_);
  if (wait_endgraph == nullptr) {
    REPORT_CALL_ERROR("E19999", "New CpuTaskWaitEndGraph failed, model_id:%u", model_id_);
    GELOGE(MEMALLOC_FAILED, "[New][CpuTaskWaitEndGraph] failed, model_id:%u", model_id_);
    return MEMALLOC_FAILED;
  }

  Status status = wait_endgraph->Init(runtime_model_id_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(wait_endgraph);
  GELOGI("Set CpuKernel wait end graph task success.");
  return SUCCESS;
}

Status DavinciModel::BindEnqueue() {
  for (size_t i = 0; i < output_queue_ids_.size(); ++i) {
    auto it = output_data_info_.find(i);
    if (it == output_data_info_.end()) {
      REPORT_INNER_ERROR("E19999", "Index:%zu can't find in output_data_info_ size:%zu in model_id:%u, check invalid",
                         i, output_data_info_.size(), model_id_);
      GELOGE(FAILED, "Index:%zu can't find in output_data_info_ size:%zu in model_id:%u",
             i, output_data_info_.size(), model_id_);
      return FAILED;
    }

    uint32_t queue_id = output_queue_ids_[i];
    if (CpuModelEnqueue(queue_id, output_mbuf_list_[i]) != SUCCESS) {
      return INTERNAL_ERROR;
    }
  }
  return SUCCESS;
}

Status DavinciModel::CpuModelEnqueue(uint32_t queue_id, uintptr_t out_mbuf) {
  GELOGI("Set CpuKernel model enqueue task enter.");
  std::shared_ptr<CpuTaskModelEnqueue> model_enqueue = MakeShared<CpuTaskModelEnqueue>(rt_entry_stream_);
  if (model_enqueue == nullptr) {
    REPORT_CALL_ERROR("E19999", "New CpuTaskModelEnqueue failed, model_id:%u", model_id_);
    GELOGE(MEMALLOC_FAILED, "[New][CpuTaskModelEnqueue] failed, model_id:%u", model_id_);
    return MEMALLOC_FAILED;
  }

  Status status = model_enqueue->Init(queue_id, out_mbuf);
  if (status != SUCCESS) {
    return status;
  }
  cpu_task_list_.push_back(model_enqueue);
  GELOGI("Set CpuKernel model enqueue task enter.");
  return SUCCESS;
}

/// @ingroup ge
/// @brief definiteness queue schedule, repeat run model.
/// @return: 0 for success / others for failed
Status DavinciModel::CpuModelRepeat() {
  GELOGI("Set CpuKernel repeat task enter.");
  std::shared_ptr<CpuTaskModelRepeat> model_repeat = MakeShared<CpuTaskModelRepeat>(rt_entry_stream_);
  if (model_repeat == nullptr) {
    REPORT_CALL_ERROR("E19999", "New CpuTaskModelRepeat failed, model_id:%u", model_id_);
    GELOGE(MEMALLOC_FAILED, "[New][CpuTaskModelRepeat] failed, model_id:%u", model_id_);
    return MEMALLOC_FAILED;
  }

  Status status = model_repeat->Init(runtime_model_id_);
  if (status != SUCCESS) {
    return status;
  }

  cpu_task_list_.push_back(model_repeat);
  GELOGI("Set CpuKernel repeat task success.");
  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc) {
  if (input_addrs_list_.empty() || input_addrs_list_[0].size() != 1) {
    GELOGI("data_op_list_ is empty or input_desc size is not 1.");
  } else {
    vector<uint32_t> input_formats;
    GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats, false),
                      "[Get][InputDescInfo] failed, model_id:%u", model_id_);
  }

  vector<uint32_t> output_formats;
  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats),
                    "[Get][OutputDescInfo] failed, model_id:%u", model_id_);
  return SUCCESS;
}

Status DavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc,
                                            vector<uint32_t> &input_formats,
                                            vector<uint32_t> &output_formats, bool by_dims) {
  if (input_addrs_list_.empty() || input_addrs_list_[0].size() != 1) {
    REPORT_INNER_ERROR("E19999", "input_addrs_list_ is empty or first member size != 1, model_id:%u, "
                       "check invalid", model_id_);
    GELOGE(FAILED, "[Check][Param] input_addrs_list_ is empty or first member size != 1, model_id:%u", model_id_);
    return FAILED;
  }

  GE_CHK_STATUS_RET(GetInputDescInfo(input_desc, input_formats, by_dims),
                    "[Get][InputDescInfo] failed, model_id:%u", model_id_);

  GE_CHK_STATUS_RET(GetOutputDescInfo(output_desc, output_formats),
                    "[Get][OutputDescInfo] failed, model_id:%u", model_id_);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [out] batch_info
/// @param [out] dynamic_type
/// @return execute result
///
Status DavinciModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) const {
  dynamic_type = dynamic_type_;
  batch_info = batch_info_;

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get combined dynamic dims info
/// @param [out] batch_info
/// @return None
///
void DavinciModel::GetCombinedDynamicDims(std::vector<std::vector<int64_t>> &batch_info) const {
  batch_info.clear();
  batch_info = combined_batch_info_;
}

///
/// @ingroup ge
/// @brief Get user designate shape order
/// @param [out] user_input_shape_order
/// @return None
///
void DavinciModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) const {
  user_input_shape_order.clear();
  user_input_shape_order = user_designate_shape_order_;
}

///
/// @ingroup ge
/// @brief Get AIPP input info
/// @param [in] index
/// @param [int] OpDescPtr
/// @return execute result
///
Status DavinciModel::InitAippInfo(uint32_t index, const OpDescPtr &op_desc) {
  if (!op_desc->HasAttr(ATTR_NAME_AIPP)) {
    GELOGW("There is not AIPP related with index %u", index);
    return SUCCESS;
  }

  domi::AippOpParams aipp_params;
  GeAttrValue::NAMED_ATTRS aipp_attr;
  GE_CHK_BOOL_RET_STATUS(AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_AIPP, aipp_attr), ACL_ERROR_GE_AIPP_NOT_EXIST,
                         "[Get][NamedAttrs] Data node:%s do not contain param aipp!", op_desc->GetName().c_str());
  GE_CHK_STATUS_RET(OpUtils::ConvertAippParams(aipp_attr, &aipp_params),
                    "[Convert][AippParams] get aipp params failed, op:%s", op_desc->GetName().c_str());
  GELOGI("Node data: %s, type: %s, current index: %u, current node related input rank: %u",
         op_desc->GetName().c_str(), op_desc->GetType().c_str(), index, aipp_params.related_input_rank());

  AippConfigInfo aipp_info;
  GE_CHK_STATUS_RET(AippUtils::ConvertAippParams2AippInfo(&aipp_params, aipp_info),
                    "[Call][ConvertAippParams2AippInfo] failed, op:%s", op_desc->GetName().c_str());

  aipp_info_list_[index] = aipp_info;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get AIPP input info
/// @param [in] index
/// @param [out] aipp_info
/// @return execute result
///
Status DavinciModel::GetAippInfo(uint32_t index, AippConfigInfo &aipp_info) const {
  const auto it = aipp_info_list_.find(index);
  if (it == aipp_info_list_.end()) {
    GELOGW("there is not AIPP related with index %u", index);
    return ACL_ERROR_GE_AIPP_NOT_EXIST;
  }

  aipp_info = it->second;
  return SUCCESS;
}

Status DavinciModel::InitAippType(uint32_t index, const OpDescPtr &op_desc, const map<uint32_t, OpDescPtr> &data_list) {
  if (!op_desc->HasAttr(ATTR_DATA_RELATED_AIPP_MODE)) {
    GELOGW("There is no aipp releated info with index %u", index);
    return SUCCESS;
  }

  // Set default value
  InputAippType aipp_type = DATA_WITHOUT_AIPP;
  string data_mode;
  (void)AttrUtils::GetStr(op_desc, ATTR_DATA_RELATED_AIPP_MODE, data_mode);
  if (data_mode == "static_aipp") {
    aipp_type = DATA_WITH_STATIC_AIPP;
  } else if (data_mode == "dynamic_aipp") {
    aipp_type = DATA_WITH_DYNAMIC_AIPP;
  } else if (data_mode == "dynamic_aipp_conf") {
    aipp_type = DYNAMIC_AIPP_NODE;
  } else {
    REPORT_INNER_ERROR("E19999", "Attr:%s data_mode:%s in op:%s(%s), model_id:%u, check invalid",
                       ATTR_DATA_RELATED_AIPP_MODE.c_str(), data_mode.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(ACL_ERROR_GE_AIPP_MODE_INVALID, "[Get][Attr] %s data_mode:%s in op:%s(%s), model_id:%u, check invalid",
           ATTR_DATA_RELATED_AIPP_MODE.c_str(), data_mode.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return ACL_ERROR_GE_AIPP_MODE_INVALID;
  }

  size_t aipp_index = 0xFFFFFFFF;  // default invalid value
  if (aipp_type == DATA_WITH_DYNAMIC_AIPP) {
    string releated_name;
    (void)AttrUtils::GetStr(op_desc, ATTR_DATA_AIPP_DATA_NAME_MAP, releated_name);
    for (const auto item : data_list) {
      if (item.second->GetName() == releated_name) {
        GELOGI("Find aipp_data [%s] index %u from index %u", releated_name.c_str(), item.first, index);
        aipp_index = item.first;
      }
    }

    if (aipp_index == 0xFFFFFFFF) {
      GELOGW("Can not find aipp data node from index %u", index);
      return SUCCESS;
    }
  }

  aipp_type_list_[index] = { aipp_type, aipp_index };
  return SUCCESS;
}

Status DavinciModel::GetAippType(uint32_t index, InputAippType &aipp_type, size_t &aipp_index) const {
  GE_CHK_BOOL_RET_STATUS(index < input_addrs_list_.size(), PARAM_INVALID,
                         "[Check][Param] Index %u is invalid", index);
  const auto it = aipp_type_list_.find(index);
  if (it == aipp_type_list_.end()) {
    GELOGW("There is no aipp releated info with index %u", index);
    aipp_type = DATA_WITHOUT_AIPP;
    aipp_index = 0xFFFFFFFF;
    return SUCCESS;
  }

  aipp_type = it->second.first;
  aipp_index = it->second.second;
  return SUCCESS;
}

void DavinciModel::SetDynamicSize(const std::vector<uint64_t> &batch_num, int32_t dynamic_type) {
  batch_size_.clear();
  if (batch_num.empty()) {
    GELOGD("User has not set dynammic data");
  }
  for (size_t i = 0; i < batch_num.size(); i++) {
    batch_size_.emplace_back(batch_num[i]);
  }

  dynamic_type_ = dynamic_type;
}

void DavinciModel::GetCurShape(std::vector<int64_t> &batch_info, int32_t &dynamic_type) const {
  if (batch_size_.empty()) {
    GELOGD("User does not set dynamic size");
  }
  for (size_t i = 0; i < batch_size_.size(); i++) {
    GELOGI("Start to get current shape");
    batch_info.emplace_back(batch_size_[i]);
  }

  dynamic_type = dynamic_type_;
}

Status DavinciModel::GetOpAttr(const std::string &op_name, const std::string &attr_name,
                               std::string &attr_value) const {
  auto itr = op_name_to_attrs_.find(op_name);
  if (itr == op_name_to_attrs_.end()) {
    GELOGW("Did not save op:%s attr", op_name.c_str());
    return SUCCESS;
  }
  auto attr_itr = itr->second.find(attr_name);
  if (attr_itr == itr->second.end()) {
    GELOGW("Did not save attr:%s of op:%s", attr_name.c_str(), op_name.c_str());
    return SUCCESS;
  }
  for (const auto &name : attr_itr->second) {
    attr_value += "[" + std::to_string(name.size()) + "]" + name;
  }
  GELOGD("Get attr:%s of op:%s success, attr value:%s", attr_name.c_str(), op_name.c_str(), attr_value.c_str());
  return SUCCESS;
}

void DavinciModel::GetModelAttr(vector<string> &out_shape_info) const {
  out_shape_info.insert(out_shape_info.end(), dynamic_output_shape_info_.begin(), dynamic_output_shape_info_.end());
}

void DavinciModel::SetInputDimsInfo(const vector<int64_t> &input_dims, Format &format, ShapeDescription &shape_info) {
  uint32_t n, c, h, w;
  n = format == FORMAT_NHWC ? NHWC_DIM_N : NCHW_DIM_N;
  c = format == FORMAT_NHWC ? NHWC_DIM_C : NCHW_DIM_C;
  h = format == FORMAT_NHWC ? NHWC_DIM_H : NCHW_DIM_H;
  w = format == FORMAT_NHWC ? NHWC_DIM_W : NCHW_DIM_W;

  if (input_dims.size() == static_cast<size_t>(NORMAL_TENSOR_SIZE)) {
    shape_info.num = input_dims[n];
    shape_info.height = input_dims[h];
    shape_info.width = input_dims[w];
    shape_info.channel = input_dims[c];
  }
  for (size_t k = 0; k < input_dims.size(); ++k) {
    shape_info.dims.push_back(input_dims[k]);
  }
}

void DavinciModel::CreateInputDimsInfo(const OpDescPtr &op_desc, Format format,
                                       ShapeDescription &shape_info, ShapeDescription &dims_info) {
  // judge if this data is linked dynamic aipp first, multiply batch has been considered
  if (op_desc->HasAttr(ATTR_DYNAMIC_AIPP_INPUT_DIMS)) {
    vector<int64_t> dynamic_aipp_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_DYNAMIC_AIPP_INPUT_DIMS, dynamic_aipp_input_dims);
    SetInputDimsInfo(dynamic_aipp_input_dims, format, shape_info);
  } else {
    // judge if this data is multiply batch
    if (!op_desc->HasAttr(ATTR_MBATCH_ORIGIN_INPUT_DIMS)) {
      vector<int64_t> input_dims = op_desc->GetInputDescPtr(0)->GetShape().GetDims();
      SetInputDimsInfo(input_dims, format, shape_info);
    } else {
      vector<int64_t> origin_input_dims;
      (void)AttrUtils::GetListInt(op_desc, ATTR_MBATCH_ORIGIN_INPUT_DIMS, origin_input_dims);
      SetInputDimsInfo(origin_input_dims, format, shape_info);
    }
  }

  if (op_desc->HasAttr(ATTR_NAME_INPUT_DIMS)) {
    // When static aipp is set, need to get the model input dims which processed by aipp
    vector<int64_t> model_input_dims;
    (void)AttrUtils::GetListInt(op_desc, ATTR_NAME_INPUT_DIMS, model_input_dims);
    SetInputDimsInfo(model_input_dims, format, dims_info);
  } else {
    dims_info = shape_info;
  }
}

Status DavinciModel::InitInputDescInfo(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc->GetInputDescPtr(0));

  InputOutputDescInfo input;
  ShapeDescription dims_info;
  Format format = op_desc->GetInputDescPtr(0)->GetFormat();
  CreateInputDimsInfo(op_desc, format, input.shape_info, dims_info);

  input.data_type = op_desc->GetInputDescPtr(0)->GetDataType();
  input.name = op_desc->GetName();
  int64_t input_size = 0;
  GE_CHK_STATUS_RET(TensorUtils::GetSize(*op_desc->GetInputDescPtr(0), input_size),
                    "[Get][InputSize] failed in op:%s.", op_desc->GetName().c_str());
  input.size = input_size;
  input_formats_.push_back(format);
  input_descs_.push_back(input);

  input.shape_info = dims_info;
  input_descs_dims_.push_back(input);
  return SUCCESS;
}

Status DavinciModel::GetInputDescInfo(vector<InputOutputDescInfo> &input_descs,
                                      vector<uint32_t> &input_formats, bool by_dims) const {
  const vector<InputOutputDescInfo> &input_desc_info = by_dims ? input_descs_dims_ : input_descs_;
  input_descs.insert(input_descs.end(), input_desc_info.begin(), input_desc_info.end());
  input_formats.insert(input_formats.end(), input_formats_.begin(), input_formats_.end());

  return SUCCESS;
}

void DavinciModel::CreateOutput(uint32_t index, const OpDescPtr &op_desc, InputOutputDescInfo &output,
                                uint32_t &format_result) {
  /// netoutput input tensor desc
  GE_IF_BOOL_EXEC(op_desc->GetInputDescPtr(index) == nullptr,
                  REPORT_INNER_ERROR("E19999", "input_desc index:%u in op:%s(%s) not exist, model_id:%u, "
                                     "check invalid", index, op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                                     model_id_);
                  GELOGE(FAILED, "[Get][InputDescPtr] input_desc index:%u in op:%s(%s) not exist, model_id:%u",
                         index, op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
                  return);
  Format format = op_desc->GetInputDescPtr(index)->GetFormat();
  GeShape shape = op_desc->GetInputDescPtr(index)->GetShape();
  DataType data_type = op_desc->GetInputDescPtr(index)->GetDataType();

  int64_t dims[] = {1, 1, 1, 1};
  format_result = format;
  if (format == FORMAT_ND) {  // for ND tensor
    for (size_t i = 0; i < shape.GetDimNum() && i < (sizeof(dims) / sizeof(dims[0])); i++) {
      dims[i] = shape.GetDim(i);
    }
  } else {                                                                    // FOR FORMAT_NHWC or FORMAT_NCHW
    dims[0] = shape.GetDim((format == FORMAT_NHWC) ? NHWC_DIM_N : NCHW_DIM_N);  // 0: first dim
    dims[1] = shape.GetDim((format == FORMAT_NHWC) ? NHWC_DIM_C : NCHW_DIM_C);  // 1: second dim
    dims[2] = shape.GetDim((format == FORMAT_NHWC) ? NHWC_DIM_H : NCHW_DIM_H);  // 2: third dim
    dims[3] = shape.GetDim((format == FORMAT_NHWC) ? NHWC_DIM_W : NCHW_DIM_W);  // 3: forth dim
  }
  output.shape_info.num = dims[0];      // 0: first dim
  output.shape_info.channel = dims[1];  // 1: second dim
  output.shape_info.height = dims[2];   // 2: third dim
  output.shape_info.width = dims[3];    // 3: forth dim

  if (op_desc->GetInputDescPtr(index)->GetFormat() == FORMAT_FRACTAL_Z) {  // FraczToHWCK
    int64_t k = shape.GetDim(0);                                           // 0: first dim
    int64_t c = shape.GetDim(1);                                           // 1: second dim
    int64_t h = shape.GetDim(2);                                           // 2: third dim
    int64_t w = shape.GetDim(3);                                           // 3: forth dim
    output.shape_info.dims.push_back(h);
    output.shape_info.dims.push_back(w);
    output.shape_info.dims.push_back(c);
    output.shape_info.dims.push_back(k);
    format_result = FORMAT_HWCN;
  } else {
    for (size_t j = 0; j < shape.GetDimNum(); j++) {
      output.shape_info.dims.push_back(shape.GetDim(j));
    }
  }

  int64_t tensor_size = 0;
  if (AttrUtils::GetInt(op_desc->GetInputDescPtr(index), ATTR_NAME_SPECIAL_OUTPUT_SIZE, tensor_size)
     && (tensor_size > 0)) {
    GELOGI("netoutput[%s] [%d]th input has special size [%ld]", op_desc->GetName().c_str(), index, tensor_size);
  } else {
    (void)TensorUtils::CalcTensorMemSize(shape, format, data_type, tensor_size);  // no need to check value
  }
  output.size = static_cast<uint64_t>(tensor_size);
  output.data_type = op_desc->GetInputDescPtr(index)->GetDataType();
}

Status DavinciModel::InitOutputDescInfo(const OpDescPtr &op_desc, const vector<string> &out_node_name) {
  uint32_t out_size = static_cast<uint32_t>(op_desc->GetInputsSize());
  for (uint32_t i = 0; i < out_size; ++i) {
    string output_name;
    InputOutputDescInfo output;
    uint32_t format_result;
    CreateOutput(i, op_desc, output, format_result);

    std::vector<std::string> src_name = op_desc->GetSrcName();
    std::vector<int64_t> src_index = op_desc->GetSrcIndex();
    GE_CHK_BOOL_RET_STATUS(src_name.size() > i && src_index.size() > i, INTERNAL_ERROR,
                           "[Check][Param] construct output failed, as index:%u >= src name size:%zu, "
                           "or index >= src index size:%zu, op:%s.",
                           i, src_name.size(), src_index.size(), op_desc->GetName().c_str());
    // forward compatbility, if old om has no out_node_name, need to return output follow origin way
    if (out_size == out_node_name.size()) {
      // neweast plan, the index will add to name during generate model.
      bool contains_colon = out_node_name[i].find(":") != std::string::npos;
      output_name = contains_colon ? out_node_name[i] : out_node_name[i] + ":" + std::to_string(src_index[i]);
    } else {
      output_name = string("output_") + std::to_string(i) + "_" + src_name[i] + "_" + std::to_string(src_index[i]);
    }
    output.name = output_name;
    output_descs_.push_back(output);
    output_formats_.push_back(format_result);
  }

  return SUCCESS;
}

Status DavinciModel::GetOutputDescInfo(vector<InputOutputDescInfo> &output_descs,
                                       vector<uint32_t> &output_formats) const {
  output_descs.insert(output_descs.end(), output_descs_.begin(), output_descs_.end());
  output_formats.insert(output_formats.end(), output_formats_.begin(), output_formats_.end());
  return SUCCESS;
}

Status DavinciModel::CopyInputData(const InputData &input_data) {
  const std::vector<DataBuffer> &blobs = input_data.blobs;
  for (const auto &data : input_data_info_) {
    if (data.first >= blobs.size()) {
      REPORT_INNER_ERROR("E19999", "index:%u in input_data_info_ >= input_data.blobs.size:%zu, model_id:%u, "
                         "check invalid", data.first, blobs.size(), model_id_);
      GELOGE(FAILED, "[Check][Param] Blobs not match: blobs=%zu, tensor=%zu, index=%u, size=%ld, op_name(%s)",
             blobs.size(), input_data_info_.size(), data.first, data.second.GetDataInfo().at(0).first,
             data.second.GetOpName().c_str());
      return FAILED;
    }

    const DataBuffer &data_buf = blobs[data.first];
    rtMemcpyKind_t kind =
      data_buf.placement == kPlacementHostData ? RT_MEMCPY_HOST_TO_DEVICE : RT_MEMCPY_DEVICE_TO_DEVICE;
    if (data_buf.length == 0) {
      GELOGW("No data need to memcpy!");
      return SUCCESS;
    }
    uint64_t data_size = data.second.GetDataSize();
    GE_CHK_BOOL_RET_STATUS(data_size >= data_buf.length, PARAM_INVALID,
                           "[Check][Param] input data size(%lu) does not match model required size(%lu), "
                           "op_name(%s), ret failed.", data_buf.length, data_size, data.second.GetOpName().c_str());
    void *mem_addr = data.second.GetBasicAddr();
    void *data_buf_addr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(data_buf.data));
    uint64_t data_buf_length = data_buf.length;
    GELOGI("CopyPlainData memcpy graph_%u type[F] input[%s] rank[%u] dst[%p] src[%p] mem_size[%lu] datasize[%lu]",
           runtime_param_.graph_id, data.second.GetOpName().c_str(), data.first, mem_addr, data_buf_addr, data_size,
           data_buf_length);
    GE_CHK_RT_RET(rtMemcpy(mem_addr, data_size, data_buf_addr, data_buf_length, kind));
  }

  return SUCCESS;
}

Status DavinciModel::SyncVarData() {
  GELOGI("Sync var data, model id:%u", model_id_);

  if (global_step_addr_ != nullptr && global_step_size_ != 0) {
    const vector<uint64_t> v_step = { iterator_count_ };
    GE_CHK_RT_RET(rtMemcpy(global_step_addr_, global_step_size_, v_step.data(), v_step.size() * sizeof(uint64_t),
                           RT_MEMCPY_HOST_TO_DEVICE));
  }

  return SUCCESS;
}

Status DavinciModel::InitModelProfile() {
  for (const auto &task : task_list_) {
    GE_CHECK_NOTNULL(task);
    const FusionOpInfo *fusion_op_info = task->GetFusionOpInfo();
    // when type is RT_MODEL_TASK_KERNEL, ctx is not null
    if ((fusion_op_info == nullptr) || fusion_op_info->original_op_names.empty()) {
      continue;
    }

    GELOGI("task.id = %u, opNum = %zu", task->GetTaskID(), fusion_op_info->original_op_names.size());
    op_id_map_.insert(std::make_pair(fusion_op_info->op_index, task->GetTaskID()));
  }

  std::set<uint32_t> task_id_set;
  using CIT = std::multimap<uint32_t, uint32_t>::const_iterator;
  using Range = std::pair<CIT, CIT>;
  for (const auto &task : task_list_) {
    GE_CHECK_NOTNULL(task);
    const FusionOpInfo *fusion_op_info = task->GetFusionOpInfo();
    if ((fusion_op_info == nullptr) || fusion_op_info->original_op_names.empty()) {
      continue;
    }

    if (task_id_set.count(task->GetTaskID()) > 0) {
      continue;
    }

    const auto &op_desc = GetOpByIndex(fusion_op_info->op_index);
    GE_CHK_BOOL_EXEC(op_desc != nullptr,
                     REPORT_INNER_ERROR("E19999", "Get op by index failed, as index:%u out of range",
                                        fusion_op_info->op_index);
                     return FAILED,
                     "[Get][Op] failed, as index:%u out of range", fusion_op_info->op_index);

    ProfileInfo profile;
    profile.fusion_info = *fusion_op_info;
    Range range = op_id_map_.equal_range(fusion_op_info->op_index);
    for (CIT range_idx = range.first; range_idx != range.second; ++range_idx) {
      profile.task_count++;
      task_id_set.insert(range_idx->second);
    }

    // memory info
    TaskMemInfo &mem_info = profile.memory_info;
    const auto input_size = ModelUtils::GetInputSize(op_desc);
    const auto output_size = ModelUtils::GetOutputSize(op_desc);
    const auto workspace_size = ModelUtils::GetWorkspaceSize(op_desc);
    const auto weight_size = ModelUtils::GetWeightSize(op_desc);
    mem_info.input_size = std::accumulate(input_size.begin(), input_size.end(), 0);
    mem_info.output_size = std::accumulate(output_size.begin(), output_size.end(), 0);
    mem_info.workspace_size = std::accumulate(workspace_size.begin(), workspace_size.end(), 0);
    mem_info.weight_size = std::accumulate(weight_size.begin(), weight_size.end(), 0);
    mem_info.total_size = mem_info.weight_size + mem_info.input_size + mem_info.output_size + mem_info.workspace_size;

    profile_list_.emplace_back(profile);
  }

  GELOGI("fusion task size: %zu, profile info size: %zu", op_id_map_.size(), profile_list_.size());
  return SUCCESS;
}

Status DavinciModel::SinkModelProfile() {
  auto &prof_mgr = ProfilingManager::Instance();
  // Model Header
  std::string name = om_name_.empty() ? name_ : om_name_;
  uint32_t model_id = this->Id();
  int64_t start_time = this->GetLoadBeginTime();
  int64_t end_time = this->GetLoadEndTime();

  Json model_load_info;
  model_load_info[kModelName] = name;
  model_load_info[kModeleId] = model_id;
  model_load_info[kLoadStartTime] = start_time;
  model_load_info[kLoadEndTime] = end_time;
  // fusion op info
  using CIT = std::multimap<uint32_t, uint32_t>::const_iterator;
  using Range = std::pair<CIT, CIT>;
  for (const ProfileInfo &profile : profile_list_) {
    Json fusion_op_info;
    string fusion_op_name = profile.fusion_info.op_name;
    uint32_t op_num = profile.fusion_info.original_op_names.size();
    vector<string> original_name;
    for (uint32_t k = 0; k < op_num; k++) {
      original_name.emplace_back(profile.fusion_info.original_op_names[k]);
    }
    uint32_t stream_id = 0;
    auto iter = profiler_report_op_info_.find(fusion_op_name);
    if (iter != profiler_report_op_info_.end()) {
      stream_id = iter->second.second;
    }
    fusion_op_info[kFusionOpName] = fusion_op_name;
    fusion_op_info[kOriginalOpNum] = op_num;
    fusion_op_info[kOriginalOpName] = original_name;
    fusion_op_info[kStreamId] = stream_id;
    fusion_op_info[kFusionOpMemoryInfo][kInputSize] = profile.memory_info.input_size;
    fusion_op_info[kFusionOpMemoryInfo][kOutputSize] = profile.memory_info.output_size;
    fusion_op_info[kFusionOpMemoryInfo][kWeightSize] = profile.memory_info.weight_size;
    fusion_op_info[kFusionOpMemoryInfo][kWorkSpaceSize] = profile.memory_info.workspace_size;
    fusion_op_info[kFusionOpMemoryInfo][kTotalSize] = profile.memory_info.total_size;
    fusion_op_info[kTaskCount] = profile.task_count;
    vector<uint32_t> task_id;
    Range task_range = op_id_map_.equal_range(profile.fusion_info.op_index);
    for (CIT idx = task_range.first; idx != task_range.second; ++idx) {
      task_id.push_back(idx->second);
    }
    fusion_op_info[kTaskId] = task_id;
    model_load_info[kFusionOpInfo] += fusion_op_info;
  }

  std::string tag_name("model_load_info_" + std::to_string(this->Id()));
  std::string reported_data;
  try {
    reported_data = model_load_info.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    REPORT_INNER_ERROR("E19999", "Convert model_load_info JSON to string failed, model_id:%u, reason:%s",
                       model_id_, e.what());
    GELOGE(FAILED, "[Convert][JSON] to string failed, model_id:%u, reason:%s.", model_id_, e.what());
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Convert model_load_info JSON to string failed, model_id:%u", model_id_);
    GELOGE(FAILED, "[Convert][JSON] to string failed, model_id:%u.", model_id_);
  }
  reported_data.append(",")
               .append("\n");
  prof_mgr.ReportData(device_id_, reported_data, tag_name);
  return SUCCESS;
}

Status DavinciModel::SinkTimeProfile(const InputData &current_data) {
  auto &prof_mgr = ProfilingManager::Instance();

  string name = om_name_.empty() ? name_ : om_name_;
  Json model_time_info;
  model_time_info[kModelName] = name;
  model_time_info[kModeleId] = this->Id();
  model_time_info[kRequestId] = current_data.request_id;
  model_time_info[kThreadId] = mmGetTid();
  model_time_info[kInputBeginTime] = time_info_.processBeginTime;
  model_time_info[kInputEndTime] = time_info_.processEndTime;
  model_time_info[kInferBeginTime] = time_info_.inferenceBeginTime;
  model_time_info[kInferEndTime] = time_info_.inferenceEndTime;
  model_time_info[kOutputBeginTime] = time_info_.dumpBeginTime;
  model_time_info[kOutputEndTime] = time_info_.dumpEndTime;

  // report model data tag name
  std::string tag_name;
  tag_name.append("model_time_info_")
    .append(std::to_string(this->Id()))
    .append("_")
    .append(std::to_string(current_data.index));
  std::string reported_data;
  try {
    reported_data = model_time_info.dump(kInteval, ' ', false, Json::error_handler_t::ignore);
  } catch (std::exception &e) {
    REPORT_INNER_ERROR("E19999", "Convert model_time_info JSON to string failed, model_id:%u, reason:%s",
                       model_id_, e.what());
    GELOGE(FAILED, "[Convert][JSON] to string failed, model_id:%u, reason:%s.", model_id_, e.what());
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Convert model_time_info JSON to string failed, model_id:%u", model_id_);
    GELOGE(FAILED, "[Convert][JSON] to string failed, model_id:%u.", model_id_);
  }
  reported_data.append(",")
               .append("\n");
  prof_mgr.ReportData(device_id_, reported_data, tag_name);

  return SUCCESS;
}

void DavinciModel::SetProfileTime(ModelProcStage stage, int64_t endTime) {
  int64_t time = endTime;

  if (time == 0) {
    mmTimespec timespec = mmGetTickCount();
    time = timespec.tv_sec * 1000 * 1000 * 1000 + timespec.tv_nsec;  // 1000 ^ 3 converts second to nanosecond
  }

  switch (stage) {
    case MODEL_LOAD_START:
      load_begin_time_ = time;
      break;
    case MODEL_LOAD_END:
      load_end_time_ = time;
      break;
    case MODEL_PRE_PROC_START:
      time_info_.processBeginTime = time;
      break;
    case MODEL_PRE_PROC_END:
      time_info_.processEndTime = time;
      break;
    case MODEL_INFER_START:
      time_info_.inferenceBeginTime = time;
      break;
    case MODEL_INFER_END:
      time_info_.inferenceEndTime = time;
      break;
    case MODEL_AFTER_PROC_START:
      time_info_.dumpBeginTime = time;
      break;
    case MODEL_AFTER_PROC_END:
      time_info_.dumpEndTime = time;
      break;
    default:
      break;
  }
  return;
}

///
/// @ingroup ge
/// @brief send Output Op result to upper layer
/// @already malloced in ModelLoad, no need to malloc again
/// @param [in] data_id: the index of output_data
/// @param [in/out] output_data: real user output_data
/// @param [in] kind: the kind of rtMemcpy
/// @return Status result
/// @author
///
Status DavinciModel::CopyOutputData(uint32_t data_id, OutputData &output_data, rtMemcpyKind_t kind) {
  if (!has_output_node_) {
    return SyncVarData();
  }

  output_data.index = data_id;
  output_data.model_id = model_id_;
  if (output_data.blobs.size() != output_data_info_.size()) {
    REPORT_INNER_ERROR("E19999", "output_data.blobs.size:%zu != output_data_info.size:%zu, model_id:%u, "
                       "check invalid", output_data.blobs.size(), output_data_info_.size(), model_id_);
    GELOGE(FAILED, "[Check][Param] output_data.blobs.size:%zu != output_data_info.size:%zu, model_id:%u",
           output_data.blobs.size(), output_data_info_.size(), model_id_);
    return FAILED;
  }

  std::vector<DataBuffer> &blobs = output_data.blobs;
  size_t idx = 0;
  for (const auto &output : output_data_info_) {
    if (output.first >= blobs.size()) {
      REPORT_INNER_ERROR("E19999", "index:%u in output_data_info_ >= output_data.blobs.size:%zu, model_id:%u, "
                         "check invalid", output.first, blobs.size(), model_id_);
      GELOGE(FAILED, "[Check][Param] index:%u in output_data_info_ >= output_data.blobs.size:%zu, model_id:%u",
             output.first, blobs.size(), model_id_);
      return FAILED;
    }

    if ((kind == RT_MEMCPY_DEVICE_TO_DEVICE) && (copy_only_addrs_.count(output.second.GetBasicAddr()) == 0)) {
      continue;  // Skip: Feed by zero copy.
    }

    DataBuffer &buffer = blobs[output.first];
    uint64_t mem_size = static_cast<uint64_t>(output.second.GetDataSize());
    if ((buffer.length == 0) || (mem_size == 0)) {
      GELOGI("Length of data is zero, No need copy. output tensor index=%u", output.first);
      continue;
    }
    if (is_dynamic_) {
      GELOGI("No need to check output data size.");
    } else if (buffer.length < mem_size) {
      REPORT_INNER_ERROR("E19999", "Buffer.length:%lu in output blob < mem_size:%lu in output_data_info, index:%u, "
                         "model_id:%u, check invalid", buffer.length, mem_size, output.first, model_id_);
      GELOGE(FAILED, "[Check][Param] Buffer.length:%lu in output blob < mem_size:%lu in output_data_info, index:%u, "
             "model_id:%u", buffer.length, mem_size, output.first, model_id_);
      return FAILED;
    } else if (buffer.length > mem_size) {
      GELOGW("Tensor data size=%lu, buffer size=%lu", mem_size, buffer.length);
    }
    int64_t data_size = output.second.GetDataSize();

    if (is_online_infer_dynamic_) {
      if (merge_nodes_gear_and_real_out_size_info_.find(idx) != merge_nodes_gear_and_real_out_size_info_.end()) {
        auto gear_and_real_out_size_info = merge_nodes_gear_and_real_out_size_info_[idx];
        data_size = gear_and_real_out_size_info[cur_dynamic_dims_];
      }
    }
    uint64_t buffer_length = buffer.length;
    void *buffer_addr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(buffer.data));

    GELOGI("CopyPlainData memcpy graph_%u type[F] output[%u] memaddr[%p] mem_size[%lu] datasize[%lu]",
           runtime_param_.graph_id, output.first, output.second.GetBasicAddr(), data_size, buffer_length);
    GE_CHK_RT_RET(rtMemcpy(buffer_addr, buffer_length, output.second.GetBasicAddr(), data_size, kind));
    idx++;
  }
  return SUCCESS;
}

Status DavinciModel::InitOutputTensorInfo(const OpDescPtr &op_desc) {
  size_t input_num = op_desc->GetInputsSize();
  if (is_getnext_sink_dynamic_) {
    input_num = input_num - kGetDynamicDimsCount;
  }

  for (size_t i = 0; i < input_num; ++i) {
    int64_t size = 0;
    auto input_desc = op_desc->GetInputDescPtr(i);
    GE_CHECK_NOTNULL(input_desc);
    auto ret = TensorUtils::GetTensorSizeInBytes(*input_desc, size);
    GE_IF_BOOL_EXEC(ret != GRAPH_SUCCESS,
                    REPORT_INNER_ERROR("E19999", "Get input TensorSize in op:%s(%s) failed, input_index:%zu, "
                                       "model_id:%u", op_desc->GetName().c_str(), op_desc->GetType().c_str(), i,
                                       model_id_);
                    GELOGE(ret, "[Get][InputTensorSize] in op:%s(%s) failed, input_index:%zu, model_id:%u",
                           op_desc->GetName().c_str(), op_desc->GetType().c_str(), i, model_id_);
                    return ret);
    const GeShape &shape = input_desc->GetShape();
    GELOGI("Output size is %ld, output shape is %s.", size, formats::JoinToString(shape.GetDims()).c_str());
    output_buffer_size_.emplace_back(size);
    output_shape_info_.emplace_back(shape);
  }

  return SUCCESS;
}

Status DavinciModel::GenOutputTensorInfo(OutputData *output_data, vector<ge::Tensor> &outputs) {
  GE_CHECK_NOTNULL(output_data);
  if (!output_data->blobs.empty()) {
    GELOGI("No need to generate output tensor info, model id:%u", model_id_);
    return SUCCESS;
  }

  vector<int64_t> output_buffer_size;
  vector<vector<int64_t>> output_shape_info;
  size_t output_num = output_buffer_size_.size();
  for (size_t i = 0; i < output_num; ++i) {
    int64_t output_size = output_buffer_size_[i];
    vector<int64_t> output_shape = output_shape_info_[i].GetDims();
    if (is_online_infer_dynamic_) {
      if (merge_nodes_gear_and_real_out_size_info_.find(i) != merge_nodes_gear_and_real_out_size_info_.end()) {
        auto gear_and_real_out_size_info = merge_nodes_gear_and_real_out_size_info_[i];
        output_size = gear_and_real_out_size_info[cur_dynamic_dims_];
        auto gear_and_real_out_shape_info = merge_nodes_gear_and_real_out_shape_info_[i];
        output_shape = gear_and_real_out_shape_info[cur_dynamic_dims_];
        is_dynamic_ = true;
      }
    }
    GELOGI("Output size is %ld, output shape is %s.", output_size, formats::JoinToString(output_shape).c_str());
    output_buffer_size.push_back(output_size);
    output_shape_info.push_back(output_shape);
  }

  GELOGI("Output blobs size:%zu, model id:%u", output_buffer_size_.size(), model_id_);
  for (size_t i = 0; i < output_buffer_size.size(); ++i) {
    auto aligned_ptr = MakeShared<AlignedPtr>(output_buffer_size[i], kAlignment);
    GE_CHECK_NOTNULL(aligned_ptr);
    GeShape ge_shape(output_shape_info[i]);
    GeTensorDesc tensor_desc;
    tensor_desc.SetShape(ge_shape);
    GeTensor ge_tensor(tensor_desc);
    ge_tensor.SetData(aligned_ptr, output_buffer_size[i]);
    ge::Tensor output_tensor = TensorAdapter::AsTensor(ge_tensor);

    auto data_ptr = aligned_ptr->MutableGet();
    output_data->blobs.push_back(
      {reinterpret_cast<void *>(data_ptr), static_cast<uint64_t>(output_buffer_size[i]), false});
    outputs.emplace_back(std::move(output_tensor));
    GELOGD("Output index:%zu, output dims is %s, data length:%lu.", i,
           formats::JoinToString(output_shape_info[i]).c_str(), output_buffer_size[i]);
  }

  return SUCCESS;
}
///
/// @ingroup ge
/// @brief send Output Op result to upper layer
/// @already malloced in ModelLoad, no need to malloc again
/// @param [in] data_id: the index of output_data
/// @param [in] rslt_flg: result flag
/// @param [in] seq_end_flag: sequence end flag
/// @param [out] output_data: real user output_data
/// @return Status result
/// @author
///
Status DavinciModel::ReturnResult(uint32_t data_id, const bool rslt_flg, const bool seq_end_flag,
                                  OutputData *output_data) {
  GE_CHK_BOOL_EXEC(listener_ != nullptr,
                   REPORT_INNER_ERROR("E19999", "listener_ is nullptr, check invalid.");
                   return PARAM_INVALID, "[Check][Param] listener_ is null.");
  std::vector<ge::Tensor> outputs;

  // return result is not required
  if (!rslt_flg && !seq_end_flag) {
    GELOGW("Compute failed, model id: %u", model_id_);
    auto model_manager = ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    auto exception_infos = model_manager->GetExceptionInfos();
    if (exception_infos.size() > 0) {
      GE_CHK_STATUS_RET(DumpExceptionInfo(exception_infos),
                        "[Dump][Exception] Dump exception info failed, model_id:%u.", model_id_);
    } else {
      GELOGI("[Dump][Exception] Exception info is null.");
    }
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, INTERNAL_ERROR, outputs),
                  "[Call][OnComputeDone] failed, model_id:%u, data_id:%u.", model_id_, data_id);
    return INTERNAL_ERROR;
  }

  if (!has_output_node_) {
    GELOGW("The tensor list of output is empty, model id: %u", model_id_);
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, INTERNAL_ERROR, outputs),
                  "[Call][OnComputeDone] failed, model_id:%u, data_id:%u.", model_id_, data_id);
    return INTERNAL_ERROR;
  }

  GE_CHECK_NOTNULL(output_data);
  output_data->index = data_id;
  output_data->model_id = model_id_;

  if (is_getnext_sink_dynamic_) {
    GELOGD("Reinit cur dynamic dims when getnext sink dynamic.");
    cur_dynamic_dims_.clear();
    cur_dynamic_dims_.resize(shape_of_cur_dynamic_dims_);
    auto ret = rtMemcpy(cur_dynamic_dims_.data(), shape_of_cur_dynamic_dims_ * sizeof(int32_t),
                        netoutput_last_input_addr_, netoutput_last_input_size_, RT_MEMCPY_DEVICE_TO_HOST);
    GE_CHK_RT_RET(ret);
  }

  GELOGD("Cur dynamic dims is %s.", formats::JoinToString(cur_dynamic_dims_).c_str());
  if (GenOutputTensorInfo(output_data, outputs) != SUCCESS) {
    return INTERNAL_ERROR;
  }

  if (CopyOutputData(data_id, *output_data, RT_MEMCPY_DEVICE_TO_HOST) != SUCCESS) {
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, INTERNAL_ERROR, outputs),
                  "[Call][OnComputeDone] failed, model_id:%u, data_id:%u.", model_id_, data_id);
    return INTERNAL_ERROR;
  }

  if (seq_end_flag) {
    GELOGW("End of sequence, model id: %u", model_id_);
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, END_OF_SEQUENCE, outputs),
                  "[Call][OnComputeDone] failed, model_id:%u, data_id:%u.", model_id_, data_id);
    return END_OF_SEQUENCE;
  }
  GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, SUCCESS, outputs),
                "[Call][OnComputeDone] failed, model_id:%u, data_id:%u.", model_id_, data_id);
  return SUCCESS;
}
///
/// @ingroup ge
/// @brief return not output to upper layer for cloud case
/// @param [in] data_id
/// @return Status result
///
Status DavinciModel::ReturnNoOutput(uint32_t data_id) {
  GELOGI("ReturnNoOutput model id:%u.", model_id_);

  GE_CHK_BOOL_EXEC(listener_ != nullptr,
                   REPORT_INNER_ERROR("E19999", "listener_ is nullptr, check invalid.");
                   return PARAM_INVALID, "[Check][Param] listener_ is null!");
  std::vector<ge::Tensor> outputs;
  GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_id, SUCCESS, outputs),
                "[Call][OnComputeDone] failed, model_id:%u, data_id:%u.", model_id_, data_id);
  return SUCCESS;
}

void *DavinciModel::Run(DavinciModel *model) {
  GE_CHK_BOOL_EXEC(model != nullptr,
                   return nullptr, "[Check][Param] model_pointer is null!")
  bool seq_end_flag = false;
  uint32_t model_id = model->Id();
  uint32_t device_id = model->GetDeviceId();
  ErrorManager::GetInstance().SetErrorContext(model->GetErrorContext());

  GELOGI("Model Run thread start, model_id:%u.", model_id);
  rtError_t rt_ret = rtSetDevice(static_cast<int32_t>(device_id));
  if (rt_ret != RT_ERROR_NONE) {

    GELOGE(FAILED, "[Run][Rtsetdevice] failed, model_id:%u, device_id:%u.", model_id, device_id);
    return nullptr;
  }
  // DeviceReset before thread run finished!
  GE_MAKE_GUARD(not_used_var, [&] { GE_CHK_RT(rtDeviceReset(device_id)); });

  ErrorManager::GetInstance().SetStage(error_message::kModelExecute, error_message::kModelExecute);
  while (model->RunFlag()) {
    // Model hasn't truly started runing before received data
    model->SetRunningFlag(false);
    bool rslt_flg = true;
    if (model->GetDataInputer() == nullptr) {
      GELOGW("Data inputer is nullptr.");
      break;
    }

    std::shared_ptr<InputDataWrapper> data_wrapper;
    Status ret = model->GetDataInputer()->Pop(data_wrapper);
    // Model run indeedly start after received data.
    model->SetRunningFlag(true);
    if (data_wrapper == nullptr || ret != SUCCESS) {
      GELOGI("data_wrapper is null!");
      continue;
    }
    GELOGI("Getting the input data, model_id:%u", model_id);
    GE_IF_BOOL_EXEC(!model->RunFlag(), break);

    InputData current_data = data_wrapper->GetInput();
    GELOGI("Model thread Run begin, model id:%u, data index:%u.", model_id, current_data.index);
    GE_TIMESTAMP_START(Model_SyncVarData);
    ret = model->SyncVarData();
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ret != SUCCESS, (void)model->ReturnResult(current_data.index, false, false, data_wrapper->GetOutput());
        continue,
        "[Call][SyncVarData] Copy input data to model failed, model_id:%u.", model_id);  // [No need to check value]
    GE_IF_BOOL_EXEC(model->is_first_execute_, GE_TIMESTAMP_EVENT_END(Model_SyncVarData, "Model Run SyncVarData"));

    GELOGI("Copy input data, model id:%u", model_id);
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(),
                    model->SetProfileTime(MODEL_PRE_PROC_START));
    ret = model->CopyInputData(current_data);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ret != SUCCESS, (void)model->ReturnResult(current_data.index, false, false, data_wrapper->GetOutput());
        continue,
        "[Call][CopyInputData] Copy input data to model failed, model_id:%u.", model_id);  // [No need to check value]
    if (model->is_online_infer_dynamic_ && !model->is_getnext_sink_dynamic_) {
      model->cur_dynamic_dims_.clear();
      GE_IF_BOOL_EXEC(current_data.blobs.empty(), break);
      auto shape_data_buffer_data = current_data.blobs.back().data;
      auto shape_data_buffer_length = current_data.blobs.back().length;
      model->cur_dynamic_dims_.assign(reinterpret_cast<int32_t *>(shape_data_buffer_data),
                                      reinterpret_cast<int32_t *>(shape_data_buffer_data) +
                                      shape_data_buffer_length / sizeof(int32_t));
      GELOGD("Data: cur dynamic dims is %s", formats::JoinToString(model->cur_dynamic_dims_).c_str());
      delete[] reinterpret_cast<int32_t *>(current_data.blobs.back().data);
      current_data.blobs.pop_back();
    }
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), model->SetProfileTime(MODEL_PRE_PROC_END));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), model->SetProfileTime(MODEL_INFER_START));
    GE_TIMESTAMP_START(rtModelExecute);
    GELOGI("rtModelExecute start.");
    rt_ret = rtModelExecute(model->rt_model_handle_, model->rt_model_stream_, 0);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE, rslt_flg = false;
                    (void)model->ReturnResult(current_data.index, false, false, data_wrapper->GetOutput());
                    continue);
    GELOGI("rtModelExecute end");
    GE_IF_BOOL_EXEC(model->is_first_execute_, GE_TIMESTAMP_EVENT_END(rtModelExecute, "GraphExcute::rtModelExecute"));

    GE_TIMESTAMP_START(rtStreamSynchronize);
    GELOGI("rtStreamSynchronize start.");
    rt_ret = rtStreamSynchronize(model->rt_model_stream_);
    if (rt_ret == kEndOfSequence || rt_ret == kEndOfSequenceNew) {
      seq_end_flag = true;
    }
    if (rt_ret == kModelAbortNormal || rt_ret == kModelAbortNormalNew) {
      GELOGI("The model with multiple datasets aborts normally.");
    } else {
      GE_IF_BOOL_EXEC(
        rt_ret != RT_ERROR_NONE, rslt_flg = false; GELOGI("seq_end_flg: %d", seq_end_flag);
        (void)model->ReturnResult(current_data.index, false, seq_end_flag,
                                  data_wrapper->GetOutput());  // [No need to check value]
        continue);
    }

    GELOGI("rtStreamSynchronize end.");
    GE_IF_BOOL_EXEC(model->is_first_execute_,
                    GE_TIMESTAMP_EVENT_END(rtStreamSynchronize, "GraphExcute::Wait for rtStreamSynchronize"));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), model->SetProfileTime(MODEL_INFER_END));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(),
                    model->SetProfileTime(MODEL_AFTER_PROC_START));
    GE_TIMESTAMP_START(ReturnResult3);
    // copy output data from device to host
    GE_IF_BOOL_EXEC(model->has_output_node_,
                    (void)model->ReturnResult(current_data.index, rslt_flg, false, data_wrapper->GetOutput()));
    // copy output data from device to host for variable graph
    GE_IF_BOOL_EXEC(!model->has_output_node_, (void)model->ReturnNoOutput(current_data.index));
    GE_IF_BOOL_EXEC(model->is_first_execute_,
                    GE_TIMESTAMP_EVENT_END(ReturnResult3, "GraphExcute::CopyDataFromDeviceToHost"));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(),
                    model->SetProfileTime(MODEL_AFTER_PROC_END));
    GE_IF_BOOL_EXEC(ProfilingManager::Instance().ProfilingModelExecuteOn(), (void)model->SinkTimeProfile(current_data));

    model->iterator_count_++;
    model->is_first_execute_ = false;
    // model run finished
    model->SetRunningFlag(false);
    GELOGI("run iterator count is %lu, model_id:%u", model->iterator_count_, model->model_id_);
  }

  GELOGI("Model run end, model id:%u", model->model_id_);
  return nullptr;
}

///
/// @ingroup ge
/// @brief call API provided by data inputer to destroy thread
/// @param [in] no
/// @return Status Destroy result
/// @author
///
Status DavinciModel::DestroyThread() {
  run_flg_ = false;

  if (data_inputer_ != nullptr) {
    data_inputer_->Stop();
  }

  if (thread_id_.joinable()) {
    thread_id_.join();
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief create model std::thread,
/// @brief start to execute Model
/// @param [in] no
/// @return Status create model thread and execute result
/// @author
///
Status DavinciModel::ModelRunStart() {
  GE_CHK_BOOL_RET_STATUS(data_inputer_ != nullptr, INTERNAL_ERROR,
                         "[Check][Param] data_inputer_ is nullptr, model id:%u.", model_id_);

  LockRunFlg();
  GE_MAKE_GUARD(tmp_lock, [&] { UnlockRunFlg(); });

  GE_CHK_BOOL_RET_STATUS(!run_flg_, INTERNAL_ERROR, "[Check][Param] Model already started, model id:%u.", model_id_);

  run_flg_ = true;

  // create stream instance which rt_model_handel is running on
  GE_CHK_RT_RET(rtStreamCreate(&rt_model_stream_, priority_));
  is_inner_model_stream_ = true;

  string opt = "0";
  (void)ge::GetContext().GetOption(OPTION_GE_MAX_DUMP_OP_NUM, opt);  // option may not be set up, no need to check value
  int64_t maxDumpOpNum = std::strtol(opt.c_str(), nullptr, kDecimal);
  maxDumpOpNum_ = maxDumpOpNum;

  error_context_ = ErrorManager::GetInstance().GetErrorManagerContext();
  CREATE_STD_THREAD(thread_id_, DavinciModel::Run, this);
  GELOGI("model thread create success, model id:%u.", model_id_);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief call API provided by data inputer and destroy model Thread
/// @param [in] no
/// @return Status Destroy result
/// @author
///
Status DavinciModel::ModelRunStop() {
  LockRunFlg();
  GE_MAKE_GUARD(tmp_lock, [&] { UnlockRunFlg(); });

  GE_CHK_STATUS_RET(DestroyThread(), "[Destoy][Thead] failed, model id:%u.", model_id_);

  return SUCCESS;
}

void DavinciModel::UnbindTaskSinkStream() {
  // unbinding hcom stream
  UnbindHcomStream();
  if (is_stream_list_bind_) {
    for (size_t i = 0; i < stream_list_.size(); i++) {
      // unbind rt_model_handle and streams
      GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, stream_list_[i]) != RT_ERROR_NONE,
                 "Unbind stream from model failed! Index: %zu", i);
    }
  }

  if (is_inner_model_stream_) {
    if (!input_queue_ids_.empty() || !output_queue_ids_.empty()) {
      GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_model_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
    }
    // destroy stream that is bound with rt_model
    GE_LOGW_IF(rtStreamDestroy(rt_model_stream_) != RT_ERROR_NONE, "Destroy stream for rt_model failed.")
  }

  if (is_pure_head_stream_ && rt_head_stream_ != nullptr) {
    GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_head_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
    GE_LOGW_IF(rtStreamDestroy(rt_head_stream_) != RT_ERROR_NONE, "Destroy stream for rt_model failed.");
    rt_head_stream_ = nullptr;
  }

  if (rt_entry_stream_ != nullptr) {
    GE_LOGW_IF(rtModelUnbindStream(rt_model_handle_, rt_entry_stream_) != RT_ERROR_NONE, "Unbind stream failed!");
    GE_LOGW_IF(rtStreamDestroy(rt_entry_stream_) != RT_ERROR_NONE, "Destroy stream for rt_model failed.");
    rt_entry_stream_ = nullptr;
  }
}

void *DavinciModel::GetRunAddress(void *addr) const {
  if (fixed_mem_base_ == reinterpret_cast<uintptr_t>(mem_base_)) {
    return addr;
  }

  uintptr_t ptr = reinterpret_cast<uintptr_t>(addr);
  if ((fixed_mem_base_ <= ptr) && (ptr < fixed_mem_base_ + runtime_param_.mem_size)) {
    return mem_base_ + (ptr - fixed_mem_base_);
  } else {
    return addr;
  }
}

Status DavinciModel::CreateKnownZeroCopyMap(const vector<void *> &inputs, const vector<void *> &outputs) {
  GELOGI("in, inputs size: %zu, input addr size: %zu, outputs size: %zu, output addr size: %zu",
         inputs.size(), input_addrs_list_.size(), outputs.size(), output_addrs_list_.size());
  if (inputs.size() > input_addrs_list_.size()) {
    REPORT_INNER_ERROR("E19999", "input data addr %zu should less than input op num %zu.",
                       inputs.size(), input_addrs_list_.size());
    GELOGE(FAILED, "[Check][Param] input data addr %zu should less than input op num %zu.",
           inputs.size(), input_addrs_list_.size());
    return FAILED;
  }
  // remove zero copy addr in last iteration
  known_input_data_info_.clear();
  known_output_data_info_.clear();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const vector<void *> &addr_list = input_addrs_list_[i];
    void *addr = GetRunAddress(addr_list[kDataIndex]);
    known_input_data_info_[addr] = inputs[i];
    GELOGI("input %zu, v addr %p, r addr %p, p addr %p", i, addr_list[kDataIndex], addr, inputs[i]);
  }

  if (!has_output_node_) {
    GELOGW("output op num in graph is %zu", output_addrs_list_.size());
    return SUCCESS;
  }
  const vector<void *> &addr_list = output_addrs_list_.front();
  for (size_t i = 0; i < addr_list.size() && i < outputs.size(); ++i) {
    void *addr = GetRunAddress(addr_list[i]);
    known_output_data_info_[addr] = outputs[i];
    GELOGI("output %zu, v addr %p, r addr %p, p addr %p", i, addr_list[i], addr, outputs[i]);
  }

  GELOGI("create map for zero copy success, known input data info size: %zu, known output data info size: %zu",
         known_input_data_info_.size(), known_output_data_info_.size());
  return SUCCESS;
}

void DavinciModel::SetTotalIOAddrs(const vector<void *> &io_addrs) {
  if (fixed_mem_base_ == reinterpret_cast<uintptr_t>(mem_base_)) {
    total_io_addrs_.insert(total_io_addrs_.end(), io_addrs.begin(), io_addrs.end());
    return;
  }

  for (size_t i = 0; i < io_addrs.size(); ++i) {
    total_io_addrs_.emplace_back(GetRunAddress(io_addrs[i]));
  }
}

Status DavinciModel::UpdateKnownZeroCopyAddr(vector<void *> &total_io_addrs, bool update_args) {
  if (fixed_mem_base_ != reinterpret_cast<uintptr_t>(mem_base_) && update_args) {
    for (size_t i = 0; i < total_io_addrs.size(); ++i) {
      total_io_addrs[i] = GetRunAddress(total_io_addrs[i]);
    }
  }

  for (size_t i = 0; i < total_io_addrs.size(); ++i) {
    auto it_in = known_input_data_info_.find(total_io_addrs[i]);
    if (it_in != known_input_data_info_.end()) {
      GELOGI("input %zu, v addr %p, p addr %p", i, total_io_addrs[i], known_input_data_info_.at(total_io_addrs[i]));
      total_io_addrs[i] = known_input_data_info_.at(total_io_addrs[i]);
    }
    auto it_out = known_output_data_info_.find(total_io_addrs[i]);
    if (it_out != known_output_data_info_.end()) {
      GELOGI("output %zu, v addr %p, p addr %p", i, total_io_addrs[i], known_output_data_info_.at(total_io_addrs[i]));
      total_io_addrs[i] = known_output_data_info_.at(total_io_addrs[i]);
    }
  }
  GELOGI("update known zero copy addr success, total io addrs size: %zu", total_io_addrs.size());
  return SUCCESS;
}

Status DavinciModel::UpdateKnownNodeArgs(const vector<void *> &inputs, const vector<void *> &outputs) {
  GELOGI("DavinciModel::UpdateKnownNodeArgs begin");
  GE_CHK_STATUS_RET(CreateKnownZeroCopyMap(inputs, outputs),
                    "[Call][CreateKnownZeroCopyMap] failed, model_id:%u.", model_id_);
  total_io_addrs_.clear();
  for (size_t task_index = 0; task_index < task_list_.size(); ++task_index) {
    auto &task = task_list_[task_index];
    if (task != nullptr) {
      Status ret = task->UpdateArgs();
      if (ret != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "task %zu update args failed, model_id:%u", task_index, model_id_);
        GELOGE(FAILED, "[Update][Args] to task %zu failed, model_id:%u.", task_index, model_id_);
        return FAILED;
      }
    }
  }
  GE_CHK_STATUS_RET(UpdateKnownZeroCopyAddr(total_io_addrs_, false),
                    "[Call][UpdateKnownZeroCopyAddr] failed, model_id:%u.", model_id_);

  if (total_args_size_ == 0) {
    GELOGW("DavinciModel::UpdateKnownNodeArgs device args %p, dst size %u, pass rtMemcpy.", args_, total_args_size_);
  } else {
    uint32_t total_addr_size = total_io_addrs_.size() * sizeof(uint64_t);
    GELOGI("DavinciModel::UpdateKnownNodeArgs device args %p, dst size %u, src size %u", args_, total_args_size_,
           total_addr_size);

    Status rt_ret =
        rtMemcpy(args_, total_args_size_, total_io_addrs_.data(), total_addr_size, RT_MEMCPY_HOST_TO_DEVICE);
    GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%u, ret:0x%X", total_args_size_ , rt_ret);
                    GELOGE(rt_ret, "[Call][RtMemcpy] failed, size:%u, ret:0x%X", total_args_size_ , rt_ret);
                    return FAILED;)
  }

  GELOGI("DavinciModel::UpdateKnownNodeArgs success");
  return SUCCESS;
}

Status DavinciModel::InitTaskInfo(domi::ModelTaskDef &model_task_def) {
  GELOGI("InitTaskInfo in, task size %d", model_task_def.task().size());
  task_list_.resize(model_task_def.task_size());
  for (int i = 0; i < model_task_def.task_size(); ++i) {
    // dynamic shape will create task_list_ before
    const domi::TaskDef &task = model_task_def.task(i);
    if (this->task_list_[i] == nullptr) {
      task_list_[i] = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(task.type()));
    }
    GE_CHECK_NOTNULL(task_list_[i]);
    Status ret = task_list_[i]->Init(task, this);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Task index:%d init failed, ret:%d.", i, ret);
      GELOGE(ret, "[Init][Task] index:%d failed, ret:%d.", i, ret);
      return ret;
    }
  }
  GELOGI("InitTaskInfo out");
  return SUCCESS;
}

Status DavinciModel::CheckCapability(rtFeatureType_t featureType, int32_t featureInfo, bool &is_support) const {
  int64_t value = RT_CAPABILITY_SUPPORT;
  auto rt_ret = rtGetRtCapability(featureType, featureInfo, &value);
  GE_CHK_BOOL_RET_STATUS(rt_ret == RT_ERROR_NONE, FAILED, "[Call][RtGetRtCapability] failed, ret:0x%X", rt_ret);
  is_support = (value == RT_CAPABILITY_SUPPORT) ? true : false;
  return SUCCESS;
}

Status DavinciModel::MallocKnownArgs() {
  GELOGI("DavinciModel::MallocKnownArgs in");
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  if (model_task_def->task_size() == 0) {
    GELOGW("DavinciModel::MallocKnownArgs davincimodel has no task info.");
    return SUCCESS;
  }
  task_list_.resize(model_task_def->task_size());
  for (int32_t i = 0; i < model_task_def->task_size(); ++i) {
    const domi::TaskDef &taskdef = model_task_def->task(i);
    task_list_[i] = TaskInfoFactory::Instance().Create(static_cast<rtModelTaskType_t>(taskdef.type()));
    GE_CHECK_NOTNULL(task_list_[i]);
    Status ret = task_list_[i]->CalculateArgs(taskdef, this);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "task index:%d CalculateArgs failed, ret:%d", i, ret);
      GELOGE(ret, "[Calculate][Args] for taskdef index:%d failed, ret:%d", i, ret);
      return ret;
    }
  }
  rtError_t rt_ret;
  bool is_support = false;
  GE_CHK_STATUS_RET_NOLOG(CheckCapability(FEATURE_TYPE_MEMORY, MEMORY_INFO_TS_4G_LIMITED, is_support));
  auto mem_type = is_support ? RT_MEMORY_TS_4G : RT_MEMORY_HBM;
  // malloc args memory
  if (total_args_size_ != 0) {
    rt_ret = rtMalloc(&args_, total_args_size_, mem_type);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret: 0x%X", total_args_size_, rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret: 0x%X", total_args_size_, rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }
  // malloc dynamic and static hybrid memory
  if (total_hybrid_args_size_ != 0) {
    rt_ret = rtMalloc(&hybrid_addrs_, total_hybrid_args_size_, mem_type);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret: 0x%X", total_hybrid_args_size_, rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret: 0x%X", total_hybrid_args_size_, rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }
  // malloc fixed addr memory, eg: rts op
  if (total_fixed_addr_size_ != 0) {
    GELOGI("Begin to allocate fixed addr.");
    rt_ret = rtMalloc(&fixed_addrs_, total_fixed_addr_size_, mem_type);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret: 0x%X", total_hybrid_args_size_, rt_ret);
      GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret: 0x%X", total_hybrid_args_size_, rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }

  GELOGI("DavinciModel::MallocKnownArgs success, total args size %u. total fixed addr size %ld", total_args_size_,
         total_fixed_addr_size_);
  return SUCCESS;
}

void DavinciModel::SaveProfilingTaskDescInfo(const OpDescPtr &op, const TaskInfoPtr &task,
                                             const domi::TaskDef &task_def, size_t task_index) {
  bool flag = GetL1FusionEnableOption();
  char skt_enable_env[MMPA_MAX_PATH] = { 0x00 };
  INT32 res = mmGetEnv("SKT_ENABLE", skt_enable_env, MMPA_MAX_PATH);
  int64_t env_flag = (res == EN_OK) ? std::strtol(skt_enable_env, nullptr, kDecimal) : 0;
  if (env_flag != 0) {
    flag = true;
  }

  TaskDescInfo task_desc_info;
  if (!om_name_.empty()) {
    task_desc_info.model_name = om_name_;
  } else {
    task_desc_info.model_name = name_;
  }
  task_desc_info.op_name = op->GetName();
  task_desc_info.op_type = op->GetType();
  task_desc_info.block_dim = task_def.kernel().block_dim();
  task_desc_info.task_id = task->GetTaskID();
  task_desc_info.stream_id = task->GetStreamId();
  task_desc_info.shape_type = "static";
  task_desc_info.cur_iter_num = 0;
  task_desc_info.task_type = kTaskTypeInvalid;
  auto &prof_mgr = ProfilingManager::Instance();
  prof_mgr.GetOpInputOutputInfo(op, task_desc_info);
  auto model_task_type = static_cast<rtModelTaskType_t>(task_def.type());
  if (model_task_type == RT_MODEL_TASK_KERNEL) {
    const domi::KernelDef &kernel_def = task_def.kernel();
    const auto &context = kernel_def.context();
    auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
    if (kernel_type == ccKernelType::TE) {
      task_desc_info.task_type = kTaskTypeAicore;
    } else if (kernel_type == ccKernelType::AI_CPU || kernel_type == ccKernelType::CUST_AI_CPU) {
      task_desc_info.task_type = kTaskTypeAicpu;
    } else {
      GELOGD("Other kernel type: %u", context.kernel_type());
    }
  } else if (model_task_type == RT_MODEL_TASK_KERNEL_EX) {
    task_desc_info.task_type = kTaskTypeAicpu;
  } else {
    GELOGD("Skip task type: %d", static_cast<int>(model_task_type));
  }
  profiler_report_op_info_[task_desc_info.op_name] =
    std::pair<uint32_t, uint32_t>(task_desc_info.task_id, task_desc_info.stream_id);
  task_desc_info_.emplace_back(task_desc_info);
  if (flag) {
    if (task->GetSktTaskID() != 0xFFFFFFFF) {
      TaskDescInfo task_desc_info;
      string op_name = "super_kernel_" + to_string(task_index);
      task_desc_info.op_name = op_name;
      task_desc_info.task_id = task->GetSktTaskID();
      profiler_report_op_info_[task_desc_info.op_name] =
        std::pair<uint32_t, uint32_t>(task_desc_info.task_id, task_desc_info.stream_id);
      task_desc_info_.emplace_back(task_desc_info);
    }
  }
}

Status DavinciModel::DistributeTask() {
  GELOGI("do Distribute.");
  for (auto &task : cpu_task_list_) {
    if (task == nullptr) {
      GELOGW("task is null");
      continue;
    }
    GE_CHK_STATUS_RET(task->Distribute());
  }

  task_desc_info_.clear();
  const auto &model_task_def = ge_model_->GetModelTaskDefPtr();
  for (size_t task_index = 0; task_index < task_list_.size(); ++task_index) {
    auto &task_def = model_task_def->task(task_index);
    auto &task = task_list_.at(task_index);
    GE_CHECK_NOTNULL(task);
    GE_CHK_STATUS_RET(task->Distribute(), "[Call][Distribute] for Task[%zu] fail", task_index);
    // for data dump
    auto op_index = std::max(task_def.kernel().context().op_index(),
                             task_def.kernel_ex().op_index());
    OpDescPtr op = GetOpByIndex(op_index);
    GE_CHECK_NOTNULL(op);
    if (reinterpret_cast<void *>(task->GetDumpArgs()) != nullptr) {
      bool call_dump = OpNeedDump(op->GetName()) && task->CallSaveDumpInfo();
      if (call_dump || is_op_debug_reg_) {
        SaveDumpTask(task->GetTaskID(), task->GetStreamId(), op, task->GetDumpArgs());
      }
    }

    auto task_type = static_cast<rtModelTaskType_t>(task_def.type());
    bool no_need_profiling = (task_type != RT_MODEL_TASK_KERNEL) && (task_type != RT_MODEL_TASK_KERNEL_EX);
    GE_IF_BOOL_EXEC(no_need_profiling, continue);

    SaveDumpOpInfo(runtime_param_, op, task->GetTaskID(), task->GetStreamId());

    // save task info for profiling
    SaveProfilingTaskDescInfo(op, task, task_def, task_index);
  }
  // launch dump kernel to aicpu
  GE_CHK_STATUS_RET(data_dumper_.LoadDumpInfo(), "[Load][DumpInfo] failed, model_id:%u.", model_id_);
  return SUCCESS;
}

bool DavinciModel::ModelNeedDump() {
  auto all_dump_model = GetDumpProperties().GetAllDumpModel();
  bool ret = all_dump_model.find(ge::DUMP_ALL_MODEL) != all_dump_model.end() ||
             all_dump_model.find(dump_model_name_) != all_dump_model.end() ||
             all_dump_model.find(om_name_) != all_dump_model.end();
  return ret;
}

void DavinciModel::SetEndGraphId(uint32_t task_id, uint32_t stream_id) {
  if (ModelNeedDump()) {
    GELOGI("start save end_graph_info to dumper, task_id is %u, stream_id is %u", task_id, stream_id);
    data_dumper_.SaveEndGraphId(task_id, stream_id);
  }
}

///
/// @ingroup ge
/// @brief Set copy only for No task feed NetOutput address.
/// @return None.
///
void DavinciModel::SetCopyOnlyOutput() {
  for (const auto &output_outside_addrs : output_data_info_) {
    ZeroCopyOffset output_outside = output_outside_addrs.second;
    if (!output_outside.IsRelativeOffsetValid()) {
      return;
    }
    for (uint32_t out_count = 0; out_count < output_outside.GetAddrCount(); ++out_count) {
      auto &addrs_mapping_list = output_outside.GetOutsideAddrs();
      std::map<const void *, std::vector<void *>> virtual_args_addrs = addrs_mapping_list[out_count];
      for (const auto &virtual_args_addr : virtual_args_addrs) {
        const auto &args_addrs = virtual_args_addr.second;
        if (args_addrs.empty()) {  // No task feed Output addr, Need copy directly.
          GELOGI("[ZCPY] just copy %p to netoutput.", virtual_args_addr.first);
          copy_only_addrs_.insert(virtual_args_addr.first);
        }
      }
    }
  }
}

///
/// @ingroup ge
/// @brief Set disabled input zero copy addr.
/// @param [in] const void *addr: address of task
/// @return None.
///
void DavinciModel::DisableZeroCopy(const void *addr) {
  if (real_virtual_addrs_.find(addr) == real_virtual_addrs_.end()) {
    return;
  }

  // Data link to RTS Op directly.
  std::lock_guard<std::mutex> lock(outside_addrs_mutex_);
  GELOGI("[ZCPY] disable zero copy of %p.", addr);
  copy_only_addrs_.insert(addr);
}

///
/// @ingroup ge
/// @brief Save outside address used info for ZeroCopy.
/// @param [in] const OpDescPtr &op_desc: current op desc
/// @param [in] const std::vector<void *> &outside_addrs: address of task
/// @param [in] const void *info: task args
/// @param [in] const char *args: task args
/// @param [in] size_t size: size of task args
/// @param [in] size_t offset: offset of task args
/// @return None.
///
void DavinciModel::SetZeroCopyAddr(const OpDescPtr &op_desc, const std::vector<void *> &outside_addrs, const void *info,
                                   void *args, size_t size, size_t offset) {
  // Internal call has ensured that op_desc is not nullptr
  GELOGD("[ZCPY] SetZeroCopyAddr for %s.", op_desc->GetName().c_str());
  size_t nums = outside_addrs.size();
  ZeroCopyTask zero_copy_task(op_desc->GetName(), static_cast<uint8_t *>(args), size);
  for (size_t i = 0; i < nums; ++i) {
    std::lock_guard<std::mutex> lock(outside_addrs_mutex_);

    for (auto &input_outside_addrs : input_data_info_) {
      ZeroCopyOffset &input_outside = input_outside_addrs.second;
      input_outside.SetOutsideAddrsValue(zero_copy_task, outside_addrs[i], args, offset + i * kAddrLen);
    }

    for (auto &output_outside_addrs : output_data_info_) {
      ZeroCopyOffset &output_outside = output_outside_addrs.second;
      output_outside.SetOutsideAddrsValue(zero_copy_task, outside_addrs[i], args, offset + i * kAddrLen);
    }
  }

  string batch_label;
  if (!AttrUtils::GetStr(op_desc, ATTR_NAME_BATCH_LABEL, batch_label) || batch_label.empty()) {
    zero_copy_task.SetBatchLabel(kDefaultBatchLable);
  } else {
    zero_copy_task.SetBatchLabel(batch_label);
  }

  std::lock_guard<std::mutex> lock(outside_addrs_mutex_);
  if (zero_copy_task.IsTaskArgsSet()) {
    zero_copy_task.SetOriginalArgs(info, offset + nums * kAddrLen);
    zero_copy_tasks_.emplace_back(zero_copy_task);
  }
}

///
/// @ingroup ge
/// @brief Copy Check input size and model op size.
/// @param [in] const int64_t &input_size: input size.
/// @param [in] const int64_t &op_size: model op size.
/// @param [in] is_dynamic: dynamic batch input flag.
/// @return true if success
///
bool DavinciModel::CheckUserAndModelSize(const int64_t &size, const int64_t &op_size,
                                          bool is_input, bool is_dynamic) {
  const std::string input_or_output = is_input ? "input" : "output";
  if (is_dynamic) {  // dynamic is max size.
    GELOGI("No need to check user %s and model size.", input_or_output.c_str());
    return true;
  }

  if (size > op_size) {
    GELOGW(
        "User %s size [%ld] is bigger than om size need [%ld], "
        "MAY cause inference result ERROR, please check model input",
        input_or_output.c_str(), size, op_size);
  }

  if (is_dynamic_aipp_) {
    GELOGI("This is dynamic aipp model, no need to judge smaller user size");
    return true;
  }
  // Judge overflow first
  if (size > (INT64_MAX - kDataMemAlignSizeCompare)) {
    GELOGI("The user %s size [%ld] is smaller than model size [%ld] and is in the range of 64 bytes",
           input_or_output.c_str(), size, op_size);
    return true;
  }
  // The input and model input size can not be exactly equal because user input is not definite.
  if ((size + kDataMemAlignSizeCompare) < op_size) {
    REPORT_INNER_ERROR("E19999", "%s size:%ld from user add align:%u < op_size:%ld in model, model_id:%u, "
                       "check invalid",
                       input_or_output.c_str(), size, kDataMemAlignSizeCompare, op_size, model_id_);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID,
           "[Check][Param] %s size:%ld from user add align:%u < op_size:%ld in model, model_id:%u",
           input_or_output.c_str(), size, kDataMemAlignSizeCompare, op_size, model_id_);
    return false;
  }
  return true;
}

///
/// @ingroup ge
/// @brief Copy Inputs and Outputs addr to model for direct use.
/// @param [in] const InputData &input_data: model input data.
/// @param [in] OutputData &output_data: model output data.
/// @param [in] bool is_dynamic_input: whether is dynamic input, true: is dynamic input; false: not is dynamic input
/// @return SUCCESS handle successfully / PARAM_INVALID for failed
///
Status DavinciModel::CopyModelData(const InputData &input_data, OutputData &output_data, bool is_dynamic) {
  if (UpdateIoTaskArgs(input_data_info_, true, input_data.blobs, is_dynamic, input_data.batch_label) != SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Call][UpdateIoTaskArgs] [ZCPY] Update input data to model:%u failed.",
           model_id_);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  if (UpdateIoTaskArgs(output_data_info_, false, output_data.blobs, is_dynamic, input_data.batch_label) !=
      SUCCESS) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Call][UpdateIoTaskArgs] [ZCPY] Update output data to model:%u failed.",
           model_id_);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  for (ZeroCopyTask &task : zero_copy_tasks_) {
    GE_CHK_STATUS_RET(task.DistributeParam(is_async_mode_, rt_model_stream_),
                      "[Call][DistributeParam] [ZCPY] Update args failed, model_id:%u.", model_id_);
  }

  output_data.index = input_data.index;
  output_data.model_id = model_id_;
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Copy Data addr to model for direct use.
/// @param [in] data_info: model memory addr/size map { data_index, { tensor_size, tensor_addr } }.
/// @param [in] is_input: input data or output data
/// @param [in] blobs: user input/output data list.
/// @param [in] is_dynamic: whether is dynamic input, true: is dynamic input; false: not is dynamic input
/// @param [in] batch_label: batch label for multi-batch scenes
/// @return SUCCESS handle successfully / others handle failed
///
Status DavinciModel::UpdateIoTaskArgs(const std::map<uint32_t, ZeroCopyOffset> &data_info, bool is_input,
                                      const vector<DataBuffer> &blobs, bool is_dynamic, const string &batch_label) {
  if (blobs.size() != data_info.size()) {
    REPORT_INNER_ERROR("E19999", "is_input:%d blob size:%ld from user != op_size:%ld in model, mode_id:%u"
                       "check invalid", is_input, blobs.size(), data_info.size(), model_id_);
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] is_input:%d blob size:%ld "
           "from user != op_size:%ld in model, mode_id:%u",
           is_input, blobs.size(), data_info.size(), model_id_);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  for (const auto &data : data_info) {
    if (data.first >= blobs.size()) {  // check data index.
      REPORT_INNER_ERROR("E19999", "is_input:%d, data index:%u from model >= blobs.size:%zu from user, mode_id:%u"
                         "check invalid", is_input, data.first, blobs.size(), model_id_);
      GELOGE(ACL_ERROR_GE_PARAM_INVALID,
             "[Check][Param] is_input:%d, data index:%u from model >= blobs.size:%zu from user, mode_id:%u",
             is_input, data.first, blobs.size(), model_id_);
      return ACL_ERROR_GE_PARAM_INVALID;
    }

    const DataBuffer &buffer = blobs[data.first];  // index of data.
    if (buffer.data == nullptr) {
      REPORT_INNER_ERROR("E19999", "is_input:%d buffer from user is nullptr, index:%u, mode_id:%u"
                         "check invalid", is_input, data.first, model_id_);
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] data_buf.data is nullptr, "
             "index=%u, mode_id:%u", data.first, model_id_);
      return ACL_ERROR_GE_PARAM_INVALID;
    }

    if (!CheckUserAndModelSize(buffer.length, data.second.GetDataSize(), is_input, is_dynamic)) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Call][CheckInputAndModelSize] failed, op[%s]",
             data.second.GetOpName().c_str());
      return ACL_ERROR_GE_PARAM_INVALID;
    }

    void *basic_addr = data.second.GetBasicAddr();
    uint64_t data_size = data.second.GetDataSize();
    if (copy_only_addrs_.count(basic_addr) > 0) {
      if (is_input && buffer.length > 0) {
        GELOGI("[IMAS] Find addr %p need direct copy from user malloc input %p", basic_addr, buffer.data);
        rtError_t rt_ret = rtMemcpy(basic_addr, data_size, buffer.data, buffer.length, RT_MEMCPY_DEVICE_TO_DEVICE);
        if (rt_ret != RT_ERROR_NONE) {
          REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%lu, model_id:%u", data_size, model_id_);
          GELOGE(rt_ret, "[Call][RtMemcpy] failed, size:%lu, model_id:%u", data_size, model_id_);
          return RT_ERROR_TO_GE_STATUS(rt_ret);
        }
      }
      GELOGI("No need to exeucte zero copy task because this addr %p need direct copy.", basic_addr);
      continue;
    }

    for (size_t count = 0; count < data.second.GetDataCount(); ++count) {
      void *addr = data.second.GetDataInfo().at(count).second;
      void *buffer_addr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(buffer.data) +
                                                   data.second.GetRelativeOffset().at(count));
      GELOGI("[ZCPY] Copy %s blobs_index %u, virtual_addr: %p, size: %ld, user_data_addr: %p, batch_label: %s",
             is_input ? "input" : "output", data.first, addr, data.second.GetDataInfo().at(count).first,
             buffer_addr, batch_label.c_str());
      // For input data, just copy for rts task.
      for (auto &task : zero_copy_tasks_) {
        bool not_same_batch = (task.GetBatchLabel() != kDefaultBatchLable && task.GetBatchLabel() != batch_label);
        if (not_same_batch) {
          continue;
        }
        uintptr_t addr_val = reinterpret_cast<uintptr_t>(addr);
        (void)task.UpdateTaskParam(addr_val, buffer_addr);
      }
    }
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief get unique identification for op when load two or more models
/// @param [in] const OpDescPtr: current op.
/// @param [in] string identification: unique identification for current op.
/// @return SUCCESS handle successfully / others handle failed
///
void DavinciModel::GetUniqueId(const OpDescPtr &op_desc, std::string &unique_identification) {
  std::string session_graph_id;
  GE_IF_BOOL_EXEC(AttrUtils::GetStr(*op_desc, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id),
                  GELOGD("Get original type of session_graph_id."));
  if (session_graph_id.empty()) {
    return;
  } else if (session_graph_id.find("-1") != string::npos) {
    unique_identification = session_graph_id + "_" + to_string(model_id_);
  } else {
    unique_identification = session_graph_id;
  }
}

///
/// @ingroup ge
/// @brief For TVM Op, avoid Addr Reuse.
/// @return void*
///
const char *DavinciModel::GetRegisterStub(const string &binfile, const string &session_graph_id) {
  string binfile_key;
  if (session_graph_id.empty()) {
    binfile_key = binfile;
  } else {
    binfile_key = session_graph_id + "_" + binfile;
  }
  auto it = tvm_bin_kernel_.find(binfile_key);
  if (it != tvm_bin_kernel_.end()) {
    return it->c_str();
  } else {
    it = tvm_bin_kernel_.insert(tvm_bin_kernel_.end(), binfile_key);
    return it->c_str();
  }
}

///
/// @ingroup ge
/// @brief Constant Op Init.
/// @return Status
///
Status DavinciModel::InitConstant(const OpDescPtr &op_desc) {
  auto v_weights = ModelUtils::GetWeights(op_desc);
  auto v_output_size = ModelUtils::GetOutputSize(op_desc);
  auto v_output_addr = ModelUtils::GetOutputDataAddrs(runtime_param_, op_desc);
  GE_IF_BOOL_EXEC(v_weights.empty() || v_output_size.empty() || v_output_addr.empty(),
                  REPORT_INNER_ERROR("E19999", "weight.size:%zu output_length.size:%zu output_addr.size:%zu in "
                                     "op:%s(%s) has empty, model_id:%u, check invalid",
                                     v_weights.size(),v_output_size.size(), v_output_addr.size(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str() ,model_id_);
                  GELOGE(PARAM_INVALID, "const op:%s not set output", op_desc->GetName().c_str());
                  return PARAM_INVALID;);

  GeTensor *tensor = const_cast<GeTensor *>(v_weights[0].get());
  GE_IF_BOOL_EXEC(static_cast<size_t>(v_output_size[0]) < tensor->GetData().size(),
                  REPORT_INNER_ERROR("E19999", "Output size:%zu < weight size:%zu in op:%s(%s) model_id:%u, "
                                     "check invalid", v_output_size[0], tensor->GetData().size(),
                                     op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
                  GELOGE(PARAM_INVALID, "[Check][Param] Output size:%zu < weight size:%zu in op:%s(%s), model_id:%u",
                         v_output_size[0], tensor->GetData().size(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
                  return PARAM_INVALID;);

  GE_IF_BOOL_EXEC(tensor->GetData().size() == 0, GELOGW("const op:%s has no weight data.", op_desc->GetName().c_str());
                  return SUCCESS;);

  auto desc = tensor->GetTensorDesc();
  if (desc.GetDataType() == DT_STRING) {
    GeShape tensor_shape = desc.GetShape();
    /// if tensor is a scaler, it's shape size if zero, according ge_tensor.cc.
    /// the logic of GetShapeSize is wrong, the scaler tensor's GetShapeSize is zero
    /// and that of unknown shape is zero too.
    /// unknown shape will not appear here, so we can use zero judge a tensor is scaler or not
    int64_t elem_num = tensor_shape.GetShapeSize();
    if (elem_num == 0 && tensor_shape.GetDims().size() == 0) {
      elem_num = 1;
    }
    uint64_t *buff = reinterpret_cast<uint64_t *>(tensor->MutableData().data());
    GE_CHECK_NOTNULL(buff);
    if (ge::CheckInt64Uint32MulOverflow(elem_num, kBytes * kStringHeadElems) != SUCCESS) {
      GELOGE(FAILED, "[Call][CheckInt64Uint32MulOverflow] Shape size:%ld is invalid", elem_num);
      return FAILED;
    }
    uint64_t offset = elem_num * kBytes * kStringHeadElems;

    uint64_t hbm_raw_data_base_addr =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(v_output_addr[0])) + offset;
    for (int64_t i = elem_num - 1; i >= 0; --i) {
      buff[i * kStringHeadElems] = hbm_raw_data_base_addr + (buff[i * kStringHeadElems] - buff[0]);
    }
  }
  GELOGI("[IMAS]InitConstant memcpy graph_%u type[V] name[%s] output[%d] memaddr[%p] mem_size[%lu] datasize[%zu]",
         runtime_param_.graph_id, op_desc->GetName().c_str(), 0, v_output_addr[0], v_output_size[0],
         tensor->GetData().size());
  GE_CHK_RT_RET(rtMemcpy(v_output_addr[0], v_output_size[0], tensor->GetData().data(), tensor->GetData().size(),
                         RT_MEMCPY_HOST_TO_DEVICE));

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief TVM Op Init.
/// @return Status
///
Status DavinciModel::InitTbeHandle(const OpDescPtr &op_desc) {
  string bin_file = op_desc->GetName();
  auto kernel = ge_model_->GetTBEKernelStore().FindKernel(op_desc->GetName());
  auto tbe_kernel = (kernel != nullptr) ? kernel : op_desc->TryGetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
  if (tbe_kernel == nullptr) {
    REPORT_INNER_ERROR("E19999", "Get tbe_kernel for op:%s(%s) fail, model_id:%u",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] TBE: %s can't find tvm bin file!", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  GE_CHK_STATUS_RET(FunctionRegister(op_desc, bin_file, tbe_kernel, false), "Function register of bin file: %s failed",
                    bin_file.c_str());
  return SUCCESS;
}

Status DavinciModel::InitTbeHandleWithFfts(const OpDescPtr &op_desc) {
  std::vector<OpKernelBinPtr> tbe_kernel;
  tbe_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, tbe_kernel);
  GELOGD("Kernel bin ptr vec size is %zu.", tbe_kernel.size());
  if (tbe_kernel.size() != kFftsTbeHandleElementSize) {
    REPORT_INNER_ERROR("E19999", "Get tbe_kernel for op:%s(%s) fail, model_id:%u",
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] TBE: %s can't find tvm bin file, size is %zu when ffts",
           op_desc->GetName().c_str(), tbe_kernel.size());
    return INTERNAL_ERROR;
  }
  if (tbe_kernel[0] == nullptr || tbe_kernel[1] == nullptr) {
    REPORT_INNER_ERROR("E19999", "Tbe kernel for op:%s is nullptr.", op_desc->GetName().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] TBE: tvm bin file of %s is nullptr when ffts.", op_desc->GetName().c_str());
    return INTERNAL_ERROR;
  }
  vector<string> bin_file_keys;
  (void)AttrUtils::GetListStr(op_desc, kStubFuncName, bin_file_keys);
  if (bin_file_keys.size() != kFftsTbeHandleElementSize) {
    REPORT_INNER_ERROR("E19999", "Get bin_file for op:%s(%s) fail.", op_desc->GetName().c_str(),
                       op_desc->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] TBE: %s can't find bin file keys, size is %zu when ffts",
           op_desc->GetName().c_str(), bin_file_keys.size());
    return INTERNAL_ERROR;
  }
  GE_CHK_STATUS_RET(FunctionRegister(op_desc, bin_file_keys[kNonTailBlock], tbe_kernel[kNonTailBlock], true,
                                     kNonTailBlock),
                    "Function register of first bin file %s failed.", bin_file_keys[kNonTailBlock].c_str());
  GE_CHK_STATUS_RET(FunctionRegister(op_desc, bin_file_keys[kTailBlock], tbe_kernel[kTailBlock], true, kTailBlock),
                    "Function register of second bin file %s failed.", bin_file_keys[kTailBlock].c_str());
  return SUCCESS;
}

Status DavinciModel::FunctionRegister(const OpDescPtr &op_desc, string &bin_file, OpKernelBinPtr &tbe_kernel,
                                      bool is_ffts, size_t thread_index) {
  if (thread_index > 1) {
    GELOGE(INTERNAL_ERROR, "[Check][Param] failed. Thread index: %zu should less than 1.", thread_index);
    return INTERNAL_ERROR;
  }
  const char *bin_file_key;
  if (is_ffts) {
    bin_file_key = GetRegisterStub(bin_file, "");
    GELOGI("Node:%s inherit func name:%s directly.", op_desc->GetName().c_str(), bin_file_key);
  } else {
    std::string session_graph_model_id;
    GetUniqueId(op_desc, session_graph_model_id);
    bin_file_key = GetRegisterStub(bin_file, session_graph_model_id);  // from set, always valid.
  }

  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();
  std::lock_guard<std::mutex> lock(tvm_bin_mutex_);
  if (rtQueryFunctionRegistered(bin_file_key) != RT_ERROR_NONE) {
    void *bin_handle = nullptr;
    if (!kernel_store.FindTBEHandle(bin_file_key, bin_handle)) {
      GELOGD("TBE: can't find the kernel_name[%s] in HandleMap", bin_file_key);

      rtDevBinary_t binary;
      GE_CHK_STATUS_RET(InitBinaryMagic(op_desc, is_ffts, thread_index, binary), "Init binary magic of %s failed.",
                        op_desc->GetName().c_str());
      binary.version = 0;
      binary.data = tbe_kernel->GetBinData();
      binary.length = tbe_kernel->GetBinDataSize();
      GELOGD("TBE: binary.length: %lu", binary.length);
      GE_CHK_RT_RET(rtDevBinaryRegister(&binary, &bin_handle));

      GE_CHK_STATUS_RET(InitMetaData(op_desc, is_ffts, thread_index, bin_handle), "Init tvm meta data of %s failed.",
                        op_desc->GetName().c_str());
      kernel_store.StoreTBEHandle(bin_file_key, bin_handle, tbe_kernel);
    } else {
      GELOGI("TBE: find the kernel_name[%s] in HandleMap", bin_file_key);
      kernel_store.ReferTBEHandle(bin_file_key);
    }
    std::string kernel_name;
    GE_CHK_STATUS_RET(InitKernelName(op_desc, is_ffts, thread_index, kernel_name), "Init kernel name of %s failed.",
                      op_desc->GetName().c_str());
    GE_CHK_RT_RET(rtFunctionRegister(bin_handle, bin_file_key, bin_file_key, kernel_name.c_str(), 0));
    used_tbe_handle_map_[bin_file_key] = 1;  // Init used num to 1.
    return SUCCESS;
  }
  // Kernel registed, Increase used num in store.
  StoreTbeHandle(bin_file_key);
  return SUCCESS;
}

Status DavinciModel::InitBinaryMagic(const OpDescPtr &op_desc, bool is_ffts, size_t thread_index,
                                     rtDevBinary_t &binary) {
  string json_string;
  const string &tvm_magic = is_ffts ? TVM_ATTR_NAME_THREAD_MAGIC : TVM_ATTR_NAME_MAGIC;
  const static std::map<std::string, uint32_t> binary_magics = {
    {"RT_DEV_BINARY_MAGIC_ELF_AICPU", RT_DEV_BINARY_MAGIC_ELF_AICPU},
    {"RT_DEV_BINARY_MAGIC_ELF", RT_DEV_BINARY_MAGIC_ELF},
    {"RT_DEV_BINARY_MAGIC_ELF_AIVEC", RT_DEV_BINARY_MAGIC_ELF_AIVEC},
    {"RT_DEV_BINARY_MAGIC_ELF_AICUBE", RT_DEV_BINARY_MAGIC_ELF_AICUBE}
  };
  if (is_ffts) {
    vector<string> json_list;
    (void)AttrUtils::GetListStr(op_desc, tvm_magic, json_list);
    if (json_list.size() != kFftsTbeHandleElementSize) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] failed. Attr is %s, thread index is %zu, json list size is %zu.",
             tvm_magic.c_str(), thread_index, json_list.size());
      return INTERNAL_ERROR;
    }
    json_string = json_list[thread_index];
  } else {
    (void)AttrUtils::GetStr(op_desc, tvm_magic, json_string);
  }
  auto iter = binary_magics.find(json_string);
  if (iter == binary_magics.end()) {
    REPORT_INNER_ERROR("E19999", "Attr:%s value:%s in op:%s(%s), model_id:%u, check invalid",
                       tvm_magic.c_str(), json_string.c_str(), op_desc->GetName().c_str(),
                       op_desc->GetType().c_str(), model_id_);
    GELOGE(PARAM_INVALID, "[Check][Param] Attr:%s value:%s in op:%s(%s), model_id:%u, check invalid",
           TVM_ATTR_NAME_MAGIC.c_str(), json_string.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return PARAM_INVALID;
  }
  binary.magic = iter->second;
  return SUCCESS;
}

Status DavinciModel::InitMetaData(const OpDescPtr &op_desc, bool is_ffts, size_t thread_index, void *bin_handle) {
  string meta_data;
  const string &tvm_metadata = is_ffts ? TVM_ATTR_NAME_THREAD_METADATA : TVM_ATTR_NAME_METADATA;
  if (is_ffts) {
    vector<string> meta_data_list;
    (void)AttrUtils::GetListStr(op_desc, tvm_metadata, meta_data_list);
    if (meta_data_list.size() != kFftsTbeHandleElementSize) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] failed, attr is %s, thread index is %zu, meta data list size is %zu.",
             tvm_metadata.c_str(), thread_index, meta_data_list.size());
      return INTERNAL_ERROR;
    }
    meta_data = meta_data_list[thread_index];
  } else {
    (void)AttrUtils::GetStr(op_desc, tvm_metadata, meta_data);
  }
  GELOGD("TBE: meta data: %s", meta_data.empty() ? "null" : meta_data.c_str());
  if (!meta_data.empty()) {
    GE_CHK_RT_RET(rtMetadataRegister(bin_handle, meta_data.c_str()));
  }
  return SUCCESS;
}

Status DavinciModel::InitKernelName(const OpDescPtr &op_desc, bool is_ffts, size_t thread_index, string &kernel_name) {
  if (is_ffts) {
    // delete prefix, eg: *sgt_graph_nodes*/loss_scale/gradient/fp32_vals/Mean_grad/Tile
    vector<string> kernel_name_list;
    auto pos = op_desc->GetName().find("/");
    if (pos == std::string::npos) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] failed, subgraph node name: %s.", op_desc->GetName().c_str());
      return INTERNAL_ERROR;
    }
    string attr_kernel_name = op_desc->GetName().substr(pos + 1) + "_thread_kernelname";
    (void)AttrUtils::GetListStr(op_desc, attr_kernel_name, kernel_name_list);
    if (kernel_name_list.size() != kFftsTbeHandleElementSize) {
      GELOGE(INTERNAL_ERROR, "[Check][Param] failed, attr is %s, thread index is %zu, kernel name list size is %zu.",
             attr_kernel_name.c_str(), thread_index, kernel_name_list.size());
      return INTERNAL_ERROR;
    }
    kernel_name = kernel_name_list[thread_index];
  } else {
    string attr_kernel_name = op_desc->GetName() + "_kernelname";
    (void)AttrUtils::GetStr(op_desc, attr_kernel_name, kernel_name);
  }
  return SUCCESS;
}

void DavinciModel::StoreTbeHandle(const std::string &handle_key) {
  // Online mode FE may call rtFunctionRegister.
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();

  auto it = used_tbe_handle_map_.find(handle_key);
  if (it != used_tbe_handle_map_.end()) {
    // GE registered, increase reference.
    kernel_store.ReferTBEHandle(handle_key);
    it->second++;
    return;
  }

  void *bin_handle = nullptr;
  if (kernel_store.FindTBEHandle(handle_key, bin_handle)) {
    // GE registered, increase reference.
    used_tbe_handle_map_[handle_key] = 1;  // Init used num to 1.
    kernel_store.ReferTBEHandle(handle_key);
  }
}

void DavinciModel::CleanTbeHandle() {
  TBEHandleStore &kernel_store = TBEHandleStore::GetInstance();

  kernel_store.EraseTBEHandle(used_tbe_handle_map_);
  used_tbe_handle_map_.clear();
  tvm_bin_kernel_.clear();
}

///
/// @ingroup ge
/// @brief insert active_stream_indication_
/// @return Status
///
Status DavinciModel::InitStreamActive(const OpDescPtr &op_desc) {
  if (op_desc->HasAttr(ATTR_NAME_SWITCH_BRANCH_NODE_LABEL)) {
    std::vector<uint32_t> active_stream_list;
    GE_CHK_BOOL_EXEC(AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list),
                     REPORT_INNER_ERROR("E19999", "[Get][Attr] %s in op:%s(%s) failed, model_id:%u.",
                                        ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                                        op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
                     return INTERNAL_ERROR,
                    "[Get][Attr] %s in op:%s(%s) failed, model_id:%u.", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                    op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);

    for (size_t j = 0; j < active_stream_list.size(); ++j) {
      active_stream_indication_.insert(active_stream_list[j]);
      GELOGI("flowctrl_op_index_map  node:%s, active_stream_id=%u.", op_desc->GetName().c_str(), active_stream_list[j]);
    }
  }

  return SUCCESS;
}

Status DavinciModel::InitStreamSwitch(const OpDescPtr &op_desc) {
  std::vector<uint32_t> active_stream_list;
  GE_LOGI_IF(!ge::AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list),
             "GetInt ACTIVE_STREAM_LIST failed.");
  if (active_stream_list.size() != kTrueBranchStreamNum) {
    REPORT_INNER_ERROR("E19999", "Attr:%s active_stream_list.size:%zu in op:%s(%s) != kTrueBranchStreamNum:%u, "
                       "model_id:%u, check invalid",
                       ATTR_NAME_ACTIVE_STREAM_LIST.c_str(), active_stream_list.size(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(),
                       kTrueBranchStreamNum, model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Attr:%s active_stream_list.size:%zu in op:%s(%s) != %u, model_id:%u",
           ATTR_NAME_ACTIVE_STREAM_LIST.c_str(), active_stream_list.size(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), kTrueBranchStreamNum, model_id_);
    return INTERNAL_ERROR;
  }

  uint32_t true_stream_id = active_stream_list.front();
  active_stream_indication_.insert(true_stream_id);
  GELOGI("flowctrl_op_index_map  node:%s, true_stream_id=%u.", op_desc->GetName().c_str(), true_stream_id);

  return SUCCESS;
}

Status DavinciModel::InitStreamSwitchN(const OpDescPtr &op_desc) {
  std::vector<uint32_t> active_stream_list;
  if (!AttrUtils::GetListInt(op_desc, ATTR_NAME_ACTIVE_STREAM_LIST, active_stream_list)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s from op:%s(%s) fail, model_id:%u", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s from op:%s(%s) fail, model_id:%u", ATTR_NAME_ACTIVE_STREAM_LIST.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return INTERNAL_ERROR;
  }

  for (size_t j = 0; j < active_stream_list.size(); ++j) {
    active_stream_indication_.insert(active_stream_list[j]);
    GELOGI("StreamSwitchNOp node:%s, active_stream_id=%u.", op_desc->GetName().c_str(), active_stream_list[j]);
  }

  uint32_t batch_num = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s from op:%s(%s) fail, model_id:%u", ATTR_NAME_BATCH_NUM.c_str(),
                       op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) fail, model_id:%u", ATTR_NAME_BATCH_NUM.c_str(),
           op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
    return FAILED;
  }

  return SetDynamicBatchInfo(op_desc, batch_num);
}

Status DavinciModel::SetDynamicBatchInfo(const OpDescPtr &op_desc, uint32_t batch_num) {
  batch_info_.clear();
  combined_batch_info_.clear();

  (void)AttrUtils::GetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type_);
  (void)AttrUtils::GetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order_);
  for (uint32_t i = 0; i < batch_num; ++i) {
    std::vector<int64_t> batch_shape;
    const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
    if (!AttrUtils::GetListInt(op_desc, attr_name, batch_shape)) {
      REPORT_INNER_ERROR("E19999", "Get Attr:%s from op:%s(%s) fail, model_id:%u", attr_name.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
      GELOGE(FAILED, "[Get][Attr] %s from op:%s(%s) fail, model_id:%u", attr_name.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), model_id_);
      batch_info_.clear();
      return FAILED;
    }
    batch_info_.emplace_back(batch_shape);
    batch_shape.clear();
    const string attr_combined_batch = ATTR_NAME_COMBINED_BATCH + "_" + std::to_string(i);
    if (AttrUtils::GetListInt(op_desc, attr_combined_batch, batch_shape)) {
      combined_batch_info_.emplace_back(batch_shape);
    }
  }

  return SUCCESS;
}

Status DavinciModel::InitCase(const OpDescPtr &op_desc) {
  uint32_t batch_num = 0;
  if (!AttrUtils::GetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num)) {
    GELOGI("Not multi-batch Node: %s", op_desc->GetName().c_str());
    return SUCCESS;
  }

  return SetDynamicBatchInfo(op_desc, batch_num);
}

bool DavinciModel::IsBroadCastOpData(const ge::NodePtr &var_node) {
  for (auto out_anchor : var_node->GetAllOutDataAnchors()) {
    GE_RT_FALSE_CHECK_NOTNULL(out_anchor);
    for (auto in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_RT_FALSE_CHECK_NOTNULL(in_anchor);
      ge::NodePtr dst_node = in_anchor->GetOwnerNode();
      GE_RT_FALSE_CHECK_NOTNULL(dst_node);
      if (dst_node->GetType() == HCOMBROADCAST || dst_node->GetType() == HVDCALLBACKBROADCAST) {
        return true;
      }
    }
  }
  return false;
}

///
/// @ingroup ge
/// @brief Init model stream for NN model.
/// @param [in] stream   user input model stream.
/// @return Status
///
Status DavinciModel::InitModelStream(rtStream_t stream) {
  ExecuteMode curr_mode = is_async_mode_ ? ASYNCHRONIZATION : SYNCHRONIZATION;
  GE_CHK_BOOL_RET_STATUS((curr_mode == last_execute_mode_) || (last_execute_mode_ == INITIALIZATION), INTERNAL_ERROR,
                         "[Check][Param] NnExecute not support mix execute.");
  last_execute_mode_ = curr_mode;

  // asynchronize mode, use user input stream.
  if (is_async_mode_) {
    rt_model_stream_ = stream;
    is_inner_model_stream_ = false;
    return SUCCESS;
  }

  // synchronize mode, use forbidden stream.
  if (stream != nullptr) {
    if ((rt_model_stream_ != nullptr) && is_inner_model_stream_) {
      GE_LOGW_IF(rtStreamDestroy(rt_model_stream_) != RT_ERROR_NONE, "Destroy rt_stream failed!");
    }

    rt_model_stream_ = stream;
    is_inner_model_stream_ = false;
    return SUCCESS;
  }

  if (rt_model_stream_ == nullptr) {
    GE_CHK_RT_RET(rtStreamCreateWithFlags(&rt_model_stream_, priority_, RT_STREAM_FORBIDDEN_DEFAULT));
    is_inner_model_stream_ = true;
  }

  return SUCCESS;
}

///
/// @ingroup ge
/// @brief ACL case, do not start  new thread, return execute result.
/// @param [in] stream   execute model stream.
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  model input data.
/// @param [out] output_data  model output data.
///
Status DavinciModel::NnExecute(rtStream_t stream, bool async_mode, const InputData &input_data,
                               OutputData &output_data) {
  is_async_mode_ = async_mode;
  GELOGD("Model Run begin, model id:%u, data index:%u, flag:%d.", model_id_, input_data.index, is_async_mode_);
  GE_CHK_STATUS_RET(InitModelStream(stream), "[Init][ModelStream] failed, model_id:%u.", model_id_);
  is_dynamic_ = input_data.is_dynamic_batch;

  bool profiling_model_execute_on = ProfilingManager::Instance().ProfilingModelExecuteOn();
  GE_IF_BOOL_EXEC(profiling_model_execute_on, SetProfileTime(MODEL_PRE_PROC_START));
  Status ret = CopyModelData(input_data, output_data, is_dynamic_);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret,
                                 "[Copy][ModelData] failed. model id: %u", model_id_);

  GELOGD("current_data.index=%u", input_data.index);
  GE_IF_BOOL_EXEC(profiling_model_execute_on, SetProfileTime(MODEL_PRE_PROC_END));

  if (!task_list_.empty()) {
    uint64_t index_id = iterator_count_ + 1;
    uint64_t model_id = static_cast<uint64_t>(model_id_);
    int32_t device_id = static_cast<int32_t>(device_id_);
    // tag_id 0 means step begin, 1 meas step end.
    GE_CHK_STATUS_RET_NOLOG(
      ProfilingManager::Instance().ProfileStepInfo(index_id, model_id, 0, rt_model_stream_, device_id));

    GELOGD("rtModelExecute do");
    GE_IF_BOOL_EXEC(profiling_model_execute_on, SetProfileTime(MODEL_INFER_START));
    rtError_t rt_ret = rtModelExecute(rt_model_handle_, rt_model_stream_, 0);
    GE_CHK_RT_EXEC(rt_ret, return RT_ERROR_TO_GE_STATUS(rt_ret));
    GE_IF_BOOL_EXEC(profiling_model_execute_on, SetProfileTime(MODEL_INFER_END));
    GELOGD("rtModelExecute end");

    GE_CHK_STATUS_RET_NOLOG(
      ProfilingManager::Instance().ProfileStepInfo(index_id, model_id, 1, rt_model_stream_, device_id));
    iterator_count_++;
  }

  GE_IF_BOOL_EXEC(profiling_model_execute_on, SetProfileTime(MODEL_AFTER_PROC_START));
  ret = CopyOutputData(input_data.index, output_data, RT_MEMCPY_DEVICE_TO_DEVICE);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ACL_ERROR_GE_INTERNAL_ERROR,
                                 "[Copy][OutputData] to user failed, ret:%d, model_id:%u.", ret, model_id_);
  GE_IF_BOOL_EXEC(profiling_model_execute_on, SetProfileTime(MODEL_AFTER_PROC_END));

  // report model time data
  GE_IF_BOOL_EXEC(profiling_model_execute_on, (void)SinkTimeProfile(input_data));
  GELOGD("Model run end, model id:%u", model_id_);
  return SUCCESS;
}

// Add active entry stream for special env.
Status DavinciModel::AddHeadStream() {
  if (active_stream_list_.empty()) {
    REPORT_INNER_ERROR("E19999", "active_stream_list is empty in model:%u, check invalid", model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][Param] active_stream_list is empty in model:%u, check invalid", model_id_);
    return INTERNAL_ERROR;
  }

  if (active_stream_list_.size() == 1) {
    GELOGI("Just one active stream, take as head stream.");
    rt_head_stream_ = active_stream_list_[0];
    is_pure_head_stream_ = false;
  } else {
    // Create stream which rt_model_handel running on, this is S0, TS stream.
    GELOGI("Multiple active stream: %zu, create head stream.", active_stream_list_.size());
    GE_CHK_RT_RET(rtStreamCreateWithFlags(&rt_head_stream_, priority_, RT_STREAM_PERSISTENT));
    GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, rt_head_stream_, RT_INVALID_FLAG));  // Not active.
    is_pure_head_stream_ = true;

    for (auto s : active_stream_list_) {
      std::shared_ptr<CpuTaskActiveEntry> active_entry = MakeShared<CpuTaskActiveEntry>(rt_head_stream_);
      if (active_entry == nullptr) {
        REPORT_CALL_ERROR("E19999", "New CpuTaskActiveEntry failed, model_id:%u", model_id_);
        GELOGE(MEMALLOC_FAILED, "[New][CpuTaskActiveEntry] task failed, model_id:%u", model_id_);
        return MEMALLOC_FAILED;
      }

      Status status = active_entry->Init(s);
      if (status != SUCCESS) {
        return status;
      }

      cpu_task_list_.emplace_back(active_entry);
    }
  }

  // Create entry stream active head stream. AICPU stream.
  GE_CHK_RT_RET(rtStreamCreateWithFlags(&rt_entry_stream_, priority_, RT_STREAM_AICPU));
  GE_CHK_RT_RET(rtModelBindStream(rt_model_handle_, rt_entry_stream_, RT_HEAD_STREAM));
  return SUCCESS;
}

Status DavinciModel::InitEntryTask() {
  if (deploy_type_ == AICPU_DEPLOY_CROSS_THREAD) {
    GE_CHK_STATUS_RET(AddHeadStream(), "[Add][HeadStream] failed.");
    return CpuActiveStream();
  } else {
    return LoadWithQueue();
  }
}

uint8_t *DavinciModel::MallocFeatureMapMem(size_t data_size) {
  uint8_t *mem_base = nullptr;
  const string purpose("feature map,used for op input and output.");
  char ge_static_mem_env[MMPA_MAX_PATH] = {0x00};
  INT32 res = mmGetEnv(kEnvGeuseStaticMemory, ge_static_mem_env, MMPA_MAX_PATH);
  if (res == EN_OK) {
    data_size = static_cast<size_t>(VarManager::Instance(session_id_)->GetGraphMemoryMaxSize());
    string memory_key = std::to_string(0) + "_f";
    mem_base =
      MemManager::Instance().MemInstance(RT_MEMORY_HBM).MallocMemory(purpose, memory_key, data_size, GetDeviceId());
  } else {
    mem_base = MemManager::Instance().MemInstance(RT_MEMORY_HBM).MallocMemory(purpose, data_size, GetDeviceId());
  }

  if (mem_base != nullptr) {
    GE_CHK_RT(rtMemset(mem_base, data_size, 0U, data_size));
  }
  return mem_base;
}

Status DavinciModel::MallocExMem() {
  char ge_static_mem_env[MMPA_MAX_PATH] = {0x00};
  INT32 res_static_memory = mmGetEnv(kEnvGeuseStaticMemory, ge_static_mem_env, MMPA_MAX_PATH);
  for (auto &it : runtime_param_.memory_infos) {
    auto mem_size = it.second.memory_size;
    if (mem_size == 0) {
      continue;
    }
    bool sessoion_scope = ((kSessionScopeMemory & it.first) == kSessionScopeMemory);
    auto mem_type = it.first & kMemoryTypeMask;
    uint8_t *mem_base = nullptr;
    const string purpose("p2p memory, used for some op related to hcom or session scope memory");
    if (sessoion_scope) {
      mem_base = MemManager::Instance().SessionScopeMemInstance(mem_type).Malloc(mem_size, runtime_param_.session_id);
    } else if (res_static_memory == EN_OK) {
      string memory_key = std::to_string(0) + it.second.memory_key;
      mem_base =
        MemManager::Instance().MemInstance(mem_type).MallocMemory(purpose, memory_key, mem_size, GetDeviceId());
    } else {
      mem_base = MemManager::Instance().MemInstance(mem_type).MallocMemory(purpose, mem_size, GetDeviceId());
    }

    if (mem_base == nullptr) {
      REPORT_CALL_ERROR("E19999", "MallocExMem fail, type:%ld size:%zu, model_id:%u, check invalid",
                        mem_type, mem_size, model_id_);
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "Alloc ex memory failed, type:%ld size: %zu", mem_type, mem_size);
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }
    it.second.memory_base = mem_base;
    GELOGI("InitFeatureMapAndP2PMem graph_%u MallocMemory type[F] mem_type[%ld] mem_addr[%p] mem_size[%zu]",
           runtime_param_.graph_id, mem_type, mem_base, mem_size);
  }
  return SUCCESS;
}

uint8_t *DavinciModel::MallocWeightsMem(size_t weights_size) {
  uint8_t *weights_mem_base = nullptr;
  const string purpose("weights memory in inference network.");
  char ge_static_mem_env[MMPA_MAX_PATH] = {0x00};
  INT32 res = mmGetEnv(kEnvGeuseStaticMemory, ge_static_mem_env, MMPA_MAX_PATH);
  if (res == EN_OK) {
    string weight_memory_key = std::to_string(0) + "_w";
    weights_mem_base = MemManager::Instance()
                         .MemInstance(RT_MEMORY_HBM)
                         .MallocMemory(purpose, weight_memory_key, weights_size, GetDeviceId());
  } else {
    weights_mem_base =
      MemManager::Instance().MemInstance(RT_MEMORY_HBM).MallocMemory(purpose, weights_size, GetDeviceId());
  }
  return weights_mem_base;
}

void DavinciModel::FreeFeatureMapMem() {
  char ge_static_mem_env[MMPA_MAX_PATH] = {0x00};
  INT32 res = mmGetEnv(kEnvGeuseStaticMemory, ge_static_mem_env, MMPA_MAX_PATH);
  if (res == EN_OK && is_inner_mem_base_) {
    string weight_memory_key = std::to_string(0) + "_f";
    if (MemManager::Instance().MemInstance(RT_MEMORY_HBM).GetMemoryAddr(weight_memory_key) != nullptr) {
      GE_CHK_STATUS(MemManager::Instance().MemInstance(RT_MEMORY_HBM).FreeMemory(weight_memory_key, GetDeviceId()),
                    "failed to free weight memory");
    }
    mem_base_ = nullptr;
  } else {
    GE_IF_BOOL_EXEC(
      mem_base_ != nullptr && is_inner_mem_base_,
      GE_CHK_STATUS(MemManager::Instance().MemInstance(RT_MEMORY_HBM).FreeMemory(mem_base_, GetDeviceId()),
                    "failed to free feature_map memory");
      mem_base_ = nullptr);
  }
}

void DavinciModel::FreeExMem() {
  char ge_static_mem_env[MMPA_MAX_PATH] = {0x00};
  INT32 res_static_memory = mmGetEnv(kEnvGeuseStaticMemory, ge_static_mem_env, MMPA_MAX_PATH);
  for (auto &it : runtime_param_.memory_infos) {
    // free when session destory
    if ((kSessionScopeMemory & it.first) == kSessionScopeMemory) {
      continue;
    }
    auto mem_type = it.first & kMemoryTypeMask;
    if (res_static_memory == EN_OK) {
      std::string memory_key = std::to_string(0) + it.second.memory_key;
      if (MemManager::Instance().MemInstance(mem_type).GetMemoryAddr(memory_key) != nullptr) {
        GE_CHK_STATUS(MemManager::Instance().MemInstance(mem_type).FreeMemory(memory_key, GetDeviceId()),
                      "failed to free memory");
      }
      it.second.memory_base = nullptr;
    } else {
      GE_IF_BOOL_EXEC(
        it.second.memory_base != nullptr,
        GE_CHK_STATUS(MemManager::Instance().MemInstance(mem_type).FreeMemory(it.second.memory_base, GetDeviceId()),
                      "failed to free memory");
        it.second.memory_base = nullptr);
    }
  }
}

void DavinciModel::FreeWeightsMem() {
  char ge_static_mem_env[MMPA_MAX_PATH] = {0x00};
  INT32 res = mmGetEnv(kEnvGeuseStaticMemory, ge_static_mem_env, MMPA_MAX_PATH);
  if (res == EN_OK) {
    string memory_key = std::to_string(0) + "_w";
    if (MemManager::Instance().MemInstance(RT_MEMORY_HBM).GetMemoryAddr(memory_key) != nullptr) {
      GE_CHK_STATUS(MemManager::Instance().MemInstance(RT_MEMORY_HBM).FreeMemory(memory_key, GetDeviceId()),
                    "failed to free feature_map memory");
    }
    weights_mem_base_ = nullptr;
  } else {
    GE_IF_BOOL_EXEC(
      weights_mem_base_ != nullptr && weights_mem_base_ != mem_base_ && is_inner_weight_base_,
      GE_CHK_STATUS(MemManager::Instance().MemInstance(RT_MEMORY_HBM).FreeMemory(weights_mem_base_, GetDeviceId()),
                    "failed to free weight memory");
      weights_mem_base_ = nullptr);
  }
}

Status DavinciModel::TransAllVarData(ComputeGraphPtr &graph, uint32_t graph_id) {
  rtContext_t ctx = nullptr;
  rtError_t rt_ret = rtCtxGetCurrent(&ctx);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCtxGetCurrent failed, model_id:%u", model_id_);
    GELOGE(RT_FAILED, "[Call][RtCtxGetCurrent] failed, ret:0x%X, model_id:%u.", rt_ret, model_id_);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  std::vector<NodePtr> variable_node_list;
  for (ge::NodePtr &node : graph->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetType() != VARIABLE) {
      continue;
    }
    variable_node_list.emplace_back(node);
  }

  GE_CHK_STATUS_RET_NOLOG(
      TransVarDataUtils::TransAllVarData(variable_node_list, session_id_, ctx, graph_id, kThreadNum));
  return SUCCESS;
}

void DavinciModel::SetDataDumperArgs(const ComputeGraphPtr &graph, const map<string, OpDescPtr> &variable_by_name) {
  if(dump_model_name_.empty()) {
    dump_model_name_ = name_;
  }
  data_dumper_.SetModelName(dump_model_name_);
  data_dumper_.SetModelId(model_id_);
  data_dumper_.SetOmName(om_name_);
  data_dumper_.SetComputeGraph(graph);
  data_dumper_.SetRefInfo(saved_task_addrs_);

  int32_t device_id = 0;
  rtError_t rt_ret = rtGetDevice(&device_id);
  if (rt_ret != RT_ERROR_NONE || device_id < 0) {
    REPORT_CALL_ERROR("E19999", "Call rtGetDevice failed, model_id:%u", model_id_);
    GELOGE(RT_FAILED, "[Call][RtGetDevice] failed, ret = 0x%X, device_id = %d.", rt_ret, device_id);
    return;
  }
  data_dumper_.SetDeviceId(device_id);

  if (known_node_) {
    data_dumper_.SetLoopAddr(global_step_addr_, nullptr, nullptr);
  } else {
    // set loop count addr
    auto get_var_addr = [&](const string &name) -> void *{
    const auto it = variable_by_name.find(name);
    if (it != variable_by_name.end()) {
      const auto output_sizes = ModelUtils::GetOutputSize(it->second);
      const auto output_addrs = ModelUtils::GetOutputDataAddrs(runtime_param_, it->second);
      if (output_sizes.empty() || output_addrs.empty()) {
        return nullptr;
      }
      return output_addrs[0];
    }
    GELOGD("op: %s is null.", name.c_str());
    return nullptr;
  };
  data_dumper_.SetLoopAddr(get_var_addr(NODE_NAME_GLOBAL_STEP),
                           get_var_addr(NODE_NAME_FLOWCTRL_LOOP_PER_ITER),
                           get_var_addr(NODE_NAME_FLOWCTRL_LOOP_COND));
  }
}

uint32_t DavinciModel::GetFlowctrlIndex(uint32_t op_index) {
  std::lock_guard<std::mutex> lock(flowctrl_op_index_internal_map_mutex_);
  return (++flowctrl_op_index_internal_map_[op_index]) - 1;
}

void DavinciModel::PushHcclStream(rtStream_t value) {
  std::lock_guard<std::mutex> lock(all_hccl_stream_list_mutex_);
  all_hccl_stream_list_.push_back(value);
}

void DavinciModel::SaveHcclFollowStream(int64_t main_stream_id, rtStream_t stream) {
  std::lock_guard<std::mutex> lock(capacity_of_stream_mutex_);
  main_follow_stream_mapping_[main_stream_id].emplace_back(stream);
}

void DavinciModel::SetTotalFixedAddrsSize(string tensor_name, int64_t fix_addr_size) {
  if (tensor_name_to_fixed_addr_size_.find(tensor_name) == tensor_name_to_fixed_addr_size_.end()) {
    tensor_name_to_fixed_addr_size_[tensor_name] = total_fixed_addr_size_;
    total_fixed_addr_size_ += fix_addr_size;
  }
}

Status DavinciModel::InitOrigInputInfo(uint32_t index, const OpDescPtr &op_desc) {
  if (!op_desc->HasAttr(ATTR_NAME_AIPP_INPUTS) || !op_desc->HasAttr(ATTR_NAME_AIPP_OUTPUTS)) {
    GELOGI("there is not AIPP related with index %u, node: %s.", index, op_desc->GetName().c_str());
    return SUCCESS;
  }

  vector<string> inputs;
  if (AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs) && !inputs.empty()) {
    std::string input = inputs[kAippOriginInputIndex];
    GELOGI("origin input str: %s.", input.c_str());
    std::vector<std::string> infos = ge::StringUtils::Split(input, ':');
    if (infos.size() != kAippInfoNum) {
      REPORT_INNER_ERROR("E19999", "Attr:%s in op:%s(%s), aipp input size:%zu != kAippInfoNum:%u, model_id:%u, "
                         "check invalid", ATTR_NAME_AIPP_INPUTS.c_str(),
                         op_desc->GetName().c_str(), op_desc->GetType().c_str(), infos.size(), kAippInfoNum,
                         model_id_);
      GELOGE(ACL_ERROR_GE_AIPP_MODE_INVALID, "[Check][Param] Attr:%s in op:%s(%s), "
             "aipp input size:%zu != kAippInfoNum:%u, model_id:%u", ATTR_NAME_AIPP_INPUTS.c_str(),
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), infos.size(), kAippInfoNum, model_id_);
      return ACL_ERROR_GE_AIPP_MODE_INVALID;
    }

    OriginInputInfo input_info;
    input_info.format = TypeUtils::SerialStringToFormat(infos[kAippInfoFormat]);
    input_info.data_type = TypeUtils::SerialStringToDataType(infos[kAippInfoDataType]);
    input_info.dim_num = std::strtol(infos[kAippInfoDimNum].c_str(), nullptr, kDecimal);
    orig_input_info_[index] = input_info;
  } else {
    OriginInputInfo input_info = { FORMAT_RESERVED, DT_UNDEFINED, 0 };
    orig_input_info_[index] = input_info;
  }

  return SUCCESS;
}

Status DavinciModel::GetOrigInputInfo(uint32_t index, OriginInputInfo &orig_input_info) const {
  const auto it = orig_input_info_.find(index);
  if (it == orig_input_info_.end()) {
    REPORT_INNER_ERROR("E19999", "Get index:%u from orig_input_info_ fail, model_id:%u", index, model_id_);
    GELOGE(ACL_ERROR_GE_AIPP_NOT_EXIST, "[Check][Param] Get index:%u from orig_input_info_ fail, model_id:%u",
           index, model_id_);
    return ACL_ERROR_GE_AIPP_NOT_EXIST;
  }

  const OriginInputInfo &input_info = it->second;
  if (input_info.format != FORMAT_RESERVED || input_info.data_type != DT_UNDEFINED) {
    orig_input_info = input_info;
  }

  return SUCCESS;
}

void DavinciModel::ParseAIPPInfo(std::string in_out_info, InputOutputDims &dims_info) {
  GELOGI("ParseAIPPInfo: origin str: %s", in_out_info.c_str());
  std::vector<std::string> infos = ge::StringUtils::Split(in_out_info, ':');
  if (infos.size() != kAippInfoNum) {
    REPORT_INNER_ERROR("E19999", "in_out_info:%s size:%zu != kAippInfoNum:%u, model_id:%u, "
                       "check invalid", in_out_info.c_str(), infos.size(), kAippInfoNum,
                       model_id_);
    GELOGE(ACL_ERROR_GE_AIPP_MODE_INVALID, "[Check][Param] in_out_info:%s size:%zu != kAippInfoNum:%u, model_id:%u",
           in_out_info.c_str(), infos.size(), kAippInfoNum, model_id_);
    return;
  }
  dims_info.name = infos[kAippInfoTensorName];
  dims_info.size = std::strtol(infos[kAippInfoTensorSize].c_str(), nullptr, kDecimal);
  dims_info.dim_num = std::strtol(infos[kAippInfoDimNum].c_str(), nullptr, kDecimal);

  std::vector<std::string> dims = ge::StringUtils::Split(infos[kAippInfoShape], ',');
  for (const auto &dim : dims) {
    if (dim.empty()) {
      continue;
    }
    dims_info.dims.emplace_back(std::strtol(dim.c_str(), nullptr, kDecimal));
  }
}

Status DavinciModel::InitAippInputOutputDims(uint32_t index, const OpDescPtr &op_desc) {
  if (!op_desc->HasAttr(ATTR_NAME_AIPP_INPUTS) || !op_desc->HasAttr(ATTR_NAME_AIPP_OUTPUTS)) {
    GELOGI("There is not AIPP related with index %u.", index);
    return SUCCESS;
  }

  vector<string> inputs;
  vector<InputOutputDims> input_dims;
  if (AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_INPUTS, inputs) && !inputs.empty()) {
    GELOGI("Data: %s has %zu related aippInfo.", op_desc->GetName().c_str(), inputs.size());
    for (auto it : inputs) {
      InputOutputDims input_info;
      ParseAIPPInfo(it, input_info);
      input_dims.emplace_back(input_info);
      GELOGD("Aipp origin input dims info: %s", it.c_str());

      ConstGeTensorDescPtr data_input_desc = op_desc->GetInputDescPtr(kDataIndex);
      int64_t data_input_size;
      (void)TensorUtils::GetSize(*(op_desc->GetInputDescPtr(kDataIndex)), data_input_size);
      GELOGD("Related Data[%d]: tensor_name: %s, dim_num: %zu, tensor_size: %zu, format: %s, data_type: %s, shape: %s.",
          index, op_desc->GetName().c_str(), data_input_desc->GetShape().GetDimNum(), data_input_size,
          TypeUtils::FormatToSerialString(data_input_desc->GetFormat()).c_str(),
          TypeUtils::DataTypeToSerialString(data_input_desc->GetDataType()).c_str(),
          formats::JoinToString(data_input_desc->GetShape().GetDims()).c_str());
    }
  }

  vector<string> outputs;
  vector<InputOutputDims> output_dims;
  if (AttrUtils::GetListStr(op_desc, ATTR_NAME_AIPP_OUTPUTS, outputs) && !outputs.empty()) {
    for (auto it : outputs) {
      InputOutputDims output_info;
      ParseAIPPInfo(it, output_info);
      output_dims.emplace_back(output_info);
      GELOGD("Aipp output dims info: %s", it.c_str());
    }
  }

  aipp_dims_info_[index] = { input_dims, input_dims };
  return SUCCESS;
}

Status DavinciModel::GetAllAippInputOutputDims(uint32_t index, vector<InputOutputDims> &input_dims,
                                               vector<InputOutputDims> &output_dims) const {
  const auto it = aipp_dims_info_.find(index);
  if (it == aipp_dims_info_.end()) {
    REPORT_INNER_ERROR("E19999", "Get index:%u from aipp_dims_info_ fail, model_id:%u", index, model_id_);
    GELOGE(ACL_ERROR_GE_AIPP_NOT_EXIST, "[Check][Param] Get index:%u from aipp_dims_info_ fail, model_id:%u",
           index, model_id_);
    return ACL_ERROR_GE_AIPP_NOT_EXIST;
  }

  input_dims = it->second.first;
  output_dims = it->second.second;
  return SUCCESS;
}

int64_t DavinciModel::GetFixedAddrsSize(string tensor_name) {
  if (tensor_name_to_fixed_addr_size_.find(tensor_name) != tensor_name_to_fixed_addr_size_.end()) {
    return tensor_name_to_fixed_addr_size_[tensor_name];
  } else {
    return total_fixed_addr_size_;
  }
}

Status DavinciModel::InitL1DataDumperArgs() {
  auto all_dump_model = GetDumpProperties().GetAllDumpModel();
  bool find_by_om_name = all_dump_model.find(om_name_) != all_dump_model.end();
  bool find_by_model_name = all_dump_model.find(dump_model_name_) != all_dump_model.end();
  bool dump_l1fusion_op =
    (all_dump_model.find(ge::DUMP_ALL_MODEL) != all_dump_model.end()) || find_by_om_name || find_by_model_name;
  if (dump_l1fusion_op) {
    // malloc 2M for dump l1fusion op
    GE_CHK_RT_RET(rtMalloc(&l1_fusion_addr_, kDumpL1FusionOpMByteSize, RT_MEMORY_DDR));

    // send l1fusion dump addr to rts
    if (rtDumpAddrSet(rt_model_handle_, l1_fusion_addr_, kDumpL1FusionOpMByteSize, kDumpFlagOfL1Fusion) !=
        RT_ERROR_NONE) {
      // l1_fusion_addr_ will be free when DavinciModel destruct
      REPORT_CALL_ERROR("E19999", "Call rtDumpAddrSet failed, model_id:%u", model_id_);
      GELOGE(FAILED, "[Call][RtDumpAddrSet] failed, model_id:%u", model_id_);
      return FAILED;
    }

    // set addr for l1 data dump
    data_dumper_.SetL1FusionAddr(l1_fusion_addr_);
  }
  return SUCCESS;
}

Status DavinciModel::SetRunAsyncListenerCallback(const RunAsyncCallback &callback) {
  auto listener = dynamic_cast<RunAsyncListener *>(listener_.get());
  GE_CHECK_NOTNULL(listener);
  listener->SetCallback(callback);
  return SUCCESS;
}

void DavinciModel::UpdateOpIOAddrs(uint32_t task_id, uint32_t stream_id, const std::vector<void *> &io_addrs) {
  if (fixed_mem_base_ == reinterpret_cast<uintptr_t>(mem_base_)) {
    GELOGD("[Update][OpIOAddrs] No need to update op input output addr.");
    return;
  }

  OpDescInfo *op_desc_info = exception_dumper_.MutableOpDescInfo(task_id, stream_id);
  if (op_desc_info == nullptr) {
    GELOGW("[Update][OpIOAddrs] Find op desc failed, task_id: %u, stream_id: %u.", task_id, stream_id);
    return;
  }
  size_t input_size = op_desc_info->input_addrs.size();
  size_t output_size = op_desc_info->output_addrs.size();
  if (input_size + output_size != io_addrs.size()) {
    GELOGW("[Update][OpIOAddrs] Op[%s] input size[%zu] and output size[%zu] is not equal to io addr size[%zu]",
           op_desc_info->op_name.c_str(), input_size, output_size, io_addrs.size());
    return;
  }

  vector<void *> input_addrs;
  vector<void *> output_addrs;
  for (size_t i = 0; i < io_addrs.size(); i++) {
    if (i < input_size) {
      input_addrs.emplace_back(GetRunAddress(io_addrs[i]));
    } else {
      output_addrs.emplace_back(GetRunAddress(io_addrs[i]));
    }
  }
  op_desc_info->input_addrs = input_addrs;
  op_desc_info->output_addrs = output_addrs;
  GELOGD("[Update][OpIOAddrs] Op [%s] update input output addr success.", op_desc_info->op_name.c_str());
}

///
/// @ingroup ge
/// @brief Get total useful size, in known subgraph, no need to allocate zero copy memory during initialization.
/// @param [in] total_useful_size: total mem size - zero copy size.
/// @return Status
///
Status DavinciModel::GetTotalMemSizeExcludeZeroCopy(int64_t &total_useful_size) {
  if (runtime_param_.mem_size < static_cast<uint64_t>(runtime_param_.zero_copy_size)) {
    REPORT_CALL_ERROR("E19999", "total mem size[%lu] is less than zero copy size[%ld] ", runtime_param_.mem_size,
                      runtime_param_.zero_copy_size);
    GELOGE(FAILED, "[Check][TotalMemSizeExcludeZeroCopy] failed, total mem size[%lu] is less than zero copy size[%ld]",
           runtime_param_.mem_size, runtime_param_.zero_copy_size);
    return FAILED;
  }
  total_useful_size = runtime_param_.mem_size - runtime_param_.zero_copy_size;
  return SUCCESS;
}

Status DavinciModel::GetEventIdForBlockingAicpuOp(const OpDescPtr &op_desc, rtStream_t stream, uint32_t &event_id) {
  GELOGI("Get event id for aicpu blocking op:%s", op_desc->GetName().c_str());
  auto it = stream_2_event_.find(stream);
  if (it != stream_2_event_.end()) {
    auto rt_ret = rtGetEventID(it->second, &event_id);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtGetEventID failed for op:%s(%s), ret:0x%X",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str(), rt_ret);
      GELOGE(RT_FAILED, "[Call][rtGetEventID] failed for op:%s(%s), ret:0x%X",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  } else {
    rtEvent_t rt_event = nullptr;
    auto rt_ret = rtEventCreateWithFlag(&rt_event, RT_EVENT_WITH_FLAG);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtEventCreateWithFlag failed for op:%s(%s), ret:0x%X",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str(), rt_ret);
      GELOGE(RT_FAILED, "[Call][rtEventCreateWithFlag] failed for op:%s(%s), ret:0x%X",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    rt_ret = rtGetEventID(rt_event, &event_id);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtGetEventID failed for op:%s(%s), ret:0x%X",
                        op_desc->GetName().c_str(), op_desc->GetType().c_str(), rt_ret);
      GELOGE(RT_FAILED, "[Call][rtGetEventID] failed for op:%s(%s), ret:0x%X",
             op_desc->GetName().c_str(), op_desc->GetType().c_str(), rt_ret);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    stream_2_event_.emplace(stream, rt_event);
  }
  return SUCCESS;
}

Status DavinciModel::GetEventByStream(const rtStream_t &stream, rtEvent_t &rt_event) {
  auto it = stream_2_event_.find(stream);
  if (it == stream_2_event_.end()) {
    REPORT_INNER_ERROR("E19999", "Get event failed");
    GELOGE(FAILED, "[Get][Event] Get event failed");
    return FAILED;
  }
  rt_event = it->second;
  return SUCCESS;
}
}  // namespace ge
