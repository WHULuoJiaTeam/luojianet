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

#include "graph/load/model_manager/model_manager.h"

#include <string>

#include "aicpu/aicpu_schedule/aicpu_op_type_list.h"
#include "common/model_parser/model_parser.h"
#include "common/dump/dump_manager.h"
#include "framework/common/l2_cache_optimize.h"
#include "common/profiling/profiling_manager.h"
#include "common/ge_call_wrapper.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/model/ge_root_model.h"
#include "common/formats/utils/formats_trans_utils.h"

namespace ge {
thread_local uint32_t device_count = 0;
namespace {
const int kCmdParSize = 2;
const int kDumpCmdPairSize = 2;
const std::size_t kProfCmdParaMaxSize = 1000;
const std::size_t kProfStartCmdParaSize = 2;
const std::string kCmdTypeDump = "dump";
const std::string kCmdTypeProfInit = "prof_init";
const std::string kCmdTypeProfFinalize = "prof_finalize";
const std::string kCmdTypeProfStart = "prof_start";
const std::string kCmdTypeProfStop = "prof_stop";
const std::string kCmdTypeProfModelSubscribe = "prof_model_subscribe";
const std::string kCmdTypeProfModelUnsubscribe = "prof_model_cancel_subscribe";
const char *const kBatchLoadBuf = "batchLoadsoFrombuf";
const char *const kDeleteCustOp = "deleteCustOp";
const int kTimeSpecNano = 1000000000;
const int kTimeSpecMiro = 1000000;
const int kOpNameMaxSize = 100;
const uint64_t kInferSessionId = 0;
#pragma pack(push, 1)
struct CustAicpuSoBuf {
  uint64_t kernelSoBuf;
  uint32_t kernelSoBufLen;
  uint64_t kernelSoName;
  uint32_t kernelSoNameLen;
};
struct BatchLoadOpFromBufArgs {
  uint32_t soNum;
  uint64_t args;
};
#pragma pack(pop)
}  // namespace

DumpProperties ModelManager::dump_properties_;
std::mutex ModelManager::exeception_infos_mutex_;

std::shared_ptr<ModelManager> ModelManager::GetInstance() {
  static const std::shared_ptr<ModelManager> instance_ptr =
      shared_ptr<ModelManager>(new (std::nothrow) ModelManager(), ModelManager::FinalizeForPtr);
  return instance_ptr;
}

ModelManager::ModelManager() {
  max_model_id_ = 0;
  session_id_bias_ = 0;
}

Status ModelManager::KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType op_type, uint64_t session_id, uint32_t model_id,
                                    uint32_t sub_model_id) {
  STR_FWK_OP_KERNEL param_base = {};
  void *devicebase = nullptr;
  void *aicpu_kernel_addr = nullptr;
  const uint32_t kKernelType = 0;
  param_base.fwkKernelType = kKernelType;
  param_base.fwkKernelBase.fwk_kernel.opType = op_type;
  param_base.fwkKernelBase.fwk_kernel.sessionID = session_id;
  if (op_type == aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_DESTROY) {
    std::vector<uint64_t> v_aicpu_kernel;
    std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id) + "_" +
                            std::to_string(sub_model_id);
    std::lock_guard<std::recursive_mutex> lock(map_mutex_);
    auto iter = model_aicpu_kernel_.find(model_key);
    if (iter != model_aicpu_kernel_.end()) {
      GELOGD("kernel destroy session_id %lu, model_id %u, sub_model_id %u..", session_id, model_id, sub_model_id);
      v_aicpu_kernel = model_aicpu_kernel_.at(model_key);
      // Insert size of aicpu kernel vector in the first element
      v_aicpu_kernel.insert(v_aicpu_kernel.begin(), v_aicpu_kernel.size());

      auto kernel_size = sizeof(uint64_t) * (v_aicpu_kernel.size());
      rtError_t rt_ret = rtMalloc(&aicpu_kernel_addr, kernel_size, RT_MEMORY_HBM);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                      REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%zu, ret:0x%X", kernel_size, rt_ret);
                      GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%zu, ret:0x%X", kernel_size, rt_ret);
                      return RT_ERROR_TO_GE_STATUS(rt_ret);)

      rt_ret = rtMemcpy(aicpu_kernel_addr, kernel_size, v_aicpu_kernel.data(), kernel_size, RT_MEMCPY_HOST_TO_DEVICE);
      GE_IF_BOOL_EXEC(rt_ret != RT_ERROR_NONE,
                      REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", kernel_size, rt_ret);
                      GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", kernel_size, rt_ret);
                      GE_CHK_RT(rtFree(aicpu_kernel_addr)); return RT_ERROR_TO_GE_STATUS(rt_ret);)
      uint64_t kernel_id_addr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(aicpu_kernel_addr));
      param_base.fwkKernelBase.fwk_kernel.kernelID = kernel_id_addr;
      // In the scene of loading once and running many times, the kernel needs to be destroyed many times,
      // and connot be removed from kernel map.
    }
  }

  rtError_t rt_ret = rtMalloc(&(devicebase), sizeof(STR_FWK_OP_KERNEL), RT_MEMORY_HBM);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%zu, ret:0x%X", sizeof(STR_FWK_OP_KERNEL), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMalloc] failed. size:%zu, ret:0x%X", sizeof(STR_FWK_OP_KERNEL), rt_ret);
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret =
      rtMemcpy(devicebase, sizeof(STR_FWK_OP_KERNEL), &param_base, sizeof(STR_FWK_OP_KERNEL), RT_MEMCPY_HOST_TO_DEVICE);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", sizeof(STR_FWK_OP_KERNEL), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", sizeof(STR_FWK_OP_KERNEL), rt_ret);
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    GE_CHK_RT(rtFree(devicebase));
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rtStream_t stream = nullptr;
  rt_ret = rtStreamCreate(&stream, 0);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamCreate failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Create][Stream] failed. ret:0x%X", rt_ret);
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    GE_CHK_RT(rtFree(devicebase));
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }

  rt_ret = rtKernelLaunchEx(devicebase, sizeof(STR_FWK_OP_KERNEL), 0, stream);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtKernelLaunchEx failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtKernelLaunchEx] failed. ret:0x%X", rt_ret);
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    GE_CHK_RT(rtFree(devicebase));
    GE_CHK_RT(rtStreamDestroy(stream));
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtStreamSynchronize(stream);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamSynchronize failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtStreamSynchronize] failed. ret:0x%X", rt_ret);
    GE_IF_BOOL_EXEC(aicpu_kernel_addr != nullptr, GE_CHK_RT(rtFree(aicpu_kernel_addr)));
    GE_CHK_RT(rtFree(devicebase));
    GE_CHK_RT(rtStreamDestroy(stream));
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  if (aicpu_kernel_addr != nullptr) {
    rt_ret = rtFree(aicpu_kernel_addr);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtFree failed, ret:0x%X", rt_ret);
      GELOGE(RT_FAILED, "[Free][Memory] failed. ret:0x%X", rt_ret);
      GE_CHK_RT(rtFree(devicebase));
      GE_CHK_RT(rtStreamDestroy(stream));
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
  }
  rt_ret = rtFree(devicebase);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtFree failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Free][Memory] failed. ret:0x%X", rt_ret);
    GE_CHK_RT(rtStreamDestroy(stream));
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  rt_ret = rtStreamDestroy(stream);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamDestroy failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtStreamDestroy] failed. ret:0x%X", rt_ret);
    return RT_ERROR_TO_GE_STATUS(rt_ret);
  }
  return SUCCESS;
}

void ModelManager::DestroyAicpuSession(uint64_t session_id) {
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  auto it = sess_ids_.find(session_id);
  if (it == sess_ids_.end()) {
    GELOGI("The session: %lu not created.", session_id);
    return;
  } else {
    rtContext_t ctx = nullptr;
    bool has_ctx = (rtCtxGetCurrent(&ctx) == RT_ERROR_NONE);
    if (!has_ctx) {
      GELOGI("Set device %u.", GetContext().DeviceId());
      GE_CHK_RT(rtSetDevice(static_cast<int32_t>(GetContext().DeviceId())));
    }

    Status ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_SESSION_DESTROY, session_id, 0, 0);
    if (ret != SUCCESS) {
      GELOGW("The session: %lu destroy failed.", session_id);
    } else {
      (void)sess_ids_.erase(session_id);
      GELOGI("The session: %lu destroyed.", session_id);
    }

    if (!has_ctx) {
      GELOGI("Reset device %u.", GetContext().DeviceId());
      GE_CHK_RT(rtDeviceReset(static_cast<int32_t>(GetContext().DeviceId())));
    }
  }
}

ge::Status ModelManager::DestroyAicpuSessionForInfer(uint32_t model_id) {
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  auto hybrid_davinci_model = hybrid_model_map_.find(model_id);
  if (hybrid_davinci_model != hybrid_model_map_.end()) {
    uint64_t session_id = hybrid_davinci_model->second->GetSessionId();
    DestroyAicpuSession(session_id);
    return SUCCESS;
  }

  auto it = model_map_.find(model_id);
  if (it == model_map_.end()) {
    REPORT_INNER_ERROR("E19999", "Param model_id:%u can't find in model_map, check invalid", model_id);
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, "[Check][Param] model id %u does not exists.", model_id);
    return ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
  }
  uint64_t session_id = it->second->GetSessionId();
  DestroyAicpuSession(session_id);
  return SUCCESS;
}

ge::Status ModelManager::DestroyAicpuKernel(uint64_t session_id, uint32_t model_id, uint32_t sub_model_id) {
  GELOGD("destroy aicpu kernel in session_id %lu, model_id %u.", session_id, model_id);
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id) + "_" +
                          std::to_string(sub_model_id);
  if (model_aicpu_kernel_.find(model_key) != model_aicpu_kernel_.end()) {
    Status ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_KERNEL_DESTROY, session_id, model_id,
                                sub_model_id);
    if (ret != SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Call KernelLaunchEx fail, model_id:%u, sub_model_id:%u, session_id:%lu",
                        model_id, sub_model_id, session_id);
      GELOGE(FAILED, "[Call][KernelLaunchEx] fail, model_id:%u, sub_model_id:%u, session_id:%lu",
             model_id, sub_model_id, session_id);
      return FAILED;
    }
  }
  return SUCCESS;
}

ge::Status ModelManager::CreateAicpuKernel(uint64_t session_id, uint32_t model_id, uint32_t sub_model_id,
                                           uint64_t kernel_id) {
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  std::vector<uint64_t> v_aicpu_kernel;
  std::string model_key = std::to_string(session_id) + "_" + std::to_string(model_id) + "_" +
                          std::to_string(sub_model_id);
  if (model_aicpu_kernel_.find(model_key) != model_aicpu_kernel_.end()) {
    v_aicpu_kernel = model_aicpu_kernel_.at(model_key);
  }
  v_aicpu_kernel.push_back(kernel_id);
  model_aicpu_kernel_[model_key] = v_aicpu_kernel;
  return SUCCESS;
}

ModelManager::~ModelManager() {
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  model_map_.clear();
  model_aicpu_kernel_.clear();
  cust_aicpu_so_.clear();
  dump_exception_flag_ = false;

  GE_IF_BOOL_EXEC(device_count > 0, GE_CHK_RT(rtDeviceReset(0)));
}

ge::Status ModelManager::SetDynamicSize(uint32_t model_id, const std::vector<uint64_t> &batch_num,
                                        int32_t dynamic_type) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model->SetDynamicSize(batch_num, dynamic_type);
  return SUCCESS;
}

ge::Status ModelManager::DoLoadHybridModelOnline(uint32_t model_id, const string &om_name,
                                                 const shared_ptr<ge::GeRootModel> &ge_root_model,
                                                 const shared_ptr<ModelListener> &listener) {
  auto hybrid_model = hybrid::HybridDavinciModel::Create(ge_root_model);
  GE_CHECK_NOTNULL(hybrid_model);
  hybrid_model->SetListener(listener);
  hybrid_model->SetModelId(model_id);
  hybrid_model->SetDeviceId(GetContext().DeviceId());
  hybrid_model->SetOmName(om_name);
  GE_CHK_STATUS_RET(hybrid_model->Init(), "[Init][HybridModel] failed. model_id = %u", model_id);
  auto shared_model = std::shared_ptr<hybrid::HybridDavinciModel>(hybrid_model.release());
  InsertModel(model_id, shared_model);
  return SUCCESS;
}

bool ModelManager::IsNeedHybridLoad(ge::GeRootModel &ge_root_model) {
  auto root_graph = ge_root_model.GetRootGraph();
  if (root_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "root graph in param ge_root_model is nullptr, model_id:%u, "
                       "check invalid", ge_root_model.GetModelId());
    GELOGE(FAILED, "[Check][Param] root graph in param ge_root_model is nullptr, model_id:%u",
           ge_root_model.GetModelId());
    return false;
  }
  bool is_shape_unknown = root_graph->GetGraphUnknownFlag();
  bool is_dsp_partitioned_graph = false;
  (void)AttrUtils::GetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dsp_partitioned_graph);
  return is_shape_unknown || is_dsp_partitioned_graph || GetContext().GetHostExecFlag();
}

///
/// @ingroup domi_ome
/// @brief load model online
/// @return Status run result
///
Status ModelManager::LoadModelOnline(uint32_t &model_id, const shared_ptr<ge::GeRootModel> &ge_root_model,
                                     std::shared_ptr<ModelListener> listener) {
  GE_CHK_BOOL_RET_STATUS(listener.get() != nullptr, PARAM_INVALID, "[Check][Param] Param incorrect, listener is null");
  if (model_id == INVALID_MODEL_ID) {
    GenModelId(&model_id);
    GELOGD("Generate new model_id:%u", model_id);
  }
  auto name_to_model = ge_root_model->GetSubgraphInstanceNameToModel();
  string om_name;
  if (IsNeedHybridLoad(*ge_root_model)) {
    return DoLoadHybridModelOnline(model_id, om_name, ge_root_model, listener);
  }

  mmTimespec timespec = mmGetTickCount();
  std::shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(0, listener);
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model->SetProfileTime(MODEL_LOAD_START, (timespec.tv_sec * kTimeSpecNano +
                                                   timespec.tv_nsec));  // 1000 ^ 3 converts second to nanosecond
  davinci_model->SetId(model_id);
  davinci_model->SetDeviceId(GetContext().DeviceId());

  auto root_graph = ge_root_model->GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  string root_model_name = root_graph->GetName();
  GeModelPtr ge_model = name_to_model[root_model_name];
  Status ret = SUCCESS;
  do {
    GE_TIMESTAMP_START(Assign);
    GE_IF_BOOL_EXEC(SUCCESS != (ret = davinci_model->Assign(ge_model)), GELOGW("assign model to modeldef failed.");
                    break;);
    GE_TIMESTAMP_END(Assign, "GraphLoader::ModelAssign");
    uint64_t session_id = GetContext().SessionId();

    const DumpProperties &dump_properties = DumpManager::GetInstance().GetDumpProperties(session_id);
    davinci_model->SetDumpProperties(dump_properties);
    dump_properties_ = dump_properties;

    GE_TIMESTAMP_START(Init);
    GE_IF_BOOL_EXEC(SUCCESS != (ret = davinci_model->Init()), GELOGW("DavinciInit failed."); break;);
    GE_TIMESTAMP_END(Init, "GraphLoader::ModelInit");

    InsertModel(model_id, davinci_model);

    GELOGI("Parse model %u success.", model_id);
  } while (0);
  auto &profiling_manager = ProfilingManager::Instance();
  const auto &subcribe_info = profiling_manager.GetSubscribeInfo();
  if (subcribe_info.is_subscribe) {
    auto graph_id = davinci_model->GetRuntimeParam().graph_id;
    if (subcribe_info.graph_id == graph_id) {
      profiling_manager.SetGraphIdToModelMap(graph_id, model_id);
    }
    else {
      GELOGW("graph_id:%u is not in subcribe info.", graph_id);
    }
  }
  return ret;
}

void ModelManager::InsertModel(uint32_t model_id, std::shared_ptr<DavinciModel> &davinci_model) {
  GE_CHK_BOOL_EXEC(davinci_model != nullptr, return, "[Check][Param] davinci_model ptr is null, id:%u", model_id);
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  model_map_[model_id] = davinci_model;
}

void ModelManager::InsertModel(uint32_t model_id, shared_ptr<hybrid::HybridDavinciModel> &hybrid_model) {
  GE_CHK_BOOL_EXEC(hybrid_model != nullptr, return, "[Check][Param] hybrid_model ptr is null, id:%u", model_id);
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  hybrid_model_map_[model_id] = hybrid_model;
}

Status ModelManager::DeleteModel(uint32_t id) {
  // These two pointers are used to unbind erase() and model destruction process.
  std::shared_ptr<DavinciModel> tmp_model;
  std::shared_ptr<hybrid::HybridDavinciModel> tmp_hybrid_model;
  {
    std::lock_guard<std::recursive_mutex> lock(map_mutex_);

    auto it = model_map_.find(id);
    auto hybrid_model_it = hybrid_model_map_.find(id);
    if (it != model_map_.end()) {
      uint64_t session_id = it->second->GetSessionId();
      std::string model_key = std::to_string(session_id) + "_" + std::to_string(id)  + "_" +
                              std::to_string(it->second->SubModelId());
      auto iter_aicpu_kernel = model_aicpu_kernel_.find(model_key);
      if (iter_aicpu_kernel != model_aicpu_kernel_.end()) {
        (void)model_aicpu_kernel_.erase(iter_aicpu_kernel);
      }
      tmp_model = it->second;
      (void)model_map_.erase(it);
    } else if (hybrid_model_it != hybrid_model_map_.end()) {
      tmp_hybrid_model = hybrid_model_it->second;
      (void)hybrid_model_map_.erase(hybrid_model_it);
    } else {
      REPORT_INNER_ERROR("E19999", "model_id:%u not exist in model_map, check invalid", id);
      GELOGE(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, "model id %u does not exists.", id);
      return ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
    }
  }

  return SUCCESS;
}

std::shared_ptr<DavinciModel> ModelManager::GetModel(uint32_t id) {
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);

  auto it = model_map_.find(id);
  return (it == model_map_.end()) ? nullptr : it->second;
}

std::shared_ptr<hybrid::HybridDavinciModel> ModelManager::GetHybridModel(uint32_t id) {
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);

  auto it = hybrid_model_map_.find(id);
  return (it == hybrid_model_map_.end()) ? nullptr : it->second;
}

Status ModelManager::Unload(uint32_t model_id) {
  GE_CHK_STATUS_RET(DeleteModel(model_id), "[Delete][Model] failed, model id:%u", model_id);
  if (device_count > 0) {
    device_count--;
    GELOGI("Unload model %u success.", model_id);
  } else {
    GELOGI("Unload model %u success.no need reset device,device_count: %u", model_id, device_count);
  }
  std::lock_guard<std::mutex> lock(exeception_infos_mutex_);
  exception_infos_.clear();
  return SUCCESS;
}

Status ModelManager::UnloadModeldef(uint32_t model_id) {
  GE_CHK_STATUS_RET(DeleteModel(model_id), "[Delete][Model] failed, model id: %u", model_id);
  return SUCCESS;
}

Status ModelManager::DataInput(const InputData &input_data, OutputData &output_data) {
  GELOGI("calling the DataInput");
  shared_ptr<InputDataWrapper> data_wrap(new (std::nothrow) InputDataWrapper());
  GE_CHECK_NOTNULL(data_wrap);

  Status status = data_wrap->Init(input_data, output_data);
  if (status != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Init InputDataWrapper failed, input data index:%u", input_data.index);
    GELOGE(domi::PUSH_DATA_FAILED, "[Init][InputDataWrapper] failed, input data index:%u.", input_data.index);
    return domi::PUSH_DATA_FAILED;
  }

  uint32_t model_id = input_data.model_id;
  output_data.model_id = model_id;

  std::shared_ptr<DavinciModel> model = GetModel(model_id);

  GE_CHK_BOOL_RET_STATUS(model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id %u in InputData!", model_id);

  GE_IF_BOOL_EXEC(model->GetDataInputTid() == 0, model->SetDataInputTid(mmGetTid()));

  DataInputer *inputer = model->GetDataInputer();
  GE_CHECK_NOTNULL(inputer);
  if (inputer->Push(data_wrap) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "DataInputer queue is full, please call again later, model_id %u", model_id);
    GELOGE(domi::DATA_QUEUE_ISFULL, "[Call][Push] Data queue is full, please call again later, model_id %u ", model_id);
    return domi::DATA_QUEUE_ISFULL;
  }
  GELOGD("Data input success, model id:%u", model_id);

  return SUCCESS;
}

Status ModelManager::GetCurDynamicDims(const vector<vector<int64_t>> &user_real_input_dims,
                                       const vector<pair<string, vector<int64_t>>> &user_input_dims,
                                       vector<int32_t> &cur_dynamic_dims) {
  GELOGD("Start get cur dynamic dims.");
  if (user_real_input_dims.size() != user_input_dims.size()) {
    REPORT_INNER_ERROR("E19999", "Param user_real_input_dims.size:%zu != user_input_dims.size:%zu, "
                       "check invalid", user_real_input_dims.size(), user_input_dims.size());
    GELOGE(INTERNAL_ERROR,
           "[Check][Param] The input count of user:%zu should be equal to the data count of graph:%zu",
           user_real_input_dims.size(), user_input_dims.size());
    return INTERNAL_ERROR;
  }

  for (size_t i = 0; i < user_input_dims.size(); ++i) {
    if (user_real_input_dims[i].size() != user_input_dims[i].second.size()) {
      REPORT_INNER_ERROR("E19999", "Param user_real_input_dims[%zu].size:%zu != user_input_dims[%zu].size:%zu, "
                         "check invalid", i, user_real_input_dims[i].size(),
                         i, user_input_dims[i].second.size());
      GELOGE(INTERNAL_ERROR, "[Check][Param] The shape size:%zu of dynamic input:%s "
             "should be equal to the shape size of input shape:%zu.",
             user_real_input_dims[i].size(), user_input_dims[i].first.c_str(), user_input_dims[i].second.size());
      return INTERNAL_ERROR;
    }
    for (size_t j = 0; j < user_input_dims.at(i).second.size(); ++j) {
      if (user_input_dims.at(i).second.at(j) < 0) {
        cur_dynamic_dims.emplace_back(static_cast<int32_t>(user_real_input_dims[i][j]));
      }
    }
  }
  GELOGD("Cur dynamic dims is %s.", formats::JoinToString(cur_dynamic_dims).c_str());
  bool cur_dynamic_dims_valid = false;
  for (auto dynamic_dim : GetLocalOmeContext().dynamic_shape_dims) {
    if (dynamic_dim == formats::JoinToString(cur_dynamic_dims)) {
      cur_dynamic_dims_valid = true;
      break;
    }
  }
  if (!cur_dynamic_dims_valid) {
    REPORT_INNER_ERROR("E19999", "cur dynamic dims is %s, not exist in options, check invalid",
                       formats::JoinToString(cur_dynamic_dims).c_str());
    GELOGE(INTERNAL_ERROR, "[Check][Param] Cur dynamic dims is %s, not exist in options.",
           formats::JoinToString(cur_dynamic_dims).c_str());
    return INTERNAL_ERROR;
  }
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief load Input and output TensorInfo for Model
/// @return Status run result
///
Status ModelManager::DataInputTensor(uint32_t model_id, const std::vector<ge::Tensor> &inputs) {
  std::shared_ptr<DavinciModel> model = GetModel(model_id);
  auto hybrid_model = GetHybridModel(model_id);
  if (hybrid_model == nullptr) {
    GE_CHECK_NOTNULL(model);
  }

  InputData input_data;
  input_data.model_id = model_id;
  input_data.timeout = 0;
  input_data.timestamp = 0;
  input_data.index = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    DataBuffer data;
    const TensorDesc &tensor_desc = inputs[i].GetTensorDesc();
    data.data = reinterpret_cast<void *>(const_cast<uint8_t *>(inputs[i].GetData()));
    data.length = inputs[i].GetSize();
    data.placement = static_cast<uint32_t>(tensor_desc.GetPlacement());
    input_data.shapes.emplace_back(tensor_desc.GetShape().GetDims());
    input_data.blobs.push_back(data);
  }
  if (!GetLocalOmeContext().user_input_dims.empty() && GetLocalOmeContext().need_multi_batch) {
    std::vector<int32_t> cur_dynamic_dims;
    if (!GetLocalOmeContext().user_real_input_dims.empty()) {
      if (GetCurDynamicDims(GetLocalOmeContext().user_real_input_dims, GetLocalOmeContext().user_input_dims,
                            cur_dynamic_dims) != SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Get][CurDynamicDims] [Train_Dynamic] Failed to Parse real_dynamic_dims.");
        return INTERNAL_ERROR;
      }
      DataBuffer data;
      data.data = new(std::nothrow) int32_t[cur_dynamic_dims.size()];
      GE_CHECK_NOTNULL(data.data);
      uint32_t length = static_cast<uint32_t>(cur_dynamic_dims.size() * sizeof(int32_t));
      GE_CHK_BOOL_EXEC(memcpy_s(data.data, length, cur_dynamic_dims.data(), length) == EOK,
                       REPORT_CALL_ERROR("E19999", "memcpy data failed, size:%u", length);
                       delete[] reinterpret_cast<int32_t *>(data.data);
                       return INTERNAL_ERROR, "[Memcpy][Data] failed, size:%u.", length);
      data.length = length;
      input_data.blobs.push_back(data);
    }
  }

  OutputData output_data;
  output_data.model_id = model_id;
  output_data.index = 0;

  shared_ptr<InputDataWrapper> data_wrap(new (std::nothrow) InputDataWrapper());
  GE_CHECK_NOTNULL(data_wrap);

  GE_CHK_STATUS_EXEC(data_wrap->Init(input_data, output_data), return domi::PUSH_DATA_FAILED,
                     "[Init][InputDataWrapper] failed, input data model_id:%u.", model_id);

  if (hybrid_model != nullptr) {
    GE_CHK_STATUS_RET(hybrid_model->EnqueueData(data_wrap),
                      "[Enqueue][Data] Data queue is full, please call again later, model_id:%u", model_id);
    return SUCCESS;
  }

  GE_CHK_BOOL_RET_STATUS(model != nullptr, PARAM_INVALID,
                         "[Check][Param] Invalid model id %u in InputData!", model_id);

  DataInputer *inputer = model->GetDataInputer();
  GE_CHECK_NOTNULL(inputer);

  GE_CHK_STATUS_EXEC(inputer->Push(data_wrap), return domi::DATA_QUEUE_ISFULL,
                     "[Call][Push] Data queue is full, please call again later, model_id %u ", model_id);

  GELOGD("Data input success, model id:%u", model_id);

  return SUCCESS;
}
///
/// @ingroup domi_ome
/// @brief create model thread, start to execute model
/// @param [in] model_id Model ID to be started
/// @return Status model run result
/// @author
///
Status ModelManager::Start(uint32_t model_id) {
  auto hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(hybrid_model->ModelRunStart());
    GELOGI("Start hybrid model %u success.", model_id);
    return SUCCESS;
  }

  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);

  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id %u to start! ", model_id);

  Status status = davinci_model->ModelRunStart();
  if (status == SUCCESS) {
    GELOGI("Start model %u success.", model_id);
  }

  return status;
}

///
/// @ingroup domi_ome
/// @brief Model ID stop
/// @only when unloaded
/// @param [in] model_id Model ID to be stopped
/// @return Status model stop result
/// @author
///
Status ModelManager::Stop(uint32_t model_id) {
  auto hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    GE_CHK_STATUS_RET_NOLOG(hybrid_model->ModelRunStop());
    GELOGI("Stop hybrid model %u success.", model_id);
    return SUCCESS;
  }

  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id %u to stop!", model_id);

  Status status = davinci_model->ModelRunStop();
  if (status == SUCCESS) {
    GELOGI("Stop model %u success.", model_id);
  }

  return status;
}

///
/// @ingroup domi_ome
/// @brief Command handle
/// @iterator 1 only Ieference, Debug 2 modes
/// @param [in] command command to handle
/// @return Status command handle result
/// @author
///
Status ModelManager::HandleCommand(const Command &command) {
  static const std::map<std::string, std::function<uint32_t(const Command &)>> cmds = {
      {kCmdTypeDump, HandleDumpCommand}, {kCmdTypeProfInit, HandleProfInitCommand},
      {kCmdTypeProfFinalize, HandleProfFinalizeCommand}, {kCmdTypeProfStart, HandleProfStartCommand},
      {kCmdTypeProfStop, HandleProfStopCommand},
      {kCmdTypeProfModelSubscribe, HandleProfModelSubscribeCommand},
      {kCmdTypeProfModelUnsubscribe, HandleProfModelUnsubscribeCommand}};

  auto iter = cmds.find(command.cmd_type);
  if (iter == cmds.end()) {
    REPORT_INNER_ERROR("E19999", "Unsupported command:%s check", command.cmd_type.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] Unsupported command:%s", command.cmd_type.c_str());
    return PARAM_INVALID;
  } else {
    return iter->second(command);
  }
}

Status ModelManager::GetModelByCmd(const Command &command,
                                   std::shared_ptr<DavinciModel> &davinci_model) {
  if (command.cmd_params.size() < kCmdParSize) {
    REPORT_INNER_ERROR("E19999", "command.cmd_params.size:%zu < kCmdParSize:%u, command_type:%s, "
                       "check invalid", command.cmd_params.size(), kCmdParSize,
                       command.cmd_type.c_str());
    GELOGE(PARAM_INVALID, "[Check][Param] When the cmd_type is '%s', the size of cmd_params must larger than 2.",
           command.cmd_type.c_str());
    return PARAM_INVALID;
  }

  std::string map_key = command.cmd_params[0];
  std::string value = command.cmd_params[1];
   if (map_key == PROFILE_MODEL_ID) {
    int32_t model_id = 0;
    try {
      model_id = std::stoi(value);
    } catch (std::invalid_argument &) {
      REPORT_INNER_ERROR("E19999", "%s param:%s, check invalid", PROFILE_MODEL_ID.c_str(), value.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] Model id:%s is invalid.", value.c_str());
      return PARAM_INVALID;
    } catch (std::out_of_range &) {
      REPORT_INNER_ERROR("E19999", "%s param:%s, check out of range", PROFILE_MODEL_ID.c_str(), value.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] Model id:%s is out of range.", value.c_str());
      return PARAM_INVALID;
    } catch (...) {
      REPORT_INNER_ERROR("E19999", "%s param:%s, check cannot change to int", PROFILE_MODEL_ID.c_str(), value.c_str());
      GELOGE(FAILED, "[Check][Param] Model id:%s cannot change to int.", value.c_str());
      return FAILED;
    }

    auto model_manager = ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    davinci_model = model_manager->GetModel(static_cast<uint32_t>(model_id));
    if (davinci_model == nullptr) {
      REPORT_INNER_ERROR("E19999", "GetModel from model_manager fail, model_id:%u", model_id);
      GELOGE(FAILED, "[Get][Model] failed, Model id:%d is invaild or model is not loaded.", model_id);
      return FAILED;
    }
  } else {
    REPORT_INNER_ERROR("E19999", "Fisrt cmd_param not %s, check invalid", PROFILE_MODEL_ID.c_str());
    GELOGE(FAILED, "[Check][Param] The model_id parameter is not found in the command.");
    return FAILED;
  }

  return SUCCESS;
}

Status ModelManager::HandleProfModelSubscribeCommand(const Command &command) {
  std::shared_ptr<DavinciModel> davinci_model = nullptr;
  Status ret = GetModelByCmd(command, davinci_model);
  if (ret != SUCCESS) {
    return ret;
  }

  if (ProfilingManager::Instance().ProfModelSubscribe(command.module_index,
                                                      static_cast<void *>(davinci_model.get())) != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfModelSubscribe] failed, module_index:%lu.",
           command.module_index);
    return FAILED;
  }

  return SUCCESS;
}

Status ModelManager::HandleProfModelUnsubscribeCommand(const Command &command) {
  std::shared_ptr<DavinciModel> davinci_model = nullptr;
  Status ret = GetModelByCmd(command, davinci_model);
  if (ret != SUCCESS) {
    return ret;
  }
  auto &profiling_manager = ProfilingManager::Instance();
  if (profiling_manager.ProfModelUnsubscribe(static_cast<void *>(davinci_model.get())) != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfModelUnsubscribe] failed.");
    return FAILED;
  }
  auto is_subscribe = profiling_manager.GetSubscribeInfo().is_subscribe;
  if (is_subscribe) {
    profiling_manager.CleanSubscribeInfo();
  }
  return SUCCESS;
}

Status ModelManager::HandleProfInitCommand(const Command &command) {
  uint64_t module_index = command.module_index;
  if (ProfilingManager::Instance().ProfInit(module_index) != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfInit] failed, module_index:%lu.", module_index);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelManager::HandleProfFinalizeCommand(const Command &command) {
  if (ProfilingManager::Instance().ProfFinalize() != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfFinalize] failed.");
    return FAILED;
  }
  return SUCCESS;
}
/*
 * cmd para when prof start
 * "devNums:2"
 * "devIdList:1,2"
 * "profilingOption:PROF_OP_TRACE"
 * "aicoreMetrics:AICORE_ARITHMATIC_THROUGHPUT"
 */
Status ModelManager::HandleProfStartCommand(const Command &command) {
  if (command.cmd_params.size() < kProfStartCmdParaSize) {
    REPORT_INNER_ERROR("E19999", "command.cmd_params.size:%zu < %zu, check invalid",
                       command.cmd_params.size(), kProfStartCmdParaSize);
    GELOGE(PARAM_INVALID, "[Check][Param] When the cmd_type is 'profile start', "
           "the size:%zu of cmd_params must larger than 2.", command.cmd_params.size());
    return PARAM_INVALID;
  }
  if (command.cmd_params.size() > kProfCmdParaMaxSize) {
    REPORT_INNER_ERROR("E19999", "command.cmd_params.size:%zu > %zu, check invalid",
                       command.cmd_params.size(), kProfCmdParaMaxSize);
    GELOGE(PARAM_INVALID, "[Check][Param] Command param size[%zu] larger than max[1000].", command.cmd_params.size());
    return PARAM_INVALID;
  }

  std::map<std::string, std::string> cmd_params_map;
  uint32_t step = 2;
  for (uint32_t i = 0; i < command.cmd_params.size(); i += step) {
    if (i + 1 >= command.cmd_params.size()) {
      continue;
    }
    cmd_params_map[command.cmd_params[i]] = command.cmd_params[i + 1];
  }
  uint64_t module_index = command.module_index;
  if (ProfilingManager::Instance().ProfStartProfiling(module_index, cmd_params_map) != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfStartProfiling] failed, module_index:%lu.", module_index);
    return FAILED;
  }
  return SUCCESS;
}

Status ModelManager::HandleProfStopCommand(const Command &command) {
  if (command.cmd_params.size() < kProfStartCmdParaSize) {
    REPORT_INNER_ERROR("E19999", "command.cmd_params.size:%zu < %zu, check invalid",
                       command.cmd_params.size(), kProfStartCmdParaSize);
    GELOGE(PARAM_INVALID, "[Check][Param] When the cmd_type is 'profile stop', "
           "the size:%zu of cmd_params must larger than 2.", command.cmd_params.size());
    return PARAM_INVALID;
  }
  if (command.cmd_params.size() > kProfCmdParaMaxSize) {
    REPORT_INNER_ERROR("E19999", "command.cmd_params.size:%zu > %zu, check invalid",
                       command.cmd_params.size(), kProfCmdParaMaxSize);
    GELOGE(PARAM_INVALID, "[Check][Param] Command param size[%zu] larger than max[1000].", command.cmd_params.size());
    return PARAM_INVALID;
  }

  std::map<std::string, std::string> cmd_params_map;
  uint32_t step = 2;
  for (uint32_t i = 0; i < command.cmd_params.size(); i += step) {
    if (i + 1 >= command.cmd_params.size()) {
      continue;
    }
    cmd_params_map[command.cmd_params[i]] = command.cmd_params[i + 1];
  }
  uint64_t module_index = command.module_index;
  if (ProfilingManager::Instance().ProfStopProfiling(module_index, cmd_params_map) != SUCCESS) {
    GELOGE(FAILED, "[Handle][ProfStopProfiling] failed, module_index:%lu.", module_index);
    return FAILED;
  }
  return SUCCESS;
}

static Status ParserPara(const Command &command, const string &dump_key, string &dump_value) {
  auto iter = std::find(command.cmd_params.begin(), command.cmd_params.end(), dump_key);
  if (iter != command.cmd_params.end()) {
    ++iter;
    if (iter == command.cmd_params.end()) {
      REPORT_INNER_ERROR("E19999", "dump_key:%s can't find in command.param, check invalid", dump_key.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] dump_key:%s can't find in command.param, check invalid", dump_key.c_str());
      return PARAM_INVALID;
    }
    dump_value = *iter;
  }
  return SUCCESS;
}

Status ModelManager::HandleDumpCommand(const Command &command) {
  if (command.cmd_params.size() % kDumpCmdPairSize != 0) {
    REPORT_INNER_ERROR("E19999", "command.cmd_params.size:%zu MOD 2 != 0, check invalid", command.cmd_params.size());
    GELOGE(PARAM_INVALID, "[Check][Param] When the cmd_type is 'dump', "
           "the size:%zu of cmd_params must be a even number.", command.cmd_params.size());
    return PARAM_INVALID;
  }

  std::string dump_status("off");
  std::string dump_model(DUMP_ALL_MODEL);
  std::string dump_path("/");
  std::string dump_mode("output");
  std::set<std::string> dump_layers;

  auto ret = ParserPara(command, DUMP_STATUS, dump_status);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parser][DumpStatus] failed, ret:%d", ret);
    return FAILED;
  }
  GELOGI("dump status = %s.", dump_status.c_str());

  ret = ParserPara(command, DUMP_MODEL, dump_model);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parser][DumpModel] failed, ret:%d", ret);
    return FAILED;
  }
  GELOGI("dump model = %s.", dump_model.c_str());

  if (dump_status == "off" || dump_status == "OFF") {
    dump_properties_.DeletePropertyValue(dump_model);
    return SUCCESS;
  }

  for (size_t i = 0; i < command.cmd_params.size() / kDumpCmdPairSize; ++i) {
    if (command.cmd_params.at(i * kDumpCmdPairSize).find(DUMP_LAYER) != std::string::npos) {
      GELOGI("dump layer: %s.", command.cmd_params.at(i * kDumpCmdPairSize + 1).c_str());
      dump_layers.insert(command.cmd_params.at(i * kDumpCmdPairSize + 1));
    }
  }

  ret = ParserPara(command, DUMP_FILE_PATH, dump_path);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parser][DumpPath] failed, ret:%d", ret);
    return FAILED;
  }
  if (!dump_path.empty() && dump_path[dump_path.size() - 1] != '/') {
    dump_path = dump_path + "/";
  }
  dump_path = dump_path + CurrentTimeInStr() + "/";
  GELOGI("dump path = %s.", dump_path.c_str());

  ret = ParserPara(command, DUMP_MODE, dump_mode);
  if (ret != SUCCESS) {
    GELOGE(PARAM_INVALID, "[Parser][DumpMode] failed, ret:%d", ret);
    return FAILED;
  }
  GELOGI("dump mode = %s", dump_mode.c_str());

  dump_properties_.AddPropertyValue(dump_model, dump_layers);
  dump_properties_.SetDumpPath(dump_path);
  dump_properties_.SetDumpMode(dump_mode);

  return SUCCESS;
}

Status ModelManager::GetMaxUsedMemory(const uint32_t model_id, uint64_t &max_size) {
  auto hybrid_model = GetHybridModel(model_id);
  if (hybrid_model != nullptr) {
    max_size = 0;
    return SUCCESS;
  }

  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id:%u!", model_id);

  max_size = davinci_model->TotalMemSize();
  return SUCCESS;
}

Status ModelManager::GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, PARAM_INVALID,
                         "[Get][Model] failed, Invalid model id %u!", model_id);

  return davinci_model->GetInputOutputDescInfo(input_desc, output_desc);
}

Status ModelManager::GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                            vector<InputOutputDescInfo> &output_desc,
                                            std::vector<uint32_t> &inputFormats, std::vector<uint32_t> &outputFormats,
                                            bool new_model_desc) {
  std::shared_ptr<hybrid::HybridDavinciModel> hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    hybrid_davinci_model->SetModelDescVersion(new_model_desc);
    return hybrid_davinci_model->GetInputOutputDescInfo(input_desc, output_desc, inputFormats, outputFormats);
  }

  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid model id %u!", model_id);

  return davinci_model->GetInputOutputDescInfo(input_desc, output_desc, inputFormats, outputFormats, new_model_desc);
}

///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
///
Status ModelManager::GetDynamicBatchInfo(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                         int32_t &dynamic_type) {
  std::shared_ptr<hybrid::HybridDavinciModel> hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->GetDynamicBatchInfo(batch_info, dynamic_type);
  }

  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] failed, Invalid model id %u!", model_id);

  return davinci_model->GetDynamicBatchInfo(batch_info, dynamic_type);
}

///
/// @ingroup ge
/// @brief Get combined dynamic dims info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
///
Status ModelManager::GetCombinedDynamicDims(const uint32_t model_id, vector<vector<int64_t>> &batch_info) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid Model ID %u!", model_id);

  davinci_model->GetCombinedDynamicDims(batch_info);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get user designate shape order
/// @param [in] model_id
/// @param [out] user_input_shape_order
/// @return execute result
///
Status ModelManager::GetUserDesignateShapeOrder(const uint32_t model_id,
                                                std::vector<std::string> &user_input_shape_order) {
  auto hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    hybrid_davinci_model->GetUserDesignateShapeOrder(user_input_shape_order);
    return SUCCESS;
  }

  auto davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid Model ID %u!", model_id)
  davinci_model->GetUserDesignateShapeOrder(user_input_shape_order);
  return SUCCESS;
}

Status ModelManager::GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type) {
  auto davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid Model ID %u!", model_id);
  davinci_model->GetCurShape(batch_info, dynamic_type);
  return SUCCESS;
}

Status ModelManager::GetOpAttr(uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                               std::string &attr_value) {
  auto davinci_model = GetModel(model_id);
  if (davinci_model != nullptr) {
    return davinci_model->GetOpAttr(op_name, attr_name, attr_value);
  }
  std::shared_ptr<hybrid::HybridDavinciModel> hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    return hybrid_davinci_model->GetOpAttr(op_name, attr_name, attr_value);
  }
  GELOGE(ACL_ERROR_GE_EXEC_MODEL_ID_INVALID, "[Get][Model]Get model failed, invalid model id:%u.", model_id);
  REPORT_INNER_ERROR("E19999", "Get model failed, invalid model id:%u.", model_id);
  return ACL_ERROR_GE_EXEC_MODEL_ID_INVALID;
}

Status ModelManager::GetModelAttr(uint32_t model_id, std::vector<string> &dynamic_output_shape_info) {
  std::shared_ptr<hybrid::HybridDavinciModel> hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    hybrid_davinci_model->GetModelAttr(dynamic_output_shape_info);
    return SUCCESS;
  }

  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Failed, Invalid Model ID %u!", model_id);
  davinci_model->GetModelAttr(dynamic_output_shape_info);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get AIPP info
/// @param [in] model_id
/// @param [in] index
/// @param [out] aipp_info
/// @return execute result
///
Status ModelManager::GetAippInfo(const uint32_t model_id, uint32_t index, AippConfigInfo &aipp_info) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
      "[Get][Model] failed, invalid model_id is %u.", model_id);
  return davinci_model->GetAippInfo(index, aipp_info);
}

Status ModelManager::GetAippType(uint32_t model_id, uint32_t index, InputAippType &type, size_t &aipp_index) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
      "[Get][Model] failed, invalid model_id is %u.", model_id);
  return davinci_model->GetAippType(index, type, aipp_index);
}

Status ModelManager::GenSessionId(uint64_t &session_id) {
  const uint64_t kSessionTimeMask = 0xffffffffffff0000;
  const uint64_t kSessionPidMask  = 0x000000000000ff00;
  const uint64_t kSessionBiasMask = 0x00000000000000ff;

  const uint64_t kMaskPerOffset = 8;

  std::lock_guard<std::mutex> lock(session_id_create_mutex_);

  mmTimeval tv;
  if (mmGetTimeOfDay(&tv, nullptr) != 0) {
    REPORT_CALL_ERROR("E19999", "Call mmGetTimeOfDay fail. errmsg:%s", strerror(errno));
    GELOGE(INTERNAL_ERROR, "[Call][MmGetTimeOfDay] fail. errmsg:%s", strerror(errno));
    return INTERNAL_ERROR;
  }
  uint64_t timestamp = static_cast<uint64_t>(tv.tv_sec * kTimeSpecMiro + tv.tv_usec);  // 1000000us

  static uint32_t pid = mmGetPid();

  session_id_bias_++;

  session_id = ((timestamp<<kMaskPerOffset<<kMaskPerOffset) & kSessionTimeMask) +
               ((pid<<kMaskPerOffset) & kSessionPidMask) + (session_id_bias_ & kSessionBiasMask);

  GELOGD("Generate new session id: %lu.", session_id);
  return SUCCESS;
}

Status ModelManager::LoadModelOffline(uint32_t &model_id, const ModelData &model, shared_ptr<ModelListener> listener,
                                      void *dev_ptr, size_t mem_size, void *weight_ptr, size_t weight_size) {
  GE_CHK_BOOL_RET_STATUS(model.key.empty() || mmAccess2(model.key.c_str(), M_F_OK) == EN_OK,
                         ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Param] Input key file path %s is invalid, %s", model.key.c_str(), strerror(errno));
  GenModelId(&model_id);

  mmTimespec timespec = mmGetTickCount();

  ModelHelper model_helper;
  Status ret = model_helper.LoadRootModel(model);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][RootModel] failed, ret:%d, model_id:%u.", ret, model_id);
    return ret;
  }

  if (model_helper.GetModelType()) {
    bool is_shape_unknown = false;
    GE_CHK_STATUS_RET(model_helper.GetGeRootModel()->CheckIsUnknownShape(is_shape_unknown),
                      "[Check][IsUnknownShape] failed, model id:%u", model_id);
    if (is_shape_unknown || GetContext().GetHostExecFlag()) {
      return DoLoadHybridModelOnline(model_id, model.om_name, model_helper.GetGeRootModel(), listener);
    }
  }

  do {
    GeModelPtr ge_model = model_helper.GetGeModel();
    shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(model.priority, listener);
    if (davinci_model == nullptr) {
      REPORT_CALL_ERROR("E19999", "New DavinciModel fail");
      GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[New][DavinciModel] fail");
      return ACL_ERROR_GE_MEMORY_ALLOCATION;
    }
    davinci_model->SetProfileTime(MODEL_LOAD_START, (timespec.tv_sec * kTimeSpecNano +
                                                     timespec.tv_nsec));  // 1000 ^ 3 converts second to nanosecond
    ret = davinci_model->Assign(ge_model);
    if (ret != SUCCESS) {
      GELOGW("assign model failed.");
      break;
    }
    davinci_model->SetId(model_id);

    int32_t device_id = 0;
    rtError_t rt_ret = rtGetDevice(&device_id);
    if (rt_ret != RT_ERROR_NONE || device_id < 0) {
      REPORT_CALL_ERROR("E19999", "Call rtGetDevice failed, ret = 0x%X", rt_ret);
      GELOGE(rt_ret, "[Call][RtGetDevice] failed, ret = 0x%X, device_id = %d.", rt_ret, device_id);
      return RT_ERROR_TO_GE_STATUS(rt_ret);
    }
    davinci_model->SetDeviceId(device_id);
    davinci_model->SetOmName(model.om_name);
    if (DumpManager::GetInstance().GetDumpProperties(kInferSessionId).IsDumpOpen()) {
      davinci_model->SetDumpProperties(DumpManager::GetInstance().GetDumpProperties(kInferSessionId));
    } else {
      davinci_model->SetDumpProperties(dump_properties_);
    }

    /// In multi-threaded inference,  using the same session_id among multiple threads may cause some threads to fail.
    /// These session_ids come from the same model, so the values of session_id are the same.
    /// Update session_id for infer in load model to avoid the same session_id.
    uint64_t new_session_id;
    ret = GenSessionId(new_session_id);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, break, "[Generate][SessionId] for inference failed, ret:%d.", ret);
    ret = davinci_model->UpdateSessionId(new_session_id);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, break,
                                   "[Update][SessionId] for inference failed, session id:%lu.", new_session_id);

    ret = davinci_model->Init(dev_ptr, mem_size, weight_ptr, weight_size);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, break, "[Init][DavinciModel] failed, ret:%d.", ret);

    InsertModel(model_id, davinci_model);

    GELOGI("Parse model %u success.", model_id);

    GE_IF_BOOL_EXEC(ret == SUCCESS, device_count++);
  } while (0);

  return ret;
}

///
/// @ingroup ge
/// @brief ACL case, Load task list with queue.
/// @param [out] model_id: model id for manager.
/// @param [in] model_data: Model data load from offline model file.
/// @param [in] input_que_ids: input queue ids from user, num equals Data Op.
/// @param [in] output_que_ids: input queue ids from user, num equals NetOutput Op.
/// @return: 0 for success / others for fail
///
Status ModelManager::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                    const std::vector<uint32_t> &input_queue_ids,
                                    const std::vector<uint32_t> &output_queue_ids) {
  GE_CHK_BOOL_RET_STATUS(model_data.key.empty() || mmAccess2(model_data.key.c_str(), M_F_OK) == EN_OK,
                         ACL_ERROR_GE_PARAM_INVALID,
                         "[Check][Param] input key file path %s is not valid, %s",
                         model_data.key.c_str(), strerror(errno));

  ModelHelper model_helper;
  Status ret = model_helper.LoadModel(model_data);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] failed.");
    return ret;
  }

  shared_ptr<DavinciModel> davinci_model = MakeShared<DavinciModel>(model_data.priority, nullptr);
  if (davinci_model == nullptr) {
    REPORT_CALL_ERROR("E19999", "New DavinciModel fail");
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Create][Model] failed.");
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  ret = davinci_model->Assign(model_helper.GetGeModel());
  if (ret != SUCCESS) {
    GELOGE(ret, "[Assign][Model] failed, ret:%d.", ret);
    return ret;
  }

  /// In multi-threaded inference,  using the same session_id among multiple threads may cause some threads to fail.
  /// These session_ids come from the same model, so the values of session_id are the same.
  /// Update session_id for infer in load model to avoid the same session_id.
  uint64_t new_session_id;
  ret = GenSessionId(new_session_id);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret,
                                 "[Generate][SessionId] for infer failed, ret:%d.", ret);
  ret = davinci_model->UpdateSessionId(new_session_id);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret,
                                 "[Update][SessionId] for infer failed, SessionId:%lu.", new_session_id);

  GenModelId(&model_id);
  davinci_model->SetId(model_id);
  ret = davinci_model->SetQueIds(input_queue_ids, output_queue_ids);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][Ids] for model queue failed, ret:%d, model_id:%u.", ret, model_id);
    return ret;
  }

  davinci_model->SetDumpProperties(dump_properties_);

  ret = davinci_model->Init();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Init][Model] failed, ret:%d, model_id:%u.", ret, model_id);
    return ret;
  }

  InsertModel(model_id, davinci_model);
  GELOGI("Parse model %u success.", model_id);

  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief  ACL case, not start new thread, return result
/// @param [in] model_id  mode id
/// @param [in] stream   model stream
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  input data
/// @param [in] input_desc  description of input data
/// @param [out] output_data  output data
/// @param [out] output_desc  description of output data
///
Status ModelManager::ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                                  const std::vector<GeTensorDesc> &input_desc, OutputData &output_data,
                                  std::vector<GeTensorDesc> &output_desc) {
  std::shared_ptr<hybrid::HybridDavinciModel> hybrid_davinci_model = GetHybridModel(model_id);
  if (hybrid_davinci_model != nullptr) {
    auto inputs = input_data.blobs;
    auto outputs = output_data.blobs;

    Status status = hybrid_davinci_model->Execute(inputs, input_desc, outputs, output_desc, stream);
    if (status == SUCCESS) {
      GELOGI("Execute model %u success.", model_id);
    }
    return status;
  }

  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] Invalid model id %u, check whether model has been loaded or not.", model_id);

  if (davinci_model->NeedDestroyAicpuKernel()) {
    GELOGI("Start to destroy specified aicpu kernel.");
    // Zero copy is enabled by default, no need to judge.
    uint64_t session_id_davinci = davinci_model->GetSessionId();
    uint32_t model_id_davinci = davinci_model->GetModelId();
    uint32_t sub_model_id = davinci_model->SubModelId();
    Status status = DestroyAicpuKernel(session_id_davinci, model_id_davinci, sub_model_id);
    if (status != SUCCESS) {
      GELOGW("Destroy specified aicpu kernel failed, session id is %lu, model id is %u.", session_id_davinci,
             model_id_davinci);
    }
  }

  Status status = davinci_model->NnExecute(stream, async_mode, input_data, output_data);
  if (status == SUCCESS) {
    GELOGD("Execute model %u success.", model_id);
  }

  return status;
}

Status ModelManager::CreateAicpuSession(uint64_t session_id) {
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  auto it = sess_ids_.find(session_id);
  // never been created by any model
  if (it == sess_ids_.end()) {
    Status ret = KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType::FWK_ADPT_SESSION_CREATE, session_id, 0, 0);
    if (ret == SUCCESS) {
      (void)sess_ids_.insert(session_id);
      GELOGI("The session: %lu create success.", session_id);
    }
    return ret;
  }
  return SUCCESS;
}

Status ModelManager::LoadCustAicpuSo(const OpDescPtr &op_desc, const string &so_name, bool &loaded) {
  GELOGD("LoadCustAicpuSo in, op name %s, so name %s", op_desc->GetName().c_str(), so_name.c_str());
  std::lock_guard<std::mutex> lock(cust_aicpu_mutex_);
  CustAICPUKernelPtr aicpu_kernel = op_desc->TryGetExtAttr(OP_EXTATTR_CUSTAICPU_KERNEL, CustAICPUKernelPtr());
  if (aicpu_kernel == nullptr) {
    GELOGI("cust aicpu op %s has no corresponding kernel!", op_desc->GetName().c_str());
    return SUCCESS;
  }

  // get current context
  rtContext_t rt_cur_ctx = nullptr;
  auto rt_error = rtCtxGetCurrent(&rt_cur_ctx);
  if (rt_error != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCtxGetCurrent failed, ret = 0x%X", rt_error);
    GELOGE(RT_FAILED, "[Call][RtCtxGetCurrent] failed, runtime result is %d", static_cast<int>(rt_error));
    return RT_FAILED;
  }

  // use current context as resource key
  uintptr_t resource_id = reinterpret_cast<uintptr_t>(rt_cur_ctx);
  auto it = cust_aicpu_so_.find(resource_id);
  if (it == cust_aicpu_so_.end()) {
    std::map<string, CustAICPUKernelPtr> new_so_name;
    new_so_name.insert({so_name, aicpu_kernel});
    cust_aicpu_so_[resource_id] = new_so_name;
    loaded = false;
    GELOGD("LoadCustAicpuSo new aicpu so name %s, resource id %lu", so_name.c_str(), resource_id);
    return SUCCESS;
  }
  auto it_so_name = it->second.find(so_name);
  if (it_so_name == it->second.end()) {
    it->second.insert({so_name, aicpu_kernel});
    loaded = false;
    GELOGD("LoadCustAicpuSo add aicpu so name %s, resource id %lu", so_name.c_str(), resource_id);
    return SUCCESS;
  }
  loaded = true;
  GELOGD("LoadCustAicpuSo so name %s has been loaded.", so_name.c_str());
  return SUCCESS;
}

Status ModelManager::LaunchKernelCustAicpuSo(const string &kernel_name) {
  GELOGD("Aicpu kernel launch task in, kernel name %s.", kernel_name.c_str());
  std::lock_guard<std::mutex> lock(cust_aicpu_mutex_);
  if (cust_aicpu_so_.empty()) {
    return SUCCESS;
  }
  // get current context
  rtContext_t rt_cur_ctx = nullptr;
  auto rt_error = rtCtxGetCurrent(&rt_cur_ctx);
  if (rt_error != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtCtxGetCurrent failed, ret = 0x%X", rt_error);
    GELOGE(RT_FAILED, "[Call][RtCtxGetCurrent] failed, runtime result is %d", static_cast<int>(rt_error));
    return RT_FAILED;
  }
  uintptr_t resource_id = reinterpret_cast<uintptr_t>(rt_cur_ctx);
  auto it = cust_aicpu_so_.find(resource_id);
  if (it == cust_aicpu_so_.end()) {
    GELOGI("Cust aicpu so map is empty, context id %lu", resource_id);
    return SUCCESS;
  }

  rtStream_t stream = nullptr;
  vector<void *> allocated_mem;
  std::function<void()> callback = [&]() {
    for (auto mem : allocated_mem) {
      GE_CHK_RT(rtFree(mem));
    }
    if (stream != nullptr) {
      GE_CHK_RT(rtStreamDestroy(stream));
    }
  };
  GE_MAKE_GUARD(release, callback);

  rtError_t status;
  vector<CustAicpuSoBuf> v_cust_so;
  void *args = nullptr;

  for (const auto &it_so : it->second) {
    const void *aicpu_data = it_so.second->GetBinData();
    uint32_t aicpu_data_length = it_so.second->GetBinDataSize();
    string so_name = it_so.first;
    void *d_aicpu_data = nullptr;
    void *d_so_name = nullptr;

    status = rtMalloc(&d_aicpu_data, aicpu_data_length, RT_MEMORY_HBM);
    if (status != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMalloc failed, size:%u, ret = 0x%X", aicpu_data_length, status);
      GELOGE(RT_FAILED, "[Call][RtMalloc] failed, size:%u, ret = 0x%X", aicpu_data_length, status);
      return RT_ERROR_TO_GE_STATUS(status);
    }
    allocated_mem.push_back(d_aicpu_data);
    status = rtMalloc(&d_so_name, so_name.size(), RT_MEMORY_HBM);
    if (status != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMalloc fail, size:%zu, ret = 0x%X", so_name.size(), status);
      GELOGE(RT_FAILED, "[Call][RtMalloc] fail, size:%zu, ret = 0x%X", so_name.size(), status);
      return RT_ERROR_TO_GE_STATUS(status);
    }
    allocated_mem.push_back(d_so_name);
    GE_CHK_RT(rtMemcpy(d_aicpu_data, aicpu_data_length, aicpu_data, aicpu_data_length, RT_MEMCPY_HOST_TO_DEVICE));
    GE_CHK_RT(rtMemcpy(d_so_name, so_name.size(), reinterpret_cast<const void *>(so_name.c_str()),
                       so_name.size(), RT_MEMCPY_HOST_TO_DEVICE));

    CustAicpuSoBuf cust_aicpu_so_buf;
    cust_aicpu_so_buf.kernelSoBuf = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_aicpu_data));
    cust_aicpu_so_buf.kernelSoBufLen = aicpu_data_length;
    cust_aicpu_so_buf.kernelSoName = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_so_name));
    cust_aicpu_so_buf.kernelSoNameLen = so_name.size();
    v_cust_so.push_back(cust_aicpu_so_buf);
  }
  if (kernel_name == kDeleteCustOp) {
    (void)cust_aicpu_so_.erase(it);
  }

  uint32_t args_size = sizeof(CustAicpuSoBuf) * v_cust_so.size();
  status = rtMalloc(&args, args_size, RT_MEMORY_HBM);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc fail, size:%u, ret = 0x%X", args_size, status);
    GELOGE(RT_FAILED, "[Call][RtMalloc] fail, size:%u, ret = 0x%X", args_size, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }
  allocated_mem.push_back(args);
  GE_CHK_RT(rtMemcpy(args, args_size, v_cust_so.data(), args_size, RT_MEMCPY_HOST_TO_DEVICE));

  BatchLoadOpFromBufArgs batch_cust_so;
  batch_cust_so.soNum = v_cust_so.size();
  batch_cust_so.args = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(args));

  void *batch_args = nullptr;
  uint32_t batch_args_size = sizeof(BatchLoadOpFromBufArgs);
  status = rtMalloc(&batch_args, batch_args_size, RT_MEMORY_HBM);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMalloc fail, size:%u, ret = 0x%X", batch_args_size, status);
    GELOGE(RT_FAILED, "[Call][RtMalloc] fail, size:%u, ret = 0x%X", batch_args_size, status);
    return RT_ERROR_TO_GE_STATUS(status);
  }
  allocated_mem.push_back(batch_args);
  GE_CHK_RT(rtMemcpy(batch_args, batch_args_size, static_cast<void *>(&batch_cust_so),
                     batch_args_size, RT_MEMCPY_HOST_TO_DEVICE));

  GE_CHK_RT(rtStreamCreate(&stream, 0));
  GE_CHK_RT(rtCpuKernelLaunch(nullptr, kernel_name.c_str(), 1, batch_args, batch_args_size, nullptr, stream));

  status = rtStreamSynchronize(stream);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamSynchronize fail, ret = 0x%X", status);
    GELOGE(RT_FAILED, "[Call][RtStreamSynchronize] fail, ret = 0x%X", status);
    return RT_ERROR_TO_GE_STATUS(status);
  }
  GELOGI("Cpu kernel launch task success.");
  return SUCCESS;
}

Status ModelManager::ClearAicpuSo() {
  GE_CHK_STATUS_RET(LaunchKernelCustAicpuSo(kDeleteCustOp),
                    "[Call][LaunchKernelCustAicpuSo] delete cust op so failed.");
  return SUCCESS;
}

Status ModelManager::LaunchCustAicpuSo() {
  GE_CHK_STATUS_RET(LaunchKernelCustAicpuSo(kBatchLoadBuf),
                    "[Call][LaunchKernelCustAicpuSo] launch cust op so failed.");
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief get model memory size and weight
/// @param [in] const ModelData model: model type
/// @param [out] size_t memSize: model memory usage
///           size_t weightSize: model weight and memory size
/// @return SUCCESS success / others failure
///
Status ModelManager::GetModelMemAndWeightSize(const ModelData &model, size_t &mem_size, size_t &weight_size) {
  uint8_t *model_data = nullptr;
  uint32_t model_len = 0;
  Status ret = ModelParserBase::ParseModelContent(model, model_data, model_len);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ACL_ERROR_GE_PARAM_INVALID, "[Parse][ModelContent] failed!");

  OmFileLoadHelper om_file_helper;
  ret = om_file_helper.Init(model_data, model_len);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ret, "[Init][OmFileHelper] failed, ret:%d", ret);

  auto partition_table = reinterpret_cast<ModelPartitionTable *>(model_data);
  if (partition_table->num == 1) {
    REPORT_INNER_ERROR("E19999", "partition_table num in model_data is 1, check invalid");
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Param] om model is error, please use executable om model");
    return ACL_ERROR_GE_PARAM_INVALID;
  }
  ModelPartition task_partition;
  if (om_file_helper.GetModelPartition(ModelPartitionType::TASK_INFO, task_partition) != SUCCESS) {
    GELOGE(ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED, "[Get][ModelPartition] failed.");
    return ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED;
  }

  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  if (model_task_def == nullptr) {
    return MEMALLOC_FAILED;
  }
  if (task_partition.size != 0) {
    if (!ReadProtoFromArray(task_partition.data, static_cast<int>(task_partition.size), model_task_def.get())) {
      GELOGE(ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED, "[Read][Proto] From Array failed.");
      return ACL_ERROR_GE_EXEC_LOAD_TASK_PARTITION_FAILED;
    }
  }

  ModelPartition partition_weight;
  ret = om_file_helper.GetModelPartition(ModelPartitionType::WEIGHTS_DATA, partition_weight);
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(ret != SUCCESS, return ACL_ERROR_GE_EXEC_LOAD_WEIGHT_PARTITION_FAILED,
                                 "[Get][ModelPartition] failed. ret = %u", ret);

  mem_size = model_task_def->memory_size();
  weight_size = partition_weight.size;
  return SUCCESS;
}

void ModelManager::GenModelId(uint32_t *id) {
  if (id == nullptr) {
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(map_mutex_);
  *id = ++max_model_id_;
}

Status ModelManager::GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &orig_input_info) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] failed, invalid model_id is %u.", model_id);

  return davinci_model->GetOrigInputInfo(index, orig_input_info);
}

Status ModelManager::GetAllAippInputOutputDims(uint32_t model_id, uint32_t index,
                                               std::vector<InputOutputDims> &input_dims,
                                               std::vector<InputOutputDims> &output_dims) {
  std::shared_ptr<DavinciModel> davinci_model = GetModel(model_id);
  GE_CHK_BOOL_RET_STATUS(davinci_model != nullptr, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID,
                         "[Get][Model] failed, invalid model_id is %u.", model_id);

  return davinci_model->GetAllAippInputOutputDims(index, input_dims, output_dims);
}

bool ModelManager::IsDynamicShape(uint32_t model_id) {
  auto model = GetHybridModel(model_id);
  return model != nullptr;
}

ge::Status ModelManager::SyncExecuteModel(uint32_t model_id, const vector<GeTensor> &inputs,
                                          vector<GeTensor> &outputs) {
  auto model = GetHybridModel(model_id);
  if (model == nullptr) {
    REPORT_INNER_ERROR("E19999", "partition_table num in model_data is 1, check invalid");
    GELOGE(FAILED, "[Check][Param] Hybrid model not found. model id = %u.", model_id);
    return FAILED;
  }

  return model->Execute(inputs, outputs);
}

Status ModelManager::GetOpDescInfo(uint32_t device_id, uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info) {
  for (const auto &model : model_map_) {
    auto davinci_model = model.second;
    if (davinci_model->GetDeviceId() == device_id) {
      GELOGI("[Get][OpDescInfo] Start to GetOpDescInfo of device_id: %u in davinci model.", device_id);
      if (davinci_model->GetOpDescInfo(stream_id, task_id, op_desc_info)) {
        GELOGI("[Get][OpDescInfo] Find specific node of stream_id: %u, task_id: %u in davinci model.",
               stream_id, task_id);
        return SUCCESS;
      }
    }
  }
  for (const auto &model : hybrid_model_map_) {
    auto hybrid_model = model.second;
    if (hybrid_model->GetDeviceId() == device_id) {
      GELOGI("[Get][OpDescInfo] Start to GetOpDescInfo of device_id: %u in hybrid model.", device_id);
      if (hybrid_model->GetOpDescInfo(stream_id, task_id, op_desc_info)) {
        GELOGI("[Get][OpDescInfo] Find specific node of stream_id: %u, task_id: %u in hybrid model.",
               stream_id, task_id);
        return SUCCESS;
      }
    }
  }
  return FAILED;
}

Status ModelManager::EnableExceptionDump(const std::map<string, string> &options) {
  auto iter = options.find(OPTION_EXEC_ENABLE_EXCEPTION_DUMP);
  if (iter != options.end()) {
    GELOGI("Find option enable_exeception_dump is %s", iter->second.c_str());
    if (iter->second == "1") {
      dump_exception_flag_ = true;
      rtError_t rt_ret = rtSetTaskFailCallback(reinterpret_cast<rtTaskFailCallback>(ExceptionCallback));
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtSetTaskFailCallback fail, ret = 0x%X", rt_ret);
        GELOGE(RT_FAILED, "[Call][RtSetTaskFailCallback] fail, ret = 0x%X", rt_ret);
        return RT_ERROR_TO_GE_STATUS(rt_ret);
      }
    } else {
      GELOGI("Option enable exception dump is %s", iter->second.c_str());
    }
  } else {
    GELOGI("Not find option enable exception dump");
  }
  return SUCCESS;
}

Status ModelManager::LaunchKernelCheckAicpuOp(std::vector<std::string> &aicpu_optype_list,
                                              std::vector<std::string> &aicpu_tf_optype_list) {
  std::string kernel_name = "checkOpType";
  GELOGI("LaunchKernelCheckAicpuOpType in, kernel name %s", kernel_name.c_str());
  std::lock_guard<std::mutex> lock(cust_aicpu_mutex_);
  std::vector<SysOpInfo> req_aicpu_op_info_list;
  std::vector<SysOpInfo> res_aicpu_op_info_list;
  std::vector<ReturnCode> res_ret_code_list;

  if (aicpu_optype_list.empty() && aicpu_tf_optype_list.empty()) {
    GELOGI("No need to check aicpu op type.");
    return SUCCESS;
  }

  vector<void *> allocated_mem;
  rtError_t status;
  rtStream_t stream = nullptr;
  void *args = nullptr;

  void *d_req_op_list = nullptr;
  void *d_res_op_list = nullptr;
  void *d_ret_code_list = nullptr;

  size_t aicpu_op_nums = aicpu_optype_list.size();
  size_t tf_op_nums = aicpu_tf_optype_list.size();
  size_t op_nums = aicpu_op_nums + tf_op_nums;
  std::function<void()> callback = [&]() {
    for (auto mem : allocated_mem) {
      GE_CHK_RT(rtFree(mem));
    }
  };
  GE_MAKE_GUARD(release, callback);
  // malloc sysOpInfoList in SysOpCheckInfo
  GE_CHK_RT_RET(rtMalloc(&d_req_op_list, op_nums * sizeof(SysOpInfo), RT_MEMORY_HBM));
  allocated_mem.push_back(d_req_op_list);

  // malloc sysOpInfoList in SysOpCheckResp
  GE_CHK_RT_RET(rtMalloc(&d_res_op_list, op_nums * sizeof(SysOpInfo), RT_MEMORY_HBM));
  allocated_mem.push_back(d_res_op_list);

  // malloc returnCodeList in SysOpCheckResp
  GE_CHK_RT_RET(rtMalloc(&d_ret_code_list, op_nums * sizeof(ReturnCode), RT_MEMORY_HBM));
  allocated_mem.push_back(d_ret_code_list);

  for (const auto &op_type : aicpu_optype_list) {
    SysOpInfo op_info;
    // malloc op_type name in SysOpInfo
    void *d_op_type_name = nullptr;
    GE_CHK_RT_RET(rtMalloc(&d_op_type_name, op_type.length(), RT_MEMORY_HBM));

    allocated_mem.push_back(d_op_type_name);
    GE_CHK_RT(rtMemcpy(d_op_type_name, op_type.length(), op_type.c_str(), op_type.length(), RT_MEMCPY_HOST_TO_DEVICE));
    op_info.opType = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_op_type_name));
    op_info.opLen = op_type.length();
    op_info.kernelsType = CPU_KERNEL;
    req_aicpu_op_info_list.emplace_back(op_info);
  }

  for (const auto &op_type : aicpu_tf_optype_list) {
    SysOpInfo op_info;
    // malloc op_type name in SysOpInfo
    void *d_op_type_name = nullptr;
    GE_CHK_RT_RET(rtMalloc(&d_op_type_name, op_type.length(), RT_MEMORY_HBM));

    allocated_mem.push_back(d_op_type_name);
    GE_CHK_RT(rtMemcpy(d_op_type_name, op_type.size(), op_type.c_str(), op_type.size(), RT_MEMCPY_HOST_TO_DEVICE));
    op_info.opType = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_op_type_name));
    op_info.opLen = op_type.size();
    op_info.kernelsType = TF_KERNEL;
    req_aicpu_op_info_list.emplace_back(op_info);
  }
  GELOGI("Check aicpu op all attr size: %zu, real attr size: %zu.", op_nums, req_aicpu_op_info_list.size());
  GE_CHK_RT(rtMemcpy(d_req_op_list, sizeof(SysOpInfo) * req_aicpu_op_info_list.size(), req_aicpu_op_info_list.data(),
                     sizeof(SysOpInfo) * req_aicpu_op_info_list.size(), RT_MEMCPY_HOST_TO_DEVICE));

  SysOpCheckInfo op_check_info_req = { 0 };
  SysOpCheckResp op_check_info_res = { 0 };
  op_check_info_req.opListNum = op_nums;
  op_check_info_req.offSetLen = sizeof(SysOpCheckInfo);
  op_check_info_req.sysOpInfoList = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_req_op_list));

  op_check_info_res.opListNum = 0;
  op_check_info_res.isWithoutJson = 0;
  op_check_info_res.returnCodeList = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_ret_code_list));
  op_check_info_res.sysOpInfoList = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(d_res_op_list));

  uint32_t args_size = sizeof(SysOpCheckInfo) + sizeof(SysOpCheckResp);
  GE_CHK_RT_RET(rtMalloc(&args, args_size, RT_MEMORY_HBM));

  allocated_mem.push_back(args);
  GE_CHK_RT(rtMemcpy(args, sizeof(SysOpCheckInfo), reinterpret_cast<void *>(&op_check_info_req), sizeof(SysOpCheckInfo),
                     RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT(rtMemcpy(
    reinterpret_cast<void *>(static_cast<uintptr_t>(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(args)) +
    op_check_info_req.offSetLen)), sizeof(SysOpCheckResp), reinterpret_cast<void *>(&op_check_info_res),
    sizeof(SysOpCheckResp), RT_MEMCPY_HOST_TO_DEVICE));
  GE_CHK_RT(rtStreamCreate(&stream, 0));
  GE_CHK_RT(rtCpuKernelLaunch(nullptr, kernel_name.c_str(), 1, args, args_size, nullptr, stream));

  status = rtStreamSynchronize(stream);
  if (status != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtStreamSynchronize fail, ret = 0x%X", status);
    GELOGE(RT_FAILED, "[Call][RtStreamSynchronize] failed, ret:0x%X", status);
    GE_CHK_RT(rtStreamDestroy(stream));
    return RT_ERROR_TO_GE_STATUS(status);
  }

  // Check the response
  SysOpCheckResp *d_op_check_info_res =
    reinterpret_cast<SysOpCheckResp *>(reinterpret_cast<void *>(static_cast<uintptr_t>(static_cast<uint64_t>(
    reinterpret_cast<uintptr_t>(args)) + op_check_info_req.offSetLen)));
  (void)memset_s(&op_check_info_res, sizeof(SysOpCheckResp), 0, sizeof(SysOpCheckResp));
  GE_CHK_RT(rtMemcpy(&op_check_info_res, sizeof(SysOpCheckResp), d_op_check_info_res, sizeof(SysOpCheckResp),
                     RT_MEMCPY_DEVICE_TO_HOST));

  if (op_check_info_res.isWithoutJson) {
    GELOGI("No need to check aicpu in this scenoria.");
    GE_CHK_RT(rtStreamDestroy(stream));
    return SUCCESS;
  }
  uint64_t res_op_nums = op_check_info_res.opListNum;
  GELOGI("Check aicpu type, is without json: %d, res op num: %lu.", op_check_info_res.isWithoutJson, res_op_nums);
  if (res_op_nums != 0) {
    res_ret_code_list.clear();
    res_ret_code_list.resize(res_op_nums);
    res_aicpu_op_info_list.clear();
    res_aicpu_op_info_list.resize(res_op_nums);
    GE_CHK_RT(rtMemcpy(res_ret_code_list.data(), sizeof(ReturnCode) * res_op_nums,
                       reinterpret_cast<void *>(static_cast<uintptr_t>(op_check_info_res.returnCodeList)),
                       sizeof(ReturnCode) * res_op_nums, RT_MEMCPY_DEVICE_TO_HOST));
    GE_CHK_RT(rtMemcpy(res_aicpu_op_info_list.data(), sizeof(SysOpInfo) * res_op_nums,
                       reinterpret_cast<void *>(static_cast<uintptr_t>(op_check_info_res.sysOpInfoList)),
                       sizeof(SysOpInfo) * res_op_nums, RT_MEMCPY_DEVICE_TO_HOST));
    if (res_ret_code_list.size() != res_aicpu_op_info_list.size() || res_ret_code_list.size() != res_op_nums) {
      REPORT_INNER_ERROR("E19999", "res_ret_code_list.size:%zu res_aicpu_op_info_list.size:%zu res_op_nums:%lu "
                         "not equal, check invalid",
                         res_ret_code_list.size(), res_aicpu_op_info_list.size(), res_op_nums);
      GELOGE(FAILED, "[Check][Param] Number:%zu of retcode is not equal to number:%zu of op type or not equal %lu.",
             res_ret_code_list.size(), res_aicpu_op_info_list.size(), res_op_nums);
      GE_CHK_RT(rtStreamDestroy(stream));
      return FAILED;
    }
    std::string fail_reason;
    for (uint32_t i = 0; i < res_op_nums; i++) {
      ReturnCode ret_code = res_ret_code_list.at(i);
      SysOpInfo aicpu_info = res_aicpu_op_info_list.at(i);
      GELOGI("Not support aicpu op type: %lu, kernel_type:%d, opLen:%lu, ret_code:%d", aicpu_info.opType,
             aicpu_info.kernelsType, aicpu_info.opLen, ret_code);
      std::vector<char> op_name;
      op_name.clear();
      op_name.resize(kOpNameMaxSize);
      GE_CHK_RT(rtMemcpy(op_name.data(), aicpu_info.opLen,
                         reinterpret_cast<void *>(static_cast<uintptr_t>(aicpu_info.opType)),
                         aicpu_info.opLen, RT_MEMCPY_DEVICE_TO_HOST));
      std::string kernel_type =
          (static_cast<OpKernelType>(aicpu_info.kernelsType) == TF_KERNEL) ? "TF_KERNEL" : "CPU_KERNEL";
      string op_name_str(op_name.data());
      fail_reason += "op_type: " + op_name_str + " kernel_type: " + kernel_type +
                     "  ret code:" + std::to_string(static_cast<int>(ret_code)) +
                     "<0: op_type, 1: format, 2: datatype> \n";
    }
    fail_reason += "not support.";
    REPORT_INNER_ERROR("E19999", "Check aicpu op_type failed, details:%s", fail_reason.c_str());
    GELOGE(FAILED, "[Check][Param] Check aicpu op_type failed. details:%s", fail_reason.c_str());
    GE_CHK_RT(rtStreamDestroy(stream));
    return FAILED;
  }

  GE_CHK_RT(rtStreamDestroy(stream));
  GELOGI("Cpu kernel launch check optype task success.");
  return SUCCESS;
}

Status ModelManager::CheckAicpuOpList(GeModelPtr ge_model) {
  std::vector<std::string> aicpu_optype_list;
  std::vector<std::string> aicpu_tf_optype_list;
  bool aicpu_need_check = ge::AttrUtils::GetListStr(ge_model, "needCheckCpu", aicpu_optype_list);
  bool tf_need_check = ge::AttrUtils::GetListStr(ge_model, "needCheckTf", aicpu_tf_optype_list);
  if (!aicpu_need_check && !tf_need_check) {
    GELOGI("Graph:%s No need to check aicpu optype.", ge_model->GetGraph().GetName().c_str());
    return SUCCESS;
  }
  GE_CHK_STATUS_RET(LaunchKernelCheckAicpuOp(aicpu_optype_list, aicpu_tf_optype_list),
                    "[Call][LaunchKernelCheckAicpuOp] failed.");
  return SUCCESS;
}
}  // namespace ge
