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

#include "graph/load/graph_loader.h"

#include <string>
#include <vector>

#include "framework/common/helper/model_helper.h"
#include "common/model_parser/model_parser.h"
#include "graph/ge_context.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/manager/graph_var_manager.h"

namespace ge {
Status GraphLoader::UnloadModel(uint32_t model_id) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  GELOGI("UnLoad model begin, model id:%u.", model_id);

  Status ret = model_manager->Stop(model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Stop][Model] failed. model id:%u", model_id);
  }

  ret = model_manager->Unload(model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Unload][Model] failed. model id:%u", model_id);
    return ret;
  }
  GELOGI("UnLoad model success, model id:%u.", model_id);
  return SUCCESS;
}

Status GraphLoader::LoadModelOnline(uint32_t &model_id, const std::shared_ptr<ge::GeRootModel> &ge_root_model_ptr,
                                    const std::shared_ptr<ModelListener> &listener) {
  GELOGI("Load model online begin.");
  rtError_t rt_ret = rtSetDevice(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtSetDevice failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtSetDevice] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    return RT_FAILED;
  }
  if (ge_root_model_ptr == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param ge_root_model_ptr nullptr, check invalid");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[LoadGraph][Check][Param] GE load graph model_ptr is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->LoadModelOnline(model_id, ge_root_model_ptr, listener);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] Online failed. ret = %u, model_id:%u", ret, model_id);
    rt_ret = rtDeviceReset(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtDeviceReset failed, device_id:%u, ret:0x%X",
                        GetContext().DeviceId(), rt_ret);
      GELOGE(RT_FAILED, "[Call][RtDeviceReset] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    }
    return ret;
  }

  if (ge_root_model_ptr->IsSpecificStream()) {
    GELOGI("No need to start a new thread to run model in specific scene.");
    rt_ret = rtDeviceReset(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtDeviceReset failed, device_id:%u, ret:0x%X",
                        GetContext().DeviceId(), rt_ret);
      GELOGE(RT_FAILED, "[Call][RtDeviceReset] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    }
    return SUCCESS;
  }
  ret = model_manager->Start(model_id);
  if (ret != SUCCESS) {
    if (model_manager->Unload(model_id) != SUCCESS) {
      GELOGE(ret, "[Unload][Model] failed while trying to unload after a failed start, model_id:%u.", model_id);
    }

    rt_ret = rtDeviceReset(GetContext().DeviceId());
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtDeviceReset failed, device_id:%u, ret:0x%X",
                        GetContext().DeviceId(), rt_ret);
      GELOGE(RT_FAILED, "[Call][RtDeviceReset] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    }

    GELOGE(ret, "[Start][Model] failed, model_id:%u.", model_id);
    return ret;
  }
  rt_ret = rtDeviceReset(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtDeviceReset failed, device_id:%u, ret:0x%X",
                      GetContext().DeviceId(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtDeviceReset] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    return RT_FAILED;
  }
  GELOGI("Load model online success, model_id:%u.", model_id);

  return SUCCESS;
}

Status GraphLoader::GetMaxUsedMemory(uint32_t model_id, uint64_t &max_size) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetMaxUsedMemory(model_id, max_size);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][GetMaxUsedMemory] failed, model_id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphLoader::LoadDataFromFile(const std::string &path, int32_t priority, ModelData &model_data) {
  if (!CheckInputPathValid(path, "model_file")) {
    GELOGE(ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID, "[Check][Param] model path is invalid:%s", path.c_str());
    return ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID;
  }

  GELOGI("Load model begin, model path is: %s", path.c_str());

  Status ret = ModelParserBase::LoadFromFile(path.c_str(), priority, model_data);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][LoadFromFile] failed. ret = %u, path:%s", ret, path.c_str());
    if (model_data.model_data != nullptr) {
      delete[] static_cast<char *>(model_data.model_data);
      model_data.model_data = nullptr;
    }
  }
  return ret;
}

Status GraphLoader::CommandHandle(const Command &command) {
  try {
    auto model_manager = ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->HandleCommand(command);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Handle][Command] failed, module_index:%lu.", command.module_index);

      return ret;
    }
  } catch (std::bad_alloc &) {
    REPORT_INNER_ERROR("E19999", "Bad memory allocation occur");
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[Handle][Command] failed, "
           "bad memory allocation occur, module_index:%lu.", command.module_index);

    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Some exceptions occur");
    GELOGE(FAILED, "[Handle][Command] failed, some exceptions occur, module_index:%lu.", command.module_index);

    return FAILED;
  }

  return SUCCESS;
}

Status GraphLoader::LoadModelFromData(uint32_t &model_id, const ModelData &model_data, void *dev_ptr,
                                      size_t mem_size, void *weight_ptr, size_t weight_size) {
  GELOGI("Load model begin, model_id:%u.", model_id);
  // For ACL, Open Device from App.
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->LoadModelOffline(
      model_id, model_data, nullptr, dev_ptr, mem_size, weight_ptr, weight_size);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] failed, model_id:%u.", model_id);
    return ret;
  }
  GELOGI("Load model success, model_id:%u.", model_id);
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Load task list from ModelData with queue.
/// @param [out] model_id: model id allocate from manager.
/// @param [in] model_data: Model data load from offline model.
/// @param [in] input_queue_ids: input queue ids create from user.
/// @param [in] output_queue_ids: input queue ids create from user.
/// @return: 0 for success / others for fail
///
Status GraphLoader::LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                                   const std::vector<uint32_t> &input_queue_ids,
                                   const std::vector<uint32_t> &output_queue_ids) {
  GELOGI("Load model with queue begin, model_id:%u.", model_id);

  // For ACL, Open Device from App.
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Load][Model] with queue failed, model_id:%u.", model_id);
    return ret;
  }

  GELOGI("Load model with queue success, model_id:%u.", model_id);
  return SUCCESS;
}

///
/// @ingroup domi_ome
/// @brief  execute model
/// @param [in] model_id  model id
/// @param [in] stream   stream to execute model on
/// @param [in] async_mode  is asynchronize mode.
/// @param [in] input_data  model input data
/// @param [in] input_desc  description of model input data
/// @param [out] output_data  model output data
/// @param [out] output_desc  description of model output data
///
Status GraphLoader::ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                                 const std::vector<GeTensorDesc> &input_desc, OutputData &output_data,
                                 std::vector<GeTensorDesc> &output_desc) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->ExecuteModel(model_id, stream, async_mode,
                                           input_data, input_desc, output_data, output_desc);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Execute][Model] failed, model_id:%u.", model_id);
    return ret;
  }

  GELOGD("Execute model success, model_id:%u.", model_id);
  return SUCCESS;
}

Status GraphLoader::GetMemoryInfo(int64_t &free) {
  rtError_t rt_ret = rtSetDevice(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtSetDevice failed, device_id:%u, ret:0x%X",
                      GetContext().DeviceId(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtSetDevice] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    return RT_FAILED;
  }
  size_t total_mem = 0;
  size_t free_mem = 0;
  rt_ret = rtMemGetInfo(&free_mem, &total_mem);
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemGetInfo failed, ret:0x%X", rt_ret);
    GELOGE(RT_FAILED, "[Call][RtMemGetInfo] failed, ret:0x%X", rt_ret);
    return RT_FAILED;
  }
  rt_ret = rtDeviceReset(GetContext().DeviceId());
  if (rt_ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtDeviceReset failed, device_id:%u, ret:0x%X",
                      GetContext().DeviceId(), rt_ret);
    GELOGE(RT_FAILED, "[Call][RtDeviceReset] failed, device_id:%u, ret:0x%X", GetContext().DeviceId(), rt_ret);
    return RT_FAILED;
  }
  // Add small page memory size
  free = static_cast<int64_t>(free_mem + VarManager::Instance(GetContext().SessionId())->GetUseMaxMemorySize() -
                              total_mem);
  GELOGI("GetMemoryInfo free[%zu], total[%zu], return free[%ld]", free_mem, total_mem, free);
  return SUCCESS;
}

Status GraphLoader::DestroyAicpuKernel(uint64_t session_id, uint32_t model_id, uint32_t sub_model_id) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->DestroyAicpuKernel(session_id, model_id, sub_model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Destroy][AicpuKernel] failed, session_id:%lu, model_id:%u, sub_model_id:%u.",
           session_id, model_id, sub_model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphLoader::DestroyAicpuSessionForInfer(uint32_t model_id) {
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->DestroyAicpuSessionForInfer(model_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][DestroyAicpuSessionForInfer] failed, model_id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}
}  // namespace ge
