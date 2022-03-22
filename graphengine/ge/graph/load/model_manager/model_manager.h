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

#ifndef GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_
#define GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_

#include <common/model/ge_root_model.h>
#include <stdint.h>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "cce/aicpu_engine_struct.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/ge_types.h"
#include "framework/common/helper/model_helper.h"
#include "framework/common/helper/om_file_helper.h"
#include "common/properties_manager.h"
#include "framework/common/types.h"
#include "external/ge/ge_api_types.h"
#include "graph/ge_context.h"
#include "graph/model.h"
#include "hybrid/hybrid_davinci_model.h"
#include "runtime/base.h"

namespace ge {
class DavinciModel;

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY ModelManager {
 public:
  static std::shared_ptr<ModelManager> GetInstance();
  static void FinalizeForPtr(ModelManager *) {}

  ///
  /// @ingroup domi_ome
  /// @brief load and init model
  /// @param [in] model_id model id
  /// @param [in] model including model ptr and size
  /// @param [in] listener used to return result
  /// @param [in/out] info model task generate info
  /// @return Status run result
  /// @author
  ///
  ge::Status LoadModelOffline(uint32_t &model_id, const ModelData &model,
                              std::shared_ptr<ModelListener> listener = nullptr, void *dev_ptr = nullptr,
                              size_t mem_size = 0, void *weight_ptr = nullptr, size_t weight_size = 0);

  ///
  /// @ingroup domi_ome
  /// @brief load and init model
  /// @param [out] model_id model id
  /// @param [in] model modeldef datatype
  /// @param [in] listener used to return result
  /// @param [in] isTrainMode model type
  /// @return Status run result
  /// @author @
  ///
  ge::Status LoadModelOnline(uint32_t &model_id, const std::shared_ptr<ge::GeRootModel> &ge_root_model,
                             std::shared_ptr<ModelListener> listener);

  ge::Status DoLoadHybridModelOnline(uint32_t model_id, const string &model_name,
                                     const shared_ptr<ge::GeRootModel> &ge_root_model,
                                     const std::shared_ptr<ModelListener> &listener);

  ///
  /// @ingroup ge
  /// @brief ACL case, Load task list with queue.
  /// @param [out] model_id: model id for manager.
  /// @param [in] model_data: Model data load from offline model file.
  /// @param [in] input_que_ids: input queue ids from user, num equals Data Op.
  /// @param [in] output_que_ids: input queue ids from user, num equals NetOutput Op.
  /// @return: 0 for success / others for fail
  ///
  ge::Status LoadModelWithQ(uint32_t &model_id, const ModelData &model_data,
                            const std::vector<uint32_t> &input_queue_ids,
                            const std::vector<uint32_t> &output_queue_ids);

  ///
  /// @ingroup domi_ome
  /// @brief unload model and free resources
  /// @param [in] model_id model id
  /// @return Status run result
  /// @author
  ///
  ge::Status Unload(uint32_t model_id);

  ///
  /// @ingroup omm
  /// @brief unload model and free resources
  /// @param [in] model_id model id
  /// @return Status run result
  /// @author
  ///
  ge::Status UnloadModeldef(uint32_t model_id);

  ///
  /// @ingroup domi_ome
  /// @brief process input data asynchronously
  /// cannot be invoked by multiple thread
  /// if one fails, other continue
  /// @param [in] input_data   input data
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  /// @return MODEL_NOT_READY  model not ready
  /// @return PUSH_DATA_FAILED push data into model queue failed
  /// @author
  ///
  ge::Status DataInput(const InputData &input_data, OutputData &output_data);

  ge::Status DataInputTensor(uint32_t model_id, const std::vector<ge::Tensor> &inputs);

  ///
  /// @ingroup domi_ome
  /// @brief Get cur_dynamic_dims for all input.
  /// @param [in] vector<vector<int64_t>> &user_real_input_dims: dims info of all user_inputs.
  /// @param [in] vector<pair<string, vector<int64_t>>> &user_input_dims: key:name. value:dynamic dims from option.
  /// @param [out] vector<int32_t> &cur_dynamic_dims: real dims gather, where the index of -1.
  /// @return 0: SUCCESS / others: INTERNAL_ERROR
  ///
  Status GetCurDynamicDims(const vector<vector<int64_t>> &user_real_input_dims,
                           const vector<pair<string, vector<int64_t>>> &user_input_dims,
                           vector<int32_t> &cur_dynamic_dims);

  ///
  /// @ingroup domi_ome
  /// @brief model start to run
  ///
  ge::Status Start(uint32_t model_id);

  ///
  /// @ingroup domi_ome
  /// @brief  ACL case, do not start new thread, return result
  /// @param [in] model_id  model id
  /// @param [in] stream   model stream
  /// @param [in] async_mode  is asynchronize mode.
  /// @param [in] input_data  model input data
  /// @param [in] input_desc  description of model input data
  /// @param [out] output_data  model output data
  /// @param [out] output_desc  description of model output data
  ///
  ge::Status ExecuteModel(uint32_t model_id, rtStream_t stream, bool async_mode, const InputData &input_data,
                          const std::vector<GeTensorDesc> &input_desc, OutputData &output_data,
                          std::vector<GeTensorDesc> &output_desc);

  ge::Status SyncExecuteModel(uint32_t model_id, const std::vector<GeTensor> &inputs, std::vector<GeTensor> &outputs);

  ///
  /// @ingroup domi_ome
  /// @brief model stop
  ///
  ge::Status Stop(uint32_t model_id);

  ///
  /// @ingroup domi_ome
  /// @brief comment handle function
  ///
  ge::Status HandleCommand(const Command &command);
  static ge::Status HandleDumpCommand(const Command &command);
  static ge::Status HandleProfModelSubscribeCommand(const Command &command);
  static ge::Status HandleProfModelUnsubscribeCommand(const Command &command);
  static ge::Status HandleProfInitCommand(const Command &command);
  static ge::Status HandleProfFinalizeCommand(const Command &command);
  static ge::Status HandleProfStartCommand(const Command &command);
  static ge::Status HandleProfStopCommand(const Command &command);

  static ge::Status GetModelByCmd(const Command &command,
                                  std::shared_ptr<DavinciModel> &davinci_model);
  ///
  /// @ingroup domi_ome
  /// @brief get model memory usage
  /// @param [in] model_id  model id
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  ///
  ge::Status GetMaxUsedMemory(const uint32_t model_id, uint64_t &max_size);

  ///
  /// @ingroup domi_ome
  /// @brief get model input and output size
  /// @param [in] model_id  model id
  /// @param [out] input_shape   input tensor
  /// @param [out] output_shape  output tensor
  /// @return SUCCESS          success
  /// @return PARAM_INVALID    parameter invalid
  ///
  ge::Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                    std::vector<InputOutputDescInfo> &output_desc);

  ge::Status GetInputOutputDescInfo(const uint32_t model_id, std::vector<InputOutputDescInfo> &input_desc,
                                    std::vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &inputFormats,
                                    std::vector<uint32_t> &outputFormats, bool new_model_desc = false);
  ///
  /// @ingroup ge
  /// @brief Get dynamic batch_info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @param [out] dynamic_type
  /// @return execute result
  ///
  ge::Status GetDynamicBatchInfo(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                 int32_t &dynamic_type);
  ///
  /// @ingroup ge
  /// @brief Get combined dynamic dims info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @return execute result
  ///
  ge::Status GetCombinedDynamicDims(const uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info);

  ///
  /// @ingroup ge
  /// @brief Get user designate shape order
  /// @param [in] model_id
  /// @param [out] user_input_shape_order
  /// @return execute result
  ///
  Status GetUserDesignateShapeOrder(const uint32_t model_id, std::vector<std::string> &user_input_shape_order);

  ///
  /// @ingroup ge
  /// @brief Get AIPP info
  /// @param [in] model_id
  /// @param [in] index
  /// @param [out] aipp_info
  /// @return execute result
  ///
  ge::Status GetAippInfo(const uint32_t model_id, uint32_t index, AippConfigInfo &aipp_info);

  ge::Status GetAippType(uint32_t model_id, uint32_t index, InputAippType &type, size_t &aipp_index);

  ge::Status GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type);

  ge::Status GetOpAttr(uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                       std::string &attr_value);

  ge::Status GetModelAttr(uint32_t model_id, std::vector<string> &dynamic_output_shape_info);

  ge::Status SetDynamicSize(uint32_t model_id, const std::vector<uint64_t> &batch_num, int32_t dynamic_type);

  ///
  /// @ingroup domi_ome
  /// @brief Get model according to given id
  ///
  std::shared_ptr<DavinciModel> GetModel(uint32_t id);

  std::shared_ptr<hybrid::HybridDavinciModel> GetHybridModel(uint32_t id);

  ge::Status KernelLaunchEx(aicpu::FWKAdapter::FWKOperateType op_type, uint64_t session_id, uint32_t model_id,
                            uint32_t sub_model_id);

  ge::Status CreateAicpuSession(uint64_t session_id);

  static ge::Status GetModelMemAndWeightSize(const ModelData &model, size_t &mem_size, size_t &weight_size);

  void DestroyAicpuSession(uint64_t session_id);

  ge::Status DestroyAicpuKernel(uint64_t session_id, uint32_t model_id, uint32_t sub_model_id);

  ge::Status CreateAicpuKernel(uint64_t session_id, uint32_t model_id, uint32_t sub_model_id, uint64_t kernel_id);

  ge::Status DestroyAicpuSessionForInfer(uint32_t model_id);

  ge::Status LoadCustAicpuSo(const OpDescPtr &op_desc, const string &so_name, bool &loaded);

  ge::Status LaunchCustAicpuSo();

  ge::Status ClearAicpuSo();

  ge::Status LaunchKernelCustAicpuSo(const string &kernel_name);

  ge::Status LaunchKernelCheckAicpuOp(std::vector<std::string> &aicpu_optype_list,
                                      std::vector<std::string> &aicpu_tf_optype_list);

  ge::Status CheckAicpuOpList(GeModelPtr ge_model);

  ge::Status GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &orig_input_info);

  ge::Status GenSessionId(uint64_t &session_id);

  ge::Status GetAllAippInputOutputDims(uint32_t model_id, uint32_t index, std::vector<InputOutputDims> &input_dims,
                                       std::vector<InputOutputDims> &output_dims);

  bool IsDynamicShape(uint32_t model_id);
  bool IsNeedHybridLoad(ge::GeRootModel &ge_root_model);
  ge::Status GetOpDescInfo(uint32_t device_id, uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info);

  ge::Status EnableExceptionDump(const std::map<string, string> &options);

  const std::vector<rtExceptionInfo> &GetExceptionInfos() { return exception_infos_; }

  void AddExceptionInfo(const rtExceptionInfo &rt_exception_info) { exception_infos_.emplace_back(rt_exception_info); }

  static void ExceptionCallback(rtExceptionInfo *rt_exception_info) {
    std::lock_guard<std::mutex> lock(exeception_infos_mutex_);
    auto instance = ModelManager::GetInstance();
    if (instance == nullptr) {
      GELOGE(FAILED, "[Get][Instance] failed, as ret is nullptr");
      return;
    }
    instance->AddExceptionInfo(*rt_exception_info);
  }

  bool IsDumpExceptionOpen() { return dump_exception_flag_; }
 private:
  ///
  /// @ingroup domi_ome
  /// @brief constructor
  ///
  ModelManager();

  ///
  /// @ingroup domi_ome
  /// @brief destructor
  ///
  ~ModelManager();

  ///
  /// @ingroup domi_ome
  /// @brief insert new model into model manager set
  ///
  void InsertModel(uint32_t model_id, std::shared_ptr<DavinciModel> &davinci_model);
  void InsertModel(uint32_t model_id, std::shared_ptr<hybrid::HybridDavinciModel> &hybrid_model);

  ///
  /// @ingroup domi_ome
  /// @brief delete model from model manager set
  ///
  ge::Status DeleteModel(uint32_t id);

  void GenModelId(uint32_t *id);

  std::map<uint32_t, std::shared_ptr<DavinciModel>> model_map_;
  std::map<uint32_t, std::shared_ptr<hybrid::HybridDavinciModel>> hybrid_model_map_;
  std::map<std::string, std::vector<uint64_t>> model_aicpu_kernel_;
  uint32_t max_model_id_;
  std::recursive_mutex map_mutex_;
  std::mutex session_id_create_mutex_;
  static::std::mutex exeception_infos_mutex_;
  uint64_t session_id_bias_;
  std::set<uint64_t> sess_ids_;
  std::vector<rtExceptionInfo> exception_infos_;
  std::mutex cust_aicpu_mutex_;
  std::map<uintptr_t, std::map<std::string, CustAICPUKernelPtr>> cust_aicpu_so_;

  static DumpProperties dump_properties_;
  bool dump_exception_flag_ = false;
  std::map<uint64_t, bool> session_id_to_dump_server_init_flag_;
};
}  // namespace ge

#endif  // GE_GRAPH_LOAD_NEW_MODEL_MANAGER_MODEL_MANAGER_H_
