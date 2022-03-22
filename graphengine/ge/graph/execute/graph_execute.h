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

#ifndef GE_GRAPH_EXECUTE_GRAPH_EXECUTE_H_
#define GE_GRAPH_EXECUTE_GRAPH_EXECUTE_H_

#include <cstdarg>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "framework/common/ge_types.h"
#include "common/properties_manager.h"
#include "framework/common/string_util.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "external/ge/ge_api_types.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_context.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/model.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/load/model_manager/davinci_model.h"

namespace ge {
class GraphExecutor {
 public:
  GraphExecutor();

  virtual ~GraphExecutor();

  Status ExecuteGraph(GraphId graph_id, const GeRootModelPtr &ge_root_model, const std::vector<GeTensor> &input_tensor,
                      std::vector<GeTensor> &output_tensor);

  ge::Status ExecuteGraphAsync(GraphId graph_id, const GeRootModelPtr &ge_root_model,
                               const std::vector<ge::Tensor> &input_tensor, const RunAsyncCallback &callback);

  Status ExecuteGraphWithStream(GraphId graph_id,
                                rtStream_t stream,
                                const GeRootModelPtr &ge_root_model,
                                const std::vector<GeTensor> &input_tensor,
                                std::vector<GeTensor> &output_tensor);

  Status SetCondition(std::mutex *mutex, std::condition_variable *cond, std::shared_ptr<GraphModelListener> listener);

  static Status SetDynamicSize(uint32_t model_id, const std::vector<uint64_t> &batch_num, int32_t dynamic_type);

  void SetTrainFlag(bool is_train_graph);

  const std::vector<InputOutputDescInfo> &GetOutputsDesc() const { return outputs_desc_; }

  Status FreeExecuteMemory();

  static Status DataInput(const InputData &input_data, OutputData &output_data);

  static Status GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                       vector<InputOutputDescInfo> &output_desc);

  static Status GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                       vector<InputOutputDescInfo> &output_desc, std::vector<uint32_t> &input_formats,
                                       std::vector<uint32_t> &output_formats, bool new_model_desc = false);

  static Status GetAippInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_info);

  static Status GetAippType(uint32_t model_id, uint32_t index, InputAippType &type, size_t &aipp_index);

  ///
  /// @ingroup ge
  /// @brief Get dynamic batch_info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @param [out] dynamic_type
  /// @return execute result
  ///
  static Status GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                    int32_t &dynamic_type);

  ///
  /// @ingroup ge
  /// @brief Get combined dynamic dims info
  /// @param [in] model_id
  /// @param [out] batch_info
  /// @return execute result
  ///
  static Status GetCombinedDynamicDims(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info);

  ///
  /// @ingroup ge
  /// @brief Get user designate shape order
  /// @param [in] model_id
  /// @param [out] user_input_shape_order
  /// @return execute result
  ///
  static Status GetUserDesignateShapeOrder(uint32_t model_id, std::vector<std::string> &user_input_shape_order);

  static Status GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type);

  static Status GetOpAttr(uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                          std::string &attr_value);

  static Status GetModelAttr(uint32_t model_id, std::vector<string> &dynamic_output_shape_info);

  static Status GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &orig_input_info);
  static Status GetAllAippInputOutputDims(uint32_t model_id, uint32_t index, std::vector<InputOutputDims> &input_dims,
                                          std::vector<InputOutputDims> &output_dims);

  static Status GetOpDescInfo(uint32_t device_id, uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info);

  uint32_t GetExecuteModelId(const GeRootModelPtr &ge_root_model);

 private:
  Status PrepareInputData(const std::vector<GeTensor> &input_tensor, InputData &graph_input_data,
                          OutputData &graph_output_data, std::vector<InputOutputDescInfo> &output_desc);

  Status GetExecuteData(const std::vector<GeTensor> &input_tensor, std::vector<DataBuffer> &blobs,
                        std::vector<GeTensorDesc> &tensor_desc);

  Status SyncExecuteModel(uint32_t model_id, const std::vector<GeTensor> &input_tensor,
                          std::vector<GeTensor> &output_tensor);

  Status AsyncExecuteModel(const GeRootModelPtr &ge_root_model, const std::vector<ge::Tensor> &input_tensor,
                           const RunAsyncCallback &callback);

  void InitModelIdInfo(std::vector<uint32_t> &out_model_id_info, std::vector<SubGraphInfoPtr> &sub_graph_vec,
                       uint32_t output_size);

  Status FreeInOutBuffer();

  Status MallocInOutBuffer(const std::vector<uint64_t> &buffer_size, std::vector<void *> &data_addr);

  static Status SetCallback(uint32_t model_id, const GeRootModelPtr &ge_root_model,
                            const RunAsyncCallback &callback);

  Status ModelSubscribe(uint32_t graph_id);

  Status GetModelByID(uint32_t model_id, std::shared_ptr<DavinciModel> &davinci_model);

  bool init_flag_;

  bool train_graph_flag_;
  // For run graph synchronous return
  std::mutex *sync_run_mutex_;
  std::condition_variable *condition_;

  // Run graph asynchronous call back listener
  std::shared_ptr<GraphModelListener> graph_run_listener_;

  std::vector<InputOutputDescInfo> outputs_desc_;
  GraphId last_graph_id_;

  bool malloc_flag_;
  std::vector<void *> buffer_addr_;
  std::vector<uint64_t> buffer_size_;
};
}  // namespace ge

#endif  // GE_GRAPH_EXECUTE_GRAPH_EXECUTE_H_
