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

#ifndef HYBRID_HYBRID_DAVINCI_MODEL_H_
#define HYBRID_HYBRID_DAVINCI_MODEL_H_

#include <memory>
#include "external/ge/ge_api_error_codes.h"
#include "graph/load/model_manager/data_inputer.h"
#include "common/model/ge_root_model.h"

namespace ge {
namespace hybrid {
class HybridDavinciModel {
 public:
  ~HybridDavinciModel();

  HybridDavinciModel(const HybridDavinciModel &) = delete;
  HybridDavinciModel(HybridDavinciModel &&) = delete;
  HybridDavinciModel &operator=(const HybridDavinciModel &) = delete;
  HybridDavinciModel &operator=(HybridDavinciModel &&) = delete;

  static std::unique_ptr<HybridDavinciModel> Create(const GeRootModelPtr &ge_root_model);

  Status Init();

  Status Execute(const std::vector<DataBuffer> &inputs,
                 const std::vector<GeTensorDesc> &input_desc,
                 std::vector<DataBuffer> &outputs,
                 std::vector<GeTensorDesc> &output_desc,
                 rtStream_t stream);

  Status Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs);

  Status ModelRunStart();

  Status ModelRunStop();

  Status EnqueueData(const std::shared_ptr<InputDataWrapper> &data);

  void SetListener(const shared_ptr<ModelListener> &listener);

  void SetModelId(uint32_t model_id);

  void SetDeviceId(uint32_t device_id);

  void SetOmName(const string &om_name);

  uint64_t GetSessionId();

  uint32_t GetDeviceId() const;

  Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type);

  void GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order);

  void GetModelAttr(std::vector<std::string> &dynamic_output_shape_info);

  Status GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &input_formats,
                                std::vector<uint32_t> &output_formats);

  void SetModelDescVersion(bool is_new_model_desc);

  uint32_t GetDataInputerSize();

  bool GetRunningFlag() const;

  Status SetRunAsyncListenerCallback(const RunAsyncCallback &callback);

  bool GetOpDescInfo(uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info) const;

  Status GetOpAttr(const std::string &op_name, const std::string &attr_name, std::string &attr_value) const;

 private:
  HybridDavinciModel() = default;
  class Impl;
  Impl *impl_ = nullptr;
};
}  // namespace hybrid
}  // namespace ge
#endif // HYBRID_HYBRID_DAVINCI_MODEL_H_
