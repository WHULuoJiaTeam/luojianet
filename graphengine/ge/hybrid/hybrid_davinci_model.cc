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

#include <memory>
#include "hybrid/hybrid_davinci_model.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/node_executor/node_executor.h"
#include "graph/manager/graph_manager_utils.h"

namespace ge {
namespace hybrid {
class HybridDavinciModel::Impl {
 public:
  explicit Impl(GeRootModelPtr ge_model) : model_(std::move(ge_model)), executor_(&model_) {
  }

  ~Impl() {
    NodeExecutorManager::GetInstance().FinalizeExecutors();
  }

  Status Init() {
    GE_CHK_STATUS_RET(NodeExecutorManager::GetInstance().EnsureInitialized(),
                      "[Initialize][NodeExecutorManager] failed");
    GE_CHK_STATUS_RET(model_.Init(), "[Init][HybridModel] failed.")
    GE_CHK_STATUS_RET(executor_.Init(), "[Init][HybridModelAsyncExecutor] failed.")
    return SUCCESS;
  }

  Status Execute(const std::vector<DataBuffer> &inputs,
                 const std::vector<GeTensorDesc> &input_desc,
                 std::vector<DataBuffer> &outputs,
                 std::vector<GeTensorDesc> &output_desc,
                 rtStream_t stream) {
    return executor_.Execute(inputs, input_desc, outputs, output_desc);
  }

  Status Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs) {
    return executor_.Execute(inputs, outputs);
  }

  Status ModelRunStart() {
    return executor_.Start(listener_);
  }

  Status ModelRunStop() {
    return executor_.Stop();
  }

  Status EnqueueData(const std::shared_ptr<InputDataWrapper> &data) {
    return executor_.EnqueueData(data);
  }

  void SetListener(const shared_ptr<ModelListener> &listener) {
    listener_ = listener;
  }

  void SetModelId(uint32_t model_id) {
    executor_.SetModelId(model_id);
    model_.SetModelId(model_id);
  }

  void SetDeviceId(uint32_t device_id) {
    model_.SetDeviceId(device_id);
    executor_.SetDeviceId(device_id);
  }

  void SetOmName(const string &model_name) {
    model_.SetOmName(model_name);
  }

  uint32_t GetDeviceId() {
    return model_.GetDeviceId();
  }

  const GraphExecutionContext *GeContext() { return executor_.GeContext(); }

  uint64_t GetSessionId() {
    return model_.GetSessionId();
  }

  Status GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) {
    return model_.GetDynamicBatchInfo(batch_info, dynamic_type);
  }

  void GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) {
    model_.GetUserDesignateShapeOrder(user_input_shape_order);
  }

  void GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) {
    model_.GetModelAttr(dynamic_output_shape_info);
  }

  Status GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                vector<InputOutputDescInfo> &output_desc,
                                std::vector<uint32_t> &input_formats,
                                std::vector<uint32_t> &output_formats) {
    return model_.GetInputOutputDescInfo(input_desc, output_desc, input_formats, output_formats);
  }

  void SetModelDescVersion(bool is_new_model_desc) {
    model_.SetModelDescVersion(is_new_model_desc);
  }

  uint32_t GetDataInputerSize() { return executor_.GetDataInputerSize(); }

  bool GetRunningFlag() const { return executor_.GetRunningFlag(); }

  Status SetRunAsyncListenerCallback(const RunAsyncCallback &callback) {
    auto listener = dynamic_cast<RunAsyncListener *>(listener_.get());
    GE_CHECK_NOTNULL(listener);
    listener->SetCallback(callback);
    return SUCCESS;
  }

  Status GetOpAttr(const std::string &op_name, const std::string &attr_name,
                                       std::string &attr_value) {
    return model_.GetOpAttr(op_name, attr_name, attr_value);
  }

 private:
  std::shared_ptr<ModelListener> listener_;
  HybridModel model_;
  HybridModelAsyncExecutor executor_;
};

HybridDavinciModel::~HybridDavinciModel() {
  delete impl_;
}

std::unique_ptr<HybridDavinciModel> HybridDavinciModel::Create(const GeRootModelPtr &ge_root_model) {
  auto instance = std::unique_ptr<HybridDavinciModel>(new (std::nothrow)HybridDavinciModel());
  if (instance != nullptr) {
    instance->impl_ = new (std::nothrow) HybridDavinciModel::Impl(ge_root_model);
    if (instance->impl_ != nullptr) {
      return instance;
    }
  }

  return nullptr;
}

Status HybridDavinciModel::Init() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Init();
}

Status HybridDavinciModel::Execute(const std::vector<DataBuffer> &inputs,
                                   const std::vector<GeTensorDesc> &input_desc,
                                   std::vector<DataBuffer> &outputs,
                                   std::vector<GeTensorDesc> &output_desc, rtStream_t stream) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Execute(inputs, input_desc, outputs, output_desc, stream);
}

Status HybridDavinciModel::Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Execute(inputs, outputs);
}

Status HybridDavinciModel::ModelRunStart() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ModelRunStart();
}

Status HybridDavinciModel::ModelRunStop() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->ModelRunStop();
}

Status HybridDavinciModel::EnqueueData(const shared_ptr<InputDataWrapper> &data) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->EnqueueData(data);
}

void HybridDavinciModel::SetListener(const shared_ptr<ModelListener> &listener) {
  if (impl_ != nullptr) {
    impl_->SetListener(listener);
  }
}

void HybridDavinciModel::SetModelId(uint32_t model_id) {
  if (impl_ != nullptr) {
    impl_->SetModelId(model_id);
  }
}

void HybridDavinciModel::SetDeviceId(uint32_t device_id) {
  if (impl_ != nullptr) {
    impl_->SetDeviceId(device_id);
  }
}

void HybridDavinciModel::SetOmName(const string &om_name) {
  if (impl_ != nullptr) {
    impl_->SetOmName(om_name);
  }
}

uint32_t HybridDavinciModel::GetDeviceId() const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetDeviceId();
}

Status HybridDavinciModel::GetDynamicBatchInfo(std::vector<std::vector<int64_t>> &batch_info, int32_t &dynamic_type) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetDynamicBatchInfo(batch_info, dynamic_type);
}

void HybridDavinciModel::GetUserDesignateShapeOrder(std::vector<std::string> &user_input_shape_order) {
  if (impl_ != nullptr) {
    impl_->GetUserDesignateShapeOrder(user_input_shape_order);
  }
}

void HybridDavinciModel::GetModelAttr(std::vector<std::string> &dynamic_output_shape_info) {
  if (impl_ != nullptr) {
    impl_->GetModelAttr(dynamic_output_shape_info);
  }
}

Status HybridDavinciModel::GetInputOutputDescInfo(vector<InputOutputDescInfo> &input_desc,
                                                  vector<InputOutputDescInfo> &output_desc,
                                                  std::vector<uint32_t> &input_formats,
                                                  std::vector<uint32_t> &output_formats) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetInputOutputDescInfo(input_desc, output_desc, input_formats, output_formats);
}

void HybridDavinciModel::SetModelDescVersion(bool is_new_model_desc) {
  if (impl_ != nullptr) {
    impl_->SetModelDescVersion(is_new_model_desc);
  }
}

uint64_t HybridDavinciModel::GetSessionId() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetSessionId();
}

uint32_t HybridDavinciModel::GetDataInputerSize() {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetDataInputerSize();
}

bool HybridDavinciModel::GetRunningFlag() const { return impl_->GetRunningFlag(); }

Status HybridDavinciModel::SetRunAsyncListenerCallback(const RunAsyncCallback &callback) {
  return impl_->SetRunAsyncListenerCallback(callback);
}

bool HybridDavinciModel::GetOpDescInfo(uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info) const {
  if (impl_ == nullptr) {
    return false;
  }
  auto context = impl_->GeContext();
  GE_CHECK_NOTNULL(context);
  bool ret = context->exception_dumper.GetOpDescInfo(stream_id, task_id, op_desc_info);
  if (!ret) {
    for (const auto &iter : context->davinci_model) {
      if (iter->GetOpDescInfo(stream_id, task_id, op_desc_info)) {
        return true;
      }
    }
  }
  return ret;
}

Status HybridDavinciModel::GetOpAttr(const std::string &op_name, const std::string &attr_name,
                                     std::string &attr_value) const {
  GE_CHECK_NOTNULL(impl_);
  return impl_->GetOpAttr(op_name, attr_name, attr_value);
}
}  // namespace hybrid
}  // namespace ge
