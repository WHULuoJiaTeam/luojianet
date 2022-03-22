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

#include "framework/ge_runtime/model_runner.h"
#include "ge_runtime/runtime_model.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/ge/ge_util.h"
#include "framework/ge_runtime/davinci_model.h"
#include "graph/op_desc.h"

namespace ge {
namespace model_runner {

using RuntimeModelPtr = std::shared_ptr<RuntimeModel>;
using DavinciModelPtr = std::shared_ptr<DavinciModel>;

ModelRunner &ModelRunner::Instance() {
  static ModelRunner instance;  // Guaranteed to be destroyed.
  return instance;
}

bool ModelRunner::LoadDavinciModel(uint32_t device_id, uint64_t session_id, uint32_t model_id,
                                   std::shared_ptr<DavinciModel> davinci_model,
                                   std::shared_ptr<ModelListener> listener) {
  std::shared_ptr<RuntimeModel> model = MakeShared<RuntimeModel>();
  if (model == nullptr) {
    return false;
  }
  bool status = model->Load(device_id, session_id, davinci_model);
  if (!status) {
    return false;
  }

  runtime_models_[model_id] = model;
  return true;
}

bool ModelRunner::DistributeTask(uint32_t model_id) {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    GELOGE(PARAM_INVALID, "Model id %u not found.", model_id);
    return false;
  }
  return model_iter->second->DistributeTask();
}

bool ModelRunner::LoadModelComplete(uint32_t model_id) {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    GELOGE(PARAM_INVALID, "Model id %u not found.", model_id);
    return false;
  }
  return model_iter->second->LoadComplete();
}

const std::vector<uint32_t> &ModelRunner::GetTaskIdList(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    GELOGE(PARAM_INVALID, "Model id %u not found.", model_id);
    static const std::vector<uint32_t> empty_ret;
    return empty_ret;
  }

  return model_iter->second->GetTaskIdList();
}

const std::vector<uint32_t> &ModelRunner::GetStreamIdList(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    GELOGE(PARAM_INVALID, "Model id %u not found.", model_id);
    static const std::vector<uint32_t> empty_ret;
    return empty_ret;
  }

  return model_iter->second->GetStreamIdList();
}

const std::map<std::string, std::shared_ptr<RuntimeInfo>> &ModelRunner::GetRuntimeInfoMap(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    GELOGW("Model id %u not found.", model_id);
    static const std::map<std::string, std::shared_ptr<RuntimeInfo>> empty_ret;
    return empty_ret;
  }

  return model_iter->second->GetRuntimeInfoMap();
}

void *ModelRunner::GetModelHandle(uint32_t model_id) const {
  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    GELOGW("Model id %u not found.", model_id);
    return nullptr;
  }

  return model_iter->second->GetModelHandle();
}

bool ModelRunner::UnloadModel(uint32_t model_id) {
  auto iter = runtime_models_.find(model_id);
  if (iter != runtime_models_.end()) {
    (void)runtime_models_.erase(iter);
    return true;
  }

  return false;
}

bool ModelRunner::RunModel(uint32_t model_id, const InputData &input_data, OutputData *output_data) {
  if (output_data == nullptr) {
    GELOGW("Output data point is null.");
  }

  auto model_iter = runtime_models_.find(model_id);
  if (model_iter == runtime_models_.end()) {
    GELOGE(PARAM_INVALID, "Model id %u not found.", model_id);
    return false;
  }

  bool status = model_iter->second->CopyInputData(input_data);
  if (!status) {
    GELOGE(FAILED, "Copy input data fail.");
    return false;
  }

  status = model_iter->second->Run();
  if (!status) {
    GELOGE(FAILED, "Run model fail.");
    return false;
  }

  return true;
}

bool ModelRunner::GetInputOutputDescInfo(uint32_t model_id, bool zero_copy,
                                         std::vector<InputOutputDescInfo> *input_desc,
                                         std::vector<InputOutputDescInfo> *output_desc,
                                         std::vector<uint32_t> *input_format, std::vector<uint32_t> *output_format) {
  if (runtime_models_.find(model_id) == runtime_models_.end()) {
    GELOGE(PARAM_INVALID, "Model id %u not found.", model_id);
    return false;
  }

  auto model = runtime_models_[model_id];
  if (input_desc == nullptr || output_desc == nullptr) {
    GELOGE(PARAM_INVALID, "input_desc or output_desc is null.");
    return false;
  }

  bool status = model->GetInputOutputDescInfo(zero_copy, input_desc, output_desc, input_format, output_format);
  if (!status) {
    GELOGE(FAILED, "Get input output desc info fail.");
    return false;
  }

  return true;
}
}  // namespace model_runner
}  // namespace ge
