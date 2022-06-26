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

#include "src/cxx_api/model/model_impl.h"
#include <memory>
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/lite_session.h"
#include "include/context.h"
#include "src/runtime/inner_allocator.h"
#include "src/cxx_api/converters.h"
#include "src/cxx_api/graph/graph_data.h"
#include "src/cxx_api/tensor/tensor_impl.h"
#include "src/cxx_api/tensor_utils.h"
#include "src/common/log_adapter.h"
#include "src/lite_session.h"
#include "src/common/file_utils.h"
#include "src/common/config_file.h"
#include "src/cpu_info.h"
#ifdef SERVER_INFERENCE
#include "src/common/common.h"
#endif

namespace luojianet_ms {
namespace {
const char *const kExecutionPlan = "execution_plan";
constexpr size_t kMaxSectionNum = 100;
constexpr size_t kMaxConfigNumPerSection = 1000;
}  // namespace
using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_OK;

CreateTrainSessionProto *CreateTrainSessionCallbackHolder(CreateTrainSessionProto *proto) {
  static CreateTrainSessionProto *proto_ = nullptr;
  if (proto != nullptr) {
    proto_ = proto;
  }
  return proto_;
}

Status ModelImpl::Build(const void *model_data, size_t data_size, ModelType model_type,
                        const std::shared_ptr<Context> &ms_context) {
  if (model_data == nullptr) {
    MS_LOG(ERROR) << "The input model buffer is nullptr.";
    return kLiteNullptr;
  }
  if (data_size == 0) {
    MS_LOG(ERROR) << "The input model buffer size is 0.";
    return kLiteInputParamInvalid;
  }
  if (!PlatformInstructionSetSupportCheck()) {
    MS_LOG(ERROR) << "The platform exist don't support's instruction.";
    return kLiteNotSupport;
  }
  context_ = ms_context;
  auto session = std::shared_ptr<lite::LiteSession>(CreateLiteSession(ContextUtils::Convert(ms_context.get())));
  if (session == nullptr) {
    MS_LOG(ERROR) << "Allocate session failed.";
    return kLiteNullptr;
  }

  auto ret =
    session->LoadModelAndCompileByBuf(static_cast<const char *>(model_data), model_type, data_size, ms_context);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init session failed";
    return kLiteError;
  }

  session_.swap(session);
  MS_LOG(DEBUG) << "Build model success.";
  return kSuccess;
}

Status ModelImpl::Build(const std::string &model_path, ModelType model_type,
                        const std::shared_ptr<Context> &ms_context) {
  if (!PlatformInstructionSetSupportCheck()) {
    MS_LOG(ERROR) << "The platform exist don't support's instruction.";
    return kLiteNotSupport;
  }

  auto session = std::shared_ptr<lite::LiteSession>(CreateLiteSession(ContextUtils::Convert(ms_context.get())));
  if (session == nullptr) {
    MS_LOG(ERROR) << "Allocate session failed.";
    return kLiteNullptr;
  }

  auto ret = session->LoadModelAndCompileByPath(model_path, model_type, ms_context);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init session failed";
    return kLiteError;
  }

  session_.swap(session);
  MS_LOG(DEBUG) << "Build model success.";
  return kSuccess;
}

Status ModelImpl::Build() {
  MS_LOG(DEBUG) << "Start build model.";
  if (graph_ == nullptr || graph_->graph_data_ == nullptr) {
    MS_LOG(ERROR) << "Invalid graph.";
    return kLiteNullptr;
  }

  if (context_ == nullptr) {
    MS_LOG(ERROR) << "Invalid context.";
    return kLiteNullptr;
  }

  if (!PlatformInstructionSetSupportCheck()) {
    MS_LOG(ERROR) << "The platform exist don't support's instruction.";
    return kLiteNotSupport;
  }

  auto *inner_context = ContextUtils::Convert(context_.get());
  if (inner_context == nullptr) {
    MS_LOG(ERROR) << "Failed to convert Context to Lite Context";
    return kLiteNullptr;
  }

  auto create_callback = CreateTrainSessionCallbackHolder();
  if (create_callback != nullptr) {
    auto session = create_callback(graph_->graph_data_, cfg_, inner_context);
    if (session != nullptr) {
      session_ = session;
      MS_LOG(DEBUG) << "Build model success.";
      return kSuccess;
    }
  }

  auto model = graph_->graph_data_->lite_model();
  if (model == nullptr || model->buf == nullptr) {
    delete inner_context;
    MS_LOG(ERROR) << "Lite model has been freed.";
    return kLiteError;
  }

  auto session = std::shared_ptr<lite::LiteSession>(CreateLiteSession(inner_context));
  if (session == nullptr) {
    MS_LOG(ERROR) << "Allocate session failed.";
    return kLiteNullptr;
  }
  auto ret = session->CompileGraph(model.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Build model failed.";
    return static_cast<StatusCode>(ret);
  }
  session_.swap(session);
  model->Free();
  MS_LOG(DEBUG) << "Build model success.";
  return kSuccess;
}

static void ResetTensorData(std::vector<void *> old_data, const std::vector<tensor::MSTensor *> &tensors) {
  for (size_t j = 0; j < old_data.size(); j++) {
    tensors.at(j)->set_data(old_data.at(j));
  }
}

Status ModelImpl::RunGraph(const MSKernelCallBack &before, const MSKernelCallBack &after) {
  KernelCallBack before_call_back = nullptr;
  KernelCallBack after_call_back = nullptr;
  if (before != nullptr) {
    before_call_back = [&](const std::vector<luojianet_ms::tensor::MSTensor *> &before_inputs,
                           const std::vector<luojianet_ms::tensor::MSTensor *> &before_outputs,
                           const CallBackParam &call_param) {
      std::vector<MSTensor> inputs = LiteTensorsToMSTensors(before_inputs);
      std::vector<MSTensor> outputs = LiteTensorsToMSTensors(before_outputs);
      MSCallBackParam mscall_param;
      mscall_param.node_name = call_param.node_name;
      mscall_param.node_type = call_param.node_type;
      return before(inputs, outputs, mscall_param);
    };
  }

  if (after != nullptr) {
    after_call_back = [&](const std::vector<luojianet_ms::tensor::MSTensor *> &before_inputs,
                          const std::vector<luojianet_ms::tensor::MSTensor *> &before_outputs,
                          const CallBackParam &call_param) {
      std::vector<MSTensor> inputs = LiteTensorsToMSTensors(before_inputs);
      std::vector<MSTensor> outputs = LiteTensorsToMSTensors(before_outputs);
      MSCallBackParam mscall_param;
      mscall_param.node_name = call_param.node_name;
      mscall_param.node_type = call_param.node_type;
      return after(inputs, outputs, mscall_param);
    };
  }
  auto ret = session_->RunGraph(before_call_back, after_call_back);
  return static_cast<StatusCode>(ret);
}

bool ModelImpl::IsTrainModel() { return (graph_ && graph_->graph_data_ && graph_->graph_data_->IsTrainModel()); }

Status ModelImpl::LoadConfig(const std::string &config_path) {
  std::map<std::string, std::map<std::string, std::string>> all_config_info;
  int ret = lite::GetAllSectionInfoFromConfigFile(config_path, &all_config_info);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "GetAllSectionInfoFromConfigFile fail!ret: " << ret;
    return kLiteFileError;
  }
  config_info_ = all_config_info;
  std::map<std::string, std::string> config_info = all_config_info[kExecutionPlan];
  if (config_info.empty()) {
    MS_LOG(WARNING) << "No valid execution plan info in config file.";
    return kSuccess;
  }

  lite::ParserExecutionPlan(&config_info, &execution_plan_);
  return kSuccess;
}

Status ModelImpl::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    if (config_info_.size() >= kMaxSectionNum) {
      MS_LOG(ERROR) << "config too many sections!";
      return kLiteError;
    }
    config_info_[section][config.first] = config.second;
    return kSuccess;
  }
  if (iter->second.size() >= kMaxConfigNumPerSection) {
    MS_LOG(ERROR) << "config too many items!";
    return kLiteError;
  }
  iter->second[config.first] = config.second;
  return kSuccess;
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  if (outputs == nullptr) {
    MS_LOG(ERROR) << "outputs is nullptr.";
    return kLiteError;
  }
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Run graph failed.";
    return kLiteError;
  }
  auto input_tensors = session_->GetInputs();
  if (input_tensors.empty()) {
    MS_LOG(ERROR) << "Failed to get input tensor.";
    return kLiteError;
  }
  if (input_tensors.size() != inputs.size()) {
    MS_LOG(ERROR) << "Wrong input size.";
    return kLiteError;
  }
  std::vector<void *> old_data;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = input_tensors.at(i);
    auto user_input = inputs.at(i);
    if (user_input.DataType() != static_cast<enum DataType>(input->data_type())) {
      ResetTensorData(old_data, input_tensors);
      MS_LOG(ERROR) << "Tensor " << user_input.Name() << " has a different data type from input" << input->tensor_name()
                    << ".";
      return kLiteInputTensorError;
    }
    if (user_input.Data() == nullptr) {
      ResetTensorData(old_data, input_tensors);
      MS_LOG(ERROR) << "Tensor " << user_input.Name() << " has no data.";
      return kLiteInputTensorError;
    }
    if (user_input.Name() != input->tensor_name() && user_input.Name() != "MindDataTensor") {
      MS_LOG(WARNING) << "Tensor " << user_input.Name() << " has a different name from input" << input->tensor_name()
                      << ".";
    }
    old_data.push_back(input->data());
    if (input->data_type() == kObjectTypeString) {
#ifndef STRING_KERNEL_CLIP
      std::vector<int32_t> shape = TruncateShape(user_input.Shape(), input->data_type(), user_input.DataSize(), false);
      if (shape.empty() && !(user_input.Shape().empty())) {
        ResetTensorData(old_data, input_tensors);
        MS_LOG(ERROR) << "Input dims of tensor " << user_input.Name() << " is invalid.";
        return kLiteParamInvalid;
      }
      input->set_shape(shape);
      input->set_data(user_input.MutableData());
#else
      MS_LOG(ERROR) << unsupport_string_tensor_log;
      return kLiteError;
#endif
    } else {
      if (user_input.MutableData() != input->data()) {
        if (input->Size() != user_input.DataSize()) {
          ResetTensorData(old_data, input_tensors);
#ifndef ENABLE_LITE_ACL
          MS_LOG(ERROR) << "Tensor " << user_input.Name() << " has wrong data size.";
          return kLiteInputTensorError;
#else
          MS_LOG(WARNING) << "Please check tensor " << user_input.Name()
                          << " has been modified data size by DVPP method.";
          std::vector<int> truncate_shape = {static_cast<int>(user_input.DataSize())};
          input->set_shape(truncate_shape);
#endif
        }
        input->set_data(user_input.MutableData());
      }
    }
  }
  auto ret = RunGraph(before, after);
  ResetTensorData(old_data, input_tensors);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run graph failed.";
    return ret;
  }
  MS_LOG(DEBUG) << "Run graph success.";
  auto res = GetOutputs();
  if (res.empty()) {
    MS_LOG(DEBUG) << "Empty outputs.";
    return kLiteError;
  }
  outputs->clear();
  outputs->insert(outputs->end(), res.begin(), res.end());
  return kSuccess;
}

std::vector<MSTensor> ModelImpl::GetInputs() {
  std::vector<MSTensor> empty;
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  std::vector<MSTensor> res;
  auto inputs = session_->GetInputs();
  if (inputs.empty()) {
    MS_LOG(ERROR) << "The inputs of model is null.";
    return empty;
  }
  res.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto impl = std::make_shared<LiteTensorImpl>(inputs[i]);
    if (impl == nullptr || impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    auto tensor = MSTensor(impl);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    res[i] = tensor;
  }
  return res;
}

std::vector<MSTensor> ModelImpl::GetOutputs() {
  std::vector<MSTensor> empty;
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  std::vector<MSTensor> res;
  auto names = session_->GetOutputTensorNames();
  if (names.empty()) {
    MS_LOG(ERROR) << "The output tensor name of this model is null.";
    return empty;
  }
  auto outputs = session_->GetOutputs();
  if (outputs.empty()) {
    MS_LOG(ERROR) << "The outputs of model is null.";
    return empty;
  }
  if (names.size() != outputs.size()) {
    MS_LOG(ERROR) << "The size of outputs dose not match the size of names.";
    return empty;
  }
  res.resize(names.size());
  for (size_t i = 0; i < names.size(); i++) {
    auto impl = std::make_shared<LiteTensorImpl>(outputs[names[i]]);
    if (impl == nullptr || impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    auto tensor = MSTensor(impl);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    res[i] = tensor;
  }
  return res;
}

std::vector<MSTensor> ModelImpl::GetGradients() const {
  std::vector<MSTensor> empty;
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  auto params = session_->GetGradients();
  if (params.empty()) {
    MS_LOG(ERROR) << "No optimizer parameters avelibale.";
    return empty;
  }
  std::vector<MSTensor> res = LiteTensorsToMSTensors(params, false);
  return res;
}

Status ModelImpl::ApplyGradients(const std::vector<MSTensor> &gradients) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteNullptr;
  }
  if (gradients.empty()) {
    MS_LOG(ERROR) << "gradients is null.";
    return kLiteInputParamInvalid;
  }
  std::vector<tensor::MSTensor *> inner_gradients;
  inner_gradients.resize(gradients.size());
  for (size_t i = 0; i < gradients.size(); i++) {
    auto gradient = gradients[i];
    if (gradient.impl_ == nullptr) {
      MS_LOG(ERROR) << "gradient tensor " << gradient.Name() << " is null.";
      return kLiteInputTensorError;
    }
    auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(gradient.impl_);
    if (lite_impl == nullptr || lite_impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "gradient tensor " << gradient.Name() << " is null.";
      return kLiteInputTensorError;
    }
    inner_gradients[i] = lite_impl->lite_tensor();
  }
  auto ret = session_->ApplyGradients(inner_gradients);
  return static_cast<StatusCode>(ret);
}

std::vector<MSTensor> ModelImpl::GetFeatureMaps() const {
  std::vector<MSTensor> empty;
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  auto params = session_->GetFeatureMaps();
  if (params.empty()) {
    MS_LOG(ERROR) << "No optimizer parameters avelibale.";
    return empty;
  }
  std::vector<MSTensor> res = LiteTensorsToMSTensors(params, false);
  return res;
}

Status ModelImpl::UpdateFeatureMaps(const std::vector<MSTensor> &new_weights) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteNullptr;
  }
  if (new_weights.empty()) {
    MS_LOG(ERROR) << "gradients is null.";
    return kLiteInputParamInvalid;
  }
  std::vector<tensor::MSTensor *> inner_weights;
  inner_weights.resize(new_weights.size());
  for (size_t i = 0; i < new_weights.size(); i++) {
    auto new_weight = new_weights[i];
    if (new_weight.impl_ == nullptr) {
      MS_LOG(ERROR) << "weight tensor " << new_weight.Name() << " is null.";
      return kLiteInputTensorError;
    }
    auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(new_weight.impl_);
    if (lite_impl == nullptr || lite_impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "weight tensor " << new_weight.Name() << " is null.";
      return kLiteInputTensorError;
    }
    inner_weights[i] = lite_impl->lite_tensor();
  }
  auto ret = session_->UpdateFeatureMaps(inner_weights);
  return static_cast<StatusCode>(ret);
}

std::vector<MSTensor> ModelImpl::GetOptimizerParams() const {
  std::vector<MSTensor> empty;
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  auto params = session_->GetOptimizerParams();
  if (params.empty()) {
    MS_LOG(ERROR) << "No optimizer parameters avelibale.";
    return empty;
  }
  std::vector<MSTensor> res = LiteTensorsToMSTensors(params);
  return res;
}

Status ModelImpl::SetOptimizerParams(const std::vector<MSTensor> &params) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteNullptr;
  }
  if (params.empty()) {
    MS_LOG(ERROR) << "params is null.";
    return kLiteInputParamInvalid;
  }
  std::vector<tensor::MSTensor *> inner_params;
  inner_params.resize(params.size());
  for (size_t i = 0; i < params.size(); i++) {
    auto param = params[i];
    if (param.impl_ == nullptr) {
      MS_LOG(ERROR) << "Param tensor " << param.Name() << " is null.";
      return kLiteInputTensorError;
    }
    auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(param.impl_);
    if (lite_impl == nullptr || lite_impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Param tensor " << param.Name() << " is null.";
      return kLiteInputTensorError;
    }
    inner_params[i] = lite_impl->lite_tensor();
  }
  auto ret = session_->SetOptimizerParams(inner_params);
  return static_cast<StatusCode>(ret);
}

MSTensor ModelImpl::GetInputByTensorName(const std::string &name) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return MSTensor(nullptr);
  }
  auto res = session_->GetInputsByTensorName(name);
  if (res == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  auto impl = std::make_shared<LiteTensorImpl>(res);
  if (impl == nullptr || impl->lite_tensor() == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed.";
    return MSTensor(nullptr);
  }

  return MSTensor(impl);
}

std::vector<std::string> ModelImpl::GetOutputTensorNames() {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    std::vector<std::string> empty;
    return empty;
  }
  return session_->GetOutputTensorNames();
}

MSTensor ModelImpl::GetOutputByTensorName(const std::string &name) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return MSTensor(nullptr);
  }
  auto res = session_->GetOutputByTensorName(name);
  if (res == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  auto impl = std::make_shared<LiteTensorImpl>(res);
  if (impl == nullptr || impl->lite_tensor() == nullptr) {
    MS_LOG(ERROR) << "Create tensor failed.";
    return MSTensor(nullptr);
  }

  return MSTensor(impl);
}

std::vector<MSTensor> ModelImpl::GetOutputsByNodeName(const std::string &name) {
  std::vector<MSTensor> empty;
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return empty;
  }
  std::vector<MSTensor> res;
  auto outputs = session_->GetOutputsByNodeName(name);
  if (outputs.empty()) {
    MS_LOG(ERROR) << "The outputs of model is null.";
    return empty;
  }
  res.resize(outputs.size());
  for (size_t i = 0; i < outputs.size(); i++) {
    auto impl = std::make_shared<LiteTensorImpl>(outputs[i]);
    if (impl == nullptr || impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    auto tensor = MSTensor(impl);
    if (tensor == nullptr) {
      MS_LOG(ERROR) << "Create tensor failed.";
      return empty;
    }
    res[i] = tensor;
  }
  return res;
}

#ifdef ENABLE_OPENGL_TEXTURE
Status ModelImpl::BindGLTexture2DMemory(const std::map<std::string, GLuint> &inputGLTexture,
                                        std::map<std::string, GLuint> *outputGLTexture) {
  MS_LOG(INFO) << "Bind GLTexture2D to Input MsTensors and Output MsTensors";
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteError;
  }
  auto status = session_->BindGLTexture2DMemory(inputGLTexture, outputGLTexture);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Bing OpenGL Texture to OpenCl Memory failed";
    return kLiteError;
  }
  return kSuccess;
}
#endif

Status ModelImpl::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteNullptr;
  }
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Inputs is null.";
    return kLiteInputParamInvalid;
  }
  if (dims.empty()) {
    MS_LOG(ERROR) << "Dims is null.";
    return kLiteInputParamInvalid;
  }
  if (inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "The size of inputs does not match the size of dims.";
    return kLiteInputParamInvalid;
  }
  auto model_inputs = session_->GetInputs();
  if (model_inputs.empty()) {
    MS_LOG(ERROR) << "The inputs of model is null.";
    return kLiteParamInvalid;
  }
  if (inputs.size() != model_inputs.size()) {
    MS_LOG(ERROR) << "The size of inputs is incorrect.";
    return kLiteInputParamInvalid;
  }
  std::vector<tensor::MSTensor *> inner_input;
  inner_input.resize(inputs.size());
  std::vector<std::vector<int32_t>> truncated_shape;
  truncated_shape.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    if (input.impl_ == nullptr) {
      MS_LOG(ERROR) << "Input tensor " << input.Name() << " is null.";
      return kLiteInputTensorError;
    }
    auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(input.impl_);
    if (lite_impl == nullptr || lite_impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Input tensor " << input.Name() << " is null.";
      return kLiteInputTensorError;
    }
    inner_input[i] = lite_impl->lite_tensor();
    std::vector<int32_t> shape = TruncateShape(dims[i], inner_input[i]->data_type(), inner_input[i]->Size(), false);
    if (shape.empty() && !(dims[i].empty())) {
      MS_LOG(ERROR) << "Input dims[" << i << "] is invalid.";
      return kLiteParamInvalid;
    }
    truncated_shape[i] = shape;
  }
  auto ret = session_->Resize(inner_input, truncated_shape);
  return static_cast<StatusCode>(ret);
}

Status ModelImpl::UpdateWeights(const std::vector<MSTensor> &new_weights) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteNullptr;
  }
  if (new_weights.empty()) {
    MS_LOG(ERROR) << "New weights are empty.";
    return kLiteInputParamInvalid;
  }
  std::vector<tensor::MSTensor *> inner_weights;
  inner_weights.resize(new_weights.size());
  for (size_t i = 0; i < new_weights.size(); i++) {
    auto weight = new_weights[i];
    if (weight.impl_ == nullptr) {
      MS_LOG(ERROR) << "Weight tensor " << weight.Name() << " is null.";
      return kLiteInputTensorError;
    }
    auto lite_impl = std::static_pointer_cast<LiteTensorImpl>(weight.impl_);
    if (lite_impl == nullptr || lite_impl->lite_tensor() == nullptr) {
      MS_LOG(ERROR) << "Weight tensor " << weight.Name() << " is null.";
      return kLiteInputTensorError;
    }
    inner_weights[i] = lite_impl->lite_tensor();
  }
  auto ret = session_->UpdateWeights(inner_weights);
  return static_cast<StatusCode>(ret);
}

Status ModelImpl::SetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteNullptr;
  }
  auto ret = session_->SetupVirtualBatch(virtual_batch_multiplier, lr, momentum);
  return static_cast<StatusCode>(ret);
}

Status ModelImpl::SetLearningRate(float learning_rate) {
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Session is null.";
    return kLiteNullptr;
  }
  auto ret = session_->SetLearningRate(learning_rate);
  return static_cast<StatusCode>(ret);
}

float ModelImpl::GetLearningRate() {
  if (session_ == nullptr) {
    MS_LOG(WARNING) << "Session is null.";
    return 0.0;
  }
  return session_->GetLearningRate();
}

lite::LiteSession *ModelImpl::CreateLiteSession(lite::InnerContext *context) {
  auto session = new (std::nothrow) lite::LiteSession();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    delete context;
    return nullptr;
  }
#ifdef BFC_MEMORY
  auto iter = config_info_.find(lite::kConfigServerInference);
  if (iter != config_info_.end()) {
    auto numa_iter = iter->second.find(lite::kConfigNUMANodeId);
    if (numa_iter != iter->second.end()) {
      context->SetNodeId(std::atoi(numa_iter->second.c_str()));
    }
  }
#endif
  session->InitExecutionConfig(&execution_plan_);
  session->SetConfigInfo(&config_info_);

  auto ret = session->Init(context);
  if (ret != luojianet_ms::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    delete session;
    return nullptr;
  }
  return session;
}
}  // namespace luojianet_ms
