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

#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "NvInferRuntimeCommon.h"
#include "src/delegate/tensorrt/op/activation_opt_plugin.h"
#include "src/delegate/tensorrt/cuda_impl/activation.cuh"

namespace luojianet_ms::lite {
REGISTER_TENSORRT_PLUGIN(ActivationOptPluginCreater);

int ActivationOptPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                 const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
                                 void *const *outputs, void *workspace, cudaStream_t stream) noexcept {
  return RunCudaActivation(inputDesc, inputs, outputs, stream);
}

bool ActivationOptPlugin::needResize(const int *current_dims, const int *last_dims) {
  for (int i = 0; i < infer_dims_cnt_; i++) {
    if (current_dims[i] != last_dims[i]) {
      return true;
    }
  }
  return false;
}

int ActivationOptPlugin::RunCuDNNActivation(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                            void *const *outputs, cudaStream_t stream) {
  if (needResize(infer_dims_, inputDesc[0].dims.d)) {
    if (input_desc_ != nullptr) {
      CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc_));
      input_desc_ = nullptr;
    }
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc_));
    for (int i = 0; i < inputDesc[0].dims.nbDims; i++) {
      infer_dims_[i] = inputDesc[0].dims.d[i];
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(input_desc_, ConvertCudnnDataType(inputDesc[0].type), infer_dims_cnt_,
                                           infer_dims_, infer_stride_));
  }
  CHECK_NULL_RETURN(cudnn_handle_);
  CHECK_NULL_RETURN(activation_desc_);
  CHECK_NULL_RETURN(input_desc_);
  CUDNN_CHECK(cudnnSetStream(cudnn_handle_, stream));
  auto ret = CudnnActivation(cudnn_handle_, activation_desc_, input_desc_, inputs[0], input_desc_, outputs[0]);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "cudnn activation func call failed " << layer_name_;
    return ret;
  }
  return RET_OK;
}

int ActivationOptPlugin::RunCudaActivation(const nvinfer1::PluginTensorDesc *inputDesc, const void *const *inputs,
                                           void *const *outputs, cudaStream_t stream) {
  Sigmoid(static_cast<const float *>(inputs[0]), static_cast<float *>(outputs[0]), GetDimsVolume(inputDesc[0].dims),
          stream);
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *ActivationOptPlugin::clone() const noexcept {
  auto *plugin = new ActivationOptPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

size_t ActivationOptPlugin::getSerializationSize() const noexcept { return sizeof(schema::ActivationType); }

void ActivationOptPlugin::serialize(void *buffer) const noexcept {
  SerializeValue(&buffer, &activation_type_, sizeof(schema::ActivationType));
}

nvinfer1::IPluginV2 *ActivationOptPluginCreater::createPlugin(const char *name,
                                                              const nvinfer1::PluginFieldCollection *fc) noexcept {
  const nvinfer1::PluginField *fields = fc->fields;
  schema::ActivationType activation_type = static_cast<const schema::ActivationType *>(fields[0].data)[0];
  return new (std::nothrow) ActivationOptPlugin(name, activation_type);
}

nvinfer1::IPluginV2 *ActivationOptPluginCreater::deserializePlugin(const char *name, const void *serialData,
                                                                   size_t serialLength) noexcept {
  schema::ActivationType activation_type;
  DeserializeValue(&serialData, &serialLength, &activation_type, sizeof(schema::ActivationType));
  return new (std::nothrow) ActivationOptPlugin(name, activation_type);
}
}  // namespace luojianet_ms::lite
