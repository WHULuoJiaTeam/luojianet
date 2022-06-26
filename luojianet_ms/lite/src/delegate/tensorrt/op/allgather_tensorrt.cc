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

#include "src/delegate/tensorrt/op/allgather_tensorrt.h"
#include <numeric>
#include "NvInferRuntimeCommon.h"

namespace luojianet_ms::lite {
REGISTER_TENSORRT_PLUGIN(AllGatherPluginCreater);

int AllGatherTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<luojianet_ms::MSTensor> &in_tensors,
                                 const std::vector<luojianet_ms::MSTensor> &out_tensors) {
#ifndef LITE_CUDA_DISTRIBUTION
  MS_LOG(ERROR)
    << "Unsupported package for gpu distribution feature, please recompile with MS_ENABLE_CUDA_DISTRIBUTION set to on.";
  return RET_ERROR;
#else
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
#endif
}

int AllGatherTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  nvinfer1::ITensor *inputTensors[] = {tensorrt_in_tensors_[0].trt_tensor_};
  auto allgather_op = op_primitive_->value_as_AllGather();
  if (allgather_op == nullptr) {
    MS_LOG(ERROR) << "convert failed for " << op_name_;
    return RET_ERROR;
  }
  int rank = GetGPUGroupSize();
  auto plugin = std::make_shared<AllGatherPlugin>(op_name_, rank);
  MS_LOG(INFO) << op_name_ << " group size: " << rank << ", rank id: " << GetRankID();
  nvinfer1::IPluginV2Layer *allgather_layer = network->addPluginV2(inputTensors, 1, *plugin);
  if (allgather_layer == nullptr) {
    MS_LOG(ERROR) << "create AllGather layer failed for: " << op_name_;
    return RET_ERROR;
  }
  nvinfer1::ITensor *allgather_out = allgather_layer->getOutput(0);
  allgather_layer->setName(op_name_.c_str());
  allgather_out->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(
    ITensorHelper{allgather_out, tensorrt_in_tensors_[0].format_, tensorrt_in_tensors_[0].same_format_});
  this->layer_ = allgather_layer;
  return RET_OK;
}

// AllGatherPlugin
int AllGatherPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
                             const void *const *inputs, void *const *outputs, void *workspace,
                             cudaStream_t stream) noexcept {
  MS_LOG(INFO) << "all gather run at rank id: " << GetRankID() << " stream: " << stream;
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int send_element_cnt = std::accumulate(input_dims.d, input_dims.d + input_dims.nbDims, 1, std::multiplies<int64_t>());
  const void *input = inputs[0];
  void *output = outputs[0];
  auto ret = DistributionCollective::instance().AllGatherWrapper(input, output, send_element_cnt, inputDesc->type,
                                                                 stream, NCCL_WORLD_GROUP);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "AllGather nccl run failed for " << layer_name_;
    return ret;
  }
  return RET_OK;
}

nvinfer1::IPluginV2DynamicExt *AllGatherPlugin::clone() const noexcept {
  auto *plugin = new AllGatherPlugin(*this);
  plugin->setPluginNamespace(name_space_.c_str());
  return plugin;
}

nvinfer1::DimsExprs AllGatherPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs *inputs,
                                                         int nbInputs, nvinfer1::IExprBuilder &exprBuilder) noexcept {
  nvinfer1::DimsExprs out_dims{};
  out_dims.nbDims = inputs->nbDims;
  auto rank_dim = exprBuilder.constant(rank_);
  out_dims.d[0] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *inputs->d[0], *rank_dim);
  for (int i = 1; i < inputs->nbDims; i++) {
    out_dims.d[i] = inputs->d[i];
  }
  return out_dims;
}

nvinfer1::IPluginV2 *AllGatherPluginCreater::createPlugin(const char *name,
                                                          const nvinfer1::PluginFieldCollection *fc) noexcept {
  const nvinfer1::PluginField *fields = fc->fields;
  int rank = static_cast<const int *>(fields[0].data)[0];
  MS_LOG(DEBUG) << "createPlugin: " << name << " of rank: " << rank;
  return new (std::nothrow) AllGatherPlugin(name, rank);
}

nvinfer1::IPluginV2 *AllGatherPluginCreater::deserializePlugin(const char *name, const void *serialData,
                                                               size_t serialLength) noexcept {
  int rank = GetGPUGroupSize();
  MS_LOG(DEBUG) << name << " is deserialize as rank: " << rank;
  return new (std::nothrow) AllGatherPlugin(name, rank);
}
}  // namespace luojianet_ms::lite
