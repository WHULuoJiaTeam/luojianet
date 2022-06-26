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
#ifndef LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_SUBGTAPH_H_
#define LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_SUBGTAPH_H_
#include <utility>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "include/api/kernel.h"
#include "src/delegate/tensorrt/tensorrt_runtime.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"
#include "src/delegate/tensorrt/tensorrt_serializer.h"
#include "src/delegate/parameter_cache/embedding_cache_manager.h"
#include "include/api/context.h"

namespace luojianet_ms::lite {
using luojianet_ms::lite::RET_ERROR;
using luojianet_ms::lite::RET_OK;
struct CacheTensorInfo {
  std::vector<luojianet_ms::MSTensor> network_input_tensor_;
  bool front_op_can_cache_;
};

class TensorRTSubGraph : public kernel::Kernel {
 public:
  TensorRTSubGraph(std::vector<TensorRTOp *> ops, const std::vector<luojianet_ms::MSTensor> &inputs,
                   const std::vector<luojianet_ms::MSTensor> &outputs, const luojianet_ms::Context *ctx,
                   std::shared_ptr<GPUDeviceInfo> device_info, TensorRTRuntime *runtime, bool support_resize,
                   bool support_hw_resize)
      : kernel::Kernel(inputs, outputs, nullptr, ctx),
        all_ops_(std::move(ops)),
        device_info_(device_info),
        runtime_(runtime) {
    trt_specific_weight_nodes_ = {
      schema::PrimitiveType_Conv2DFusion, schema::PrimitiveType_ReduceFusion, schema::PrimitiveType_Transpose,
      schema::PrimitiveType_Gather,       schema::PrimitiveType_Reshape,      schema::PrimitiveType_PowFusion,
      schema::PrimitiveType_AddFusion,    schema::PrimitiveType_DivFusion,    schema::PrimitiveType_SubFusion,
      schema::PrimitiveType_MatMulFusion, schema::PrimitiveType_PowFusion,    schema::PrimitiveType_Eltwise,
      schema::PrimitiveType_ScaleFusion,  schema::PrimitiveType_MulFusion,    schema::PrimitiveType_Minimum,
      schema::PrimitiveType_StridedSlice, schema::PrimitiveType_PadFusion,    schema::PrimitiveType_FullConnection,
      schema::PrimitiveType_Cast,         schema::PrimitiveType_ExpandDims,   schema::PrimitiveType_Resize,
      schema::PrimitiveType_Maximum,      schema::PrimitiveType_BiasAdd,      schema::PrimitiveType_LSTM};
    if (!support_resize) {
      input_batchsize_index_ = -1;
      input_hw_index_ = -1;
    }
    if (!support_hw_resize) {
      input_hw_index_ = -1;
    }
  }

  ~TensorRTSubGraph() override;

  int Prepare() override;

  int Execute() override;

  int ReSize();

  int BuildTensorRTGraph();

  int Init(cudaStream_t stream);

  void SetCacheManager(const std::shared_ptr<cache::EmbeddingCacheManager> &cache_mgr) { cache_mgr_ = cache_mgr; }

  void SetSerializePath(const std::string &path) { serialize_file_path_ = std::move(path); }

 private:
  int BuildEngine();

  int SetDeviceConfig(cudaStream_t stream);

  bool IsInt8Mode();

  bool SupportFP16();

  nvinfer1::ITensor *SetTensorRTNetworkInput(const luojianet_ms::MSTensor &in_tensor);

  ITensorHelper FindTensorRTInputs(TensorRTOp *cur_op, const luojianet_ms::MSTensor &in_tensor);

  int MarkOutputs();

  bool IsCached(TensorRTOp *cur_op, const luojianet_ms::MSTensor &in_tensor);

  void FindCacheTensorInfo(TensorRTOp *cur_op, luojianet_ms::MSTensor device_cache_tensor);

  bool CanOpCache(TensorRTOp *cur_op);

  int HandleCacheTensor(TensorRTOp *cur_op, const luojianet_ms::MSTensor &in_tensor);

  nvinfer1::Dims ParseInputDimsProfile(const luojianet_ms::MSTensor &in_tensor);

  bool ValidInputResizeDims(const nvinfer1::Dims &construct_dims, const std::vector<int64_t> &resize_input_shape);

  std::vector<TensorRTOp *> all_ops_{};
  // subgraph input nodes.
  std::vector<TensorRTOp *> in_ops_{};
  // subgraph output nodes.
  std::vector<TensorRTOp *> out_ops_{};

  void **tensor_bindings_{nullptr};

  std::shared_ptr<GPUDeviceInfo> device_info_{nullptr};

  TensorRTRuntime *runtime_{nullptr};  // all subgraph in one delegate share a runtime_

  std::set<luojianet_ms::schema::PrimitiveType> trt_specific_weight_nodes_;

  // save in/out tensor name for subgraph isolate.
  std::vector<std::string> trt_in_tensor_name_;
  std::vector<std::string> trt_out_tensor_name_;

  std::vector<luojianet_ms::MSTensor> cache_const_inputs_;
  std::map<std::string, CacheTensorInfo> network_cache_tensor_info_;

  nvinfer1::INetworkDefinition *network_{nullptr};
  nvinfer1::IBuilderConfig *config_{nullptr};
  nvinfer1::ICudaEngine *engine_{nullptr};
  nvinfer1::IExecutionContext *trt_context_{nullptr};
  nvinfer1::IOptimizationProfile *profile_{nullptr};

  // -1 means don't support resize
  int input_batchsize_index_{0};
  int output_batchsize_index_{0};
  int input_hw_index_{0};

  std::map<std::string, std::vector<luojianet_ms::MSTensor>> model_input_to_cache_tensors_;

  std::shared_ptr<cache::EmbeddingCacheManager> cache_mgr_{nullptr};

  std::shared_ptr<TensorRTSerializer> serializer_{nullptr};

  std::string serialize_file_path_;
  cudaStream_t stream_{nullptr};
};
}  // namespace luojianet_ms::lite
#endif  // LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_TENSORRT_SUBGTAPH_H_
