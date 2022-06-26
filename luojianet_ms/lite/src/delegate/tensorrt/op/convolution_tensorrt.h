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
#ifndef LUOJIANET_MS_LITE_SRC_TENSORRT_OP_CONVOLUTION_TENSORRT_H_
#define LUOJIANET_MS_LITE_SRC_TENSORRT_OP_CONVOLUTION_TENSORRT_H_
#include <string>
#include <vector>
#include "src/delegate/tensorrt/op/tensorrt_op.h"

namespace luojianet_ms::lite {
class ConvolutionTensorRT : public TensorRTOp {
 public:
  ConvolutionTensorRT(const schema::Primitive *primitive, const std::vector<luojianet_ms::MSTensor> &in_tensors,
                      const std::vector<luojianet_ms::MSTensor> &out_tensors, const std::string &name,
                      const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~ConvolutionTensorRT() override;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<luojianet_ms::MSTensor> &in_tensors,
                const std::vector<luojianet_ms::MSTensor> &out_tensors) override;

 private:
  void SetAttributes(const schema::Conv2DFusion *ms_op, nvinfer1::IConvolutionLayer *current_layer_);

  void *pack_weight_{nullptr};
};
}  // namespace luojianet_ms::lite
#endif  // LUOJIANET_MS_LITE_SRC_TENSORRT_OP_CONVOLUTION_TENSORRT_H_
