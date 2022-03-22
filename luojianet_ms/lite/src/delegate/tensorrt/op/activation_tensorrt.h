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
#ifndef LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_OP_ACTIVATION_TENSORRT_H_
#define LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_OP_ACTIVATION_TENSORRT_H_
#include <string>
#include <vector>
#include "src/delegate/tensorrt/op/tensorrt_op.h"

namespace luojianet_ms::lite {
class ActivationTensorRT : public TensorRTOp {
 public:
  ActivationTensorRT(const schema::Primitive *primitive, const std::vector<luojianet_ms::MSTensor> &in_tensors,
                     const std::vector<luojianet_ms::MSTensor> &out_tensors, const std::string &name)
      : TensorRTOp(primitive, in_tensors, out_tensors, name) {}

  ~ActivationTensorRT() override = default;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<luojianet_ms::MSTensor> &in_tensors,
                const std::vector<luojianet_ms::MSTensor> &out_tensors) override;

  static nvinfer1::IActivationLayer *AddActivation(nvinfer1::INetworkDefinition *network,
                                                   schema::ActivationType activation_type, float alpha, float min_value,
                                                   float max_value, nvinfer1::ITensor *trt_in_tensor);

 private:
  nvinfer1::ActivationType action_code_;
};
}  // namespace luojianet_ms::lite
#endif  // LUOJIANET_MS_LITE_SRC_DELEGATE_TENSORRT_OP_ACTIVATION_TENSORRT_H_
