/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_ELEMENTWISE_TENSORRT_H_
#define MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_ELEMENTWISE_TENSORRT_H_
#include <string>
#include <vector>
#include <map>
#include "src/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
class ElementWiseTensorRT : public TensorRTOp {
 public:
  ElementWiseTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                      const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~ElementWiseTensorRT() override = default;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

 private:
  nvinfer1::ITensor *AddActivation(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *in_tensor);

  int AddConstTensor(nvinfer1::INetworkDefinition *network);

  bool SameTensor(nvinfer1::ITensor *trt_tensor, mindspore::MSTensor *ms_tensor);

  int PreprocessInputTensors(nvinfer1::INetworkDefinition *network, ITensorHelper *x_input, ITensorHelper *y_input);

  nvinfer1::ElementWiseOperation element_wise_op_;
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_DELEGATE_TENSORRT_OP_ELEMENTWISE_TENSORRT_H_
