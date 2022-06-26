/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
#include <utility>
#include <string>
#include <vector>
#include "src/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
class MatMulTensorRT : public TensorRTOp {
 public:
  MatMulTensorRT(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                 const std::vector<mindspore::MSTensor> &out_tensors, const std::string &name,
                 const schema::QuantType &quant_type)
      : TensorRTOp(primitive, in_tensors, out_tensors, name, quant_type) {}

  ~MatMulTensorRT() override;

  int IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                const std::vector<mindspore::MSTensor> &out_tensors) override;

  int AddInnerOp(nvinfer1::INetworkDefinition *network) override;

 private:
  int PreprocessMatMulInputs(nvinfer1::INetworkDefinition *network, ITensorHelper *matmul_a, ITensorHelper *matmul_b);

  nvinfer1::ITensor *ProcessWeightTensor(nvinfer1::INetworkDefinition *network);

  nvinfer1::ITensor *AddAsMatmul(nvinfer1::INetworkDefinition *network);

  nvinfer1::ITensor *AddAsFullConnect(nvinfer1::INetworkDefinition *network);

  nvinfer1::ITensor *AddAsOptPlugin(nvinfer1::INetworkDefinition *network);

  nvinfer1::ITensor *AddBias(nvinfer1::INetworkDefinition *network, nvinfer1::ITensor *input_tensor);

  bool RunOptPlugin();

  bool transpose_a_{false};
  bool transpose_b_{false};
  Format out_format_;
  schema::ActivationType activation_{schema::ActivationType::ActivationType_NO_ACTIVATION};
  void *weight_ptr_{nullptr};
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_TENSORRT_OP_MATMUL_TENSORRT_H_
