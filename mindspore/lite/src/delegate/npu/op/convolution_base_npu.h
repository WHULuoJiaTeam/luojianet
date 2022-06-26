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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_BASE_NPU_H_
#define MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_BASE_NPU_H_

#include <utility>
#include <vector>
#include <memory>
#include <string>
#include "include/graph/op/all_ops.h"
#include "src/delegate/npu/op/npu_op.h"
namespace mindspore {
constexpr int WEIGHT_INDEX = 1;
constexpr int BIAS_INDEX = 2;
constexpr int CONV_INPUT_SIZE = 3;

class ConvolutionBaseNPUOp : public NPUOp {
 public:
  ConvolutionBaseNPUOp(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors, std::string name)
      : NPUOp(primitive, in_tensors, out_tensors, std::move(name)) {}

  ~ConvolutionBaseNPUOp() override;

 protected:
  template <typename T>
  void SetQuantParam(T *conv_, const std::vector<mindspore::MSTensor> &in_tensors) {
    conv_->set_attr_x_quant_scale(in_tensors.at(0).QuantParams().front().scale);
    conv_->set_attr_x_quant_offset(in_tensors.at(0).QuantParams().front().zero_point);
    conv_->set_attr_x_quant_type(1);

    std::vector<float> filter_scales(in_tensors.at(WEIGHT_INDEX).QuantParams().size());
    for (size_t i = 0; i < in_tensors.at(WEIGHT_INDEX).QuantParams().size(); i++) {
      filter_scales[i] = in_tensors.at(WEIGHT_INDEX).QuantParams().at(i).scale;
    }
    conv_->set_attr_filter_quant_scales(filter_scales);
    conv_->set_attr_filter_quant_type(1);
  }
  int InitWeightConst(const std::vector<mindspore::MSTensor> &inputs);
  int InitBiasConst(const std::vector<mindspore::MSTensor> &inputs);
  int SetActivation(const ge::Operator *input, schema::ActivationType act_type);
  void FreeTmpWeight();
  hiai::op::Activation *act_ = nullptr;
  hiai::op::Const *weight_ = nullptr;
  hiai::op::Const *bias_ = nullptr;
  float *fp32_weight_ = nullptr;
  void *nchw_weight_ = nullptr;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_RUNTIME_DELEGATE_NPU_OP_CONVOLUTION_BASE_NPU_H_
