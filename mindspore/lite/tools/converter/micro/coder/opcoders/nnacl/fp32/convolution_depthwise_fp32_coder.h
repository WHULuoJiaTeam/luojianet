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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_CONVOLUTION_DEPTHWISE_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_CONVOLUTION_DEPTHWISE_FP32_CODER_H_

#include <vector>
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "src/runtime/kernel/arm/fp32/convolution_depthwise_fp32.h"

namespace mindspore::lite::micro::nnacl {
class ConvolutionDepthwiseFP32Coder final : public Conv2DBaseCoder {
 public:
  ConvolutionDepthwiseFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                const Model::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ConvolutionDepthwiseFP32Coder() override = default;
  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int InitWeightBias();
  float *packed_weight_{nullptr};
  float *bias_{nullptr};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_FP32_CONVOLUTION_DEPTHWISE_FP32_CODER_H_
