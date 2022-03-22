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

#ifndef LUOJIANET_MS_LITE_MICRO_CODER_OPCODER_CONVOLUTION_DEPTHWISE_INT8_CODER_H_
#define LUOJIANET_MS_LITE_MICRO_CODER_OPCODER_CONVOLUTION_DEPTHWISE_INT8_CODER_H_

#include <vector>
#include "coder/opcoders/base/conv2d_base_coder.h"
#include "src/runtime/kernel/arm/int8/convolution_depthwise_int8.h"

namespace luojianet_ms::lite::micro {
class ConvolutionDepthwiseINT8Coder : public Conv2DBaseCoder {
 public:
  ConvolutionDepthwiseINT8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                                const Model::Node *node, size_t node_index, Target target)
      : Conv2DBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ConvolutionDepthwiseINT8Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int InitBuffer(CoderContext *const context);

  int InitWeightBias(CoderContext *const context);

  int32_t *row_buffer_{nullptr};

  size_t row_buffer_size_{0};

  int16_t *packed_weight_{nullptr};

  int32_t *bias_data_{nullptr};
};
}  // namespace luojianet_ms::lite::micro

#endif  // LUOJIANET_MS_LITE_MICRO_CODER_OPCODER_CONVOLUTION_DEPTHWISE_INT8_CODER_H_
