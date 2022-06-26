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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_CMSIS_NN_SOFTMAX_INT8_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_CMSIS_NN_SOFTMAX_INT8_CODER_H_

#include <string>
#include <memory>
#include <vector>
#include "coder/opcoders/base/softmax_base_coder.h"

namespace mindspore::lite::micro::cmsis {
class SoftMaxInt8Coder final : public SoftmaxBaseCoder {
 public:
  SoftMaxInt8Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                   const Model::Node *node, size_t node_index, Target target)
      : SoftmaxBaseCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~SoftMaxInt8Coder() override = default;

  int Prepare(CoderContext *const context) override;

  int DoCode(CoderContext *const context) override;

 private:
  int32_t num_rows_{0};
  int32_t row_size_{0};
  int32_t mult_{0};
  int32_t shift_{0};
  int32_t diff_min_{0};
};
}  // namespace mindspore::lite::micro::cmsis
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_CMSIS_NN_SOFTMAX_INT8_CODER_H_
