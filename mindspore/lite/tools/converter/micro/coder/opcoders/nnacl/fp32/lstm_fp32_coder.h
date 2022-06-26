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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_LSTM_FP32_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_LSTM_FP32_CODER_H_

#include <vector>
#include "coder/opcoders/op_coder.h"
#include "nnacl/lstm_parameter.h"

namespace mindspore::lite::micro::nnacl {
class LstmFP32Coder final : public OperatorCoder {
 public:
  LstmFP32Coder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~LstmFP32Coder() override = default;

  int Prepare(CoderContext *const context) override;
  int DoCode(CoderContext *const context) override;

 private:
  int InitParam();
  int ReSize(CoderContext *const context);
  int MallocRunBuffer(CoderContext *const context);
  int InitInputWeightBias(CoderContext *const context);
  int InitStateWeightBias(CoderContext *const context);

  float *weight_i_ptr_{nullptr};
  float *weight_h_ptr_{nullptr};
  float *input_bias_{nullptr};
  float *state_bias_{nullptr};
  float *buffer_[7] = {nullptr};
  int row_tile_{0};
  int col_tile_{0};
  int weight_batch_{0};
  bool is_vec_{false};
  LstmParameter *lstm_param_{nullptr};
};
}  // namespace mindspore::lite::micro::nnacl
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_NNACL_LSTM_FP32_CODER_H_
