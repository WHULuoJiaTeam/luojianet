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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_EQUALIZER_BIQUAD_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_EQUALIZER_BIQUAD_OP_H_

#include <cmath>
#include <memory>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

class EqualizerBiquadOp : public TensorOp {
 public:
  static const float kQ;

  EqualizerBiquadOp(int32_t sample_rate, float center_freq, float gain, float Q)
      : sample_rate_(sample_rate), center_freq_(center_freq), gain_(gain), Q_(Q) {}

  ~EqualizerBiquadOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kEqualizerBiquadOp; }

 protected:
  int32_t sample_rate_;
  float center_freq_;
  float gain_;
  float Q_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_EQUALIZER_BIQUAD_OP_H_
