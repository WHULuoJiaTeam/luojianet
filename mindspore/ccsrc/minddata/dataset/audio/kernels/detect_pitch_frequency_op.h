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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_DETECT_PITCH_FREQUENCY_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_DETECT_PITCH_FREQUENCY_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class DetectPitchFrequencyOp : public TensorOp {
 public:
  DetectPitchFrequencyOp(int32_t sample_rate, float frame_time, int32_t win_length, int32_t freq_low, int32_t freq_high)
      : sample_rate_(sample_rate),
        frame_time_(frame_time),
        win_length_(win_length),
        freq_low_(freq_low),
        freq_high_(freq_high) {}

  ~DetectPitchFrequencyOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": sample_rate: " << sample_rate_ << ", frame_time: " << frame_time_
        << ", win_length: " << win_length_ << ", freq_low: " << freq_low_ << ", freq_high: " << freq_high_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kDetectPitchFrequencyOp; }

 private:
  int32_t sample_rate_;
  float frame_time_;
  int32_t win_length_;
  int32_t freq_low_;
  int32_t freq_high_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_DETECT_PITCH_FREQUENCY_OP_H_
