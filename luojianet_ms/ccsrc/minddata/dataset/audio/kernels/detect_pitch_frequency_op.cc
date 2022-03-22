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
#include "minddata/dataset/audio/kernels/detect_pitch_frequency_op.h"

#include "minddata/dataset/audio/kernels/audio_utils.h"
#include "minddata/dataset/util/status.h"

namespace luojianet_ms {
namespace dataset {
Status DetectPitchFrequencyOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // check input tensor dimension, it should be greater than 0.
  RETURN_IF_NOT_OK(ValidateLowRank("DetectPitchFrequency", input, kMinAudioDim, "<..., time>"));
  // check input type, it should be DE_FLOAT16, DE_FLOAT32 or DE_FLOAT64
  RETURN_IF_NOT_OK(ValidateTensorFloat("DetectPitchFrequency", input));
  return DetectPitchFrequency(input, output, sample_rate_, frame_time_, win_length_, freq_low_, freq_high_);
}
}  // namespace dataset
}  // namespace luojianet_ms
