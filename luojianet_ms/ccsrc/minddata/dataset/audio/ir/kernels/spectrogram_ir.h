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

#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_SPECTROGRAM_IR_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_SPECTROGRAM_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace luojianet_ms {
namespace dataset {
namespace audio {
constexpr char kSpectrogramOperation[] = "Spectrogram";

class SpectrogramOperation : public TensorOperation {
 public:
  SpectrogramOperation(int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad, WindowType window,
                       float power, bool normalized, bool center, BorderType pad_mode, bool onesided);

  ~SpectrogramOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSpectrogramOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
  float power_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_SPECTROGRAM_IR_H_
