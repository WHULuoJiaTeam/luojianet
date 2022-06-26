/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_FADE_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_FADE_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
namespace audio {
constexpr char kFadeOperation[] = "Fade";

class FadeOperation : public TensorOperation {
 public:
  FadeOperation(int32_t fade_in_len, int32_t fade_out_len, FadeShape fade_shape);

  ~FadeOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kFadeOperation; }

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t fade_in_len_;
  int32_t fade_out_len_;
  FadeShape fade_shape_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_IR_KERNELS_FADE_IR_H_
