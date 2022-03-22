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

#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_ADJUST_GAMMA_OP_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_ADJUST_GAMMA_OP_H_

#include <memory>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/cv_tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

namespace luojianet_ms {
namespace dataset {
class AdjustGammaOp : public TensorOp {
 public:
  /// Default gain to be used
  static const float kGain;
  AdjustGammaOp(const float &gamma, const float &gain) : gamma_(gamma), gain_(gain) {}

  ~AdjustGammaOp() override = default;

  /// Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const AdjustGammaOp &so) {
    so.Print(out);
    return out;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kAdjustGammaOp; }

 private:
  float gamma_;
  float gain_;
};
}  // namespace dataset
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_ADJUST_GAMMA_OP_H_
