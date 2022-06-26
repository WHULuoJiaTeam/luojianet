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

#include "minddata/dataset/kernels/image/random_lighting_op.h"

#include "minddata/dataset/kernels/image/image_utils.h"

namespace mindspore {
namespace dataset {
const float RandomLightingOp::kAlpha = 0.05;

Status RandomLightingOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);

  float rnd_r = dist_(rnd_rgb_);
  float rnd_g = dist_(rnd_rgb_);
  float rnd_b = dist_(rnd_rgb_);
  return RandomLighting(input, output, rnd_r, rnd_g, rnd_b);
}
}  // namespace dataset
}  // namespace mindspore
