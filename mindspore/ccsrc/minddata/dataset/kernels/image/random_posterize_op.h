/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_POSTERIZE_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_POSTERIZE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/kernels/image/posterize_op.h"

namespace mindspore {
namespace dataset {
class RandomPosterizeOp : public PosterizeOp {
 public:
  /// Default values
  static const std::vector<uint8_t> kBitRange;

  /// \brief Constructor
  /// \param[in] bit_range: Minimum and maximum bits in range
  explicit RandomPosterizeOp(const std::vector<uint8_t> &bit_range = kBitRange);

  ~RandomPosterizeOp() override = default;

  std::string Name() const override { return kRandomPosterizeOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  /// Member variables
 private:
  std::string kRandomPosterizeOp = "RandomPosterizeOp";
  std::vector<uint8_t> bit_range_;
  std::mt19937 rnd_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_POSTERIZE_OP_H_
