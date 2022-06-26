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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_AUTO_CONTRAST_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_AUTO_CONTRAST_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
class RandomAutoContrastOp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const float kCutOff;
  static const std::vector<uint32_t> kIgnore;
  static const float kDefProbability;

  RandomAutoContrastOp(float cutoff, const std::vector<uint32_t> &ignore, float prob = kDefProbability)
      : cutoff_(cutoff), ignore_(ignore), distribution_(prob) {
    is_deterministic_ = false;
    rnd_.seed(GetSeed());
  }

  ~RandomAutoContrastOp() override = default;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const RandomAutoContrastOp &so) {
    so.Print(out);
    return out;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandomAutoContrastOp; }

 private:
  std::mt19937 rnd_;
  float cutoff_;
  std::vector<uint32_t> ignore_;
  std::bernoulli_distribution distribution_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_RANDOM_AUTO_CONTRAST_OP_H_
