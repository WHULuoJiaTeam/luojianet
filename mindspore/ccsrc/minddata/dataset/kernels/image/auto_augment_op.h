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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_AUTO_AUGMENT_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_AUTO_AUGMENT_OP_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/image_utils.h"
#else
#include "minddata/dataset/kernels/image/lite_image_utils.h"
#endif
#include "minddata/dataset/kernels/image/math_utils.h"
#include "minddata/dataset/util/status.h"

typedef std::vector<std::vector<std::tuple<std::string, float, int32_t>>> Transforms;
typedef std::map<std::string, std::tuple<std::vector<float>, bool>> Space;

namespace mindspore {
namespace dataset {
class AutoAugmentOp : public TensorOp {
 public:
  AutoAugmentOp(AutoAugmentPolicy policy, InterpolationMode interpolation, const std::vector<uint8_t> &fill_value);

  ~AutoAugmentOp() override = default;

  std::string Name() const override { return kAutoAugmentOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  void GetParams(int transform_num, int *transform_id, std::vector<float> *probs, std::vector<int32_t> *signs);

  Transforms GetTransforms(AutoAugmentPolicy policy);

  Space GetSpace(int32_t num_bins, const std::vector<dsize_t> &image_size);

  Status ApplyAugment(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, const std::string &op_name,
                      float magnitude);

  AutoAugmentPolicy policy_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
  std::mt19937 rnd_;
  Transforms transforms_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_IMAGE_AUTO_AUGMENT_OP_H_
