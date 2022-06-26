/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_LAYER_NORM_FUSION_H_
#define MINDSPORE_CORE_OPS_LAYER_NORM_FUSION_H_
#include <vector>
#include <memory>

#include "ops/layer_norm.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLayerNormFusion = "LayerNormFusion";
/// \brief LayerNormFusion defined LayerNorm operator prototype of lite.
class MIND_API LayerNormFusion : public LayerNorm {
 public:
  MIND_API_BASE_MEMBER(LayerNormFusion);
  /// \brief Constructor.
  LayerNormFusion() : LayerNorm(kNameLayerNormFusion) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] begin_norm_axis Define the first normalization dimension of input.
  /// \param[in] begin_params_axis Define the first parameter dimension.
  /// \param[in] epsilon Define a value added to the denominator for numerical stability.
  /// \param[in] elementwise_affine Define a boolean value to indicate that the operation is element-wise or not.
  void Init(const int64_t begin_norm_axis = 1, const int64_t begin_params_axis = 1, const float epsilon = 1e-7,
            const bool elementwise_affine = false);

  /// \brief Method to set elementwise_affine attribute.
  ///
  /// \param[in] elementwise_affine Define a boolean value to indicate that the operation is element-wise or not.
  void set_elementwise_affine(const bool elementwise_affine);

  /// \brief Method to get elementwise_affine attribute.
  ///
  /// \return a boolean value.
  bool get_elementwise_affine() const;
};

abstract::AbstractBasePtr LayerNormFusionInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_LAYER_NORM_FUSION_H_
