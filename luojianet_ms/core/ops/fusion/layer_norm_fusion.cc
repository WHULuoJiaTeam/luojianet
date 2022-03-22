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

#include "ops/fusion/layer_norm_fusion.h"

namespace luojianet_ms {
namespace ops {
void LayerNormFusion::Init(const int64_t begin_norm_axis, const int64_t begin_params_axis, const float epsilon,
                           const bool elementwise_affine) {
  this->set_begin_norm_axis(begin_norm_axis);
  this->set_begin_params_axis(begin_params_axis);
  this->set_epsilon(epsilon);
  this->set_elementwise_affine(elementwise_affine);
}

void LayerNormFusion::set_elementwise_affine(const bool elementwise_affine) {
  (void)AddAttr(kElementwiseAffine, MakeValue(elementwise_affine));
}

bool LayerNormFusion::get_elementwise_affine() const {
  auto value_ptr = GetAttr(kElementwiseAffine);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<bool>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameLayerNormFusion, LayerNormFusion);
}  // namespace ops
}  // namespace luojianet_ms
