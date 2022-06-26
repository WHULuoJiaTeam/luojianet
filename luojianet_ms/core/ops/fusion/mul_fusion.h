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

#ifndef LUOJIANET_MS_CORE_OPS_MUL_FUSION_H_
#define LUOJIANET_MS_CORE_OPS_MUL_FUSION_H_
#include "ops/mul.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameMulFusion = "MulFusion";
/// \brief MulFusion defined Mul operator prototype of lite.
class MIND_API MulFusion : public Mul {
 public:
  MIND_API_BASE_MEMBER(MulFusion);
  /// \brief Constructor.
  MulFusion() : Mul(kNameMulFusion) { InitIOName({"x", "y"}, {"output"}); }

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] activation_type Define the activation type.
  void Init(const ActivationType &activation_type);

  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType &activation_type);

  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_MUL_FUSION_H_
