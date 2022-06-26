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

#ifndef MINDSPORE_CORE_OPS_FUSION_MAT_MUL_FUSION_H_
#define MINDSPORE_CORE_OPS_FUSION_MAT_MUL_FUSION_H_
#include <vector>
#include <memory>

#include "ops/mat_mul.h"
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMatMulFusion = "MatMulFusion";
/// \brief Multiplies matrix a and matrix b. Refer to Python API @ref mindspore.ops.MatMul for more details.
class MIND_API MatMulFusion : public MatMul {
 public:
  MIND_API_BASE_MEMBER(MatMulFusion);
  /// \brief Constructor.
  MatMulFusion() : MatMul(kNameMatMulFusion) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.MatMulFusion for the inputs.
  void Init(bool transpose_a = false, bool transpose_b = false, const ActivationType &activation_type = NO_ACTIVATION);
  /// \brief Method to set activation type.
  ///
  /// \param[in] activation_type Define the activation type.
  void set_activation_type(const ActivationType activation_type);
  /// \brief Method to get activation type.
  ///
  /// \return activation type.
  ActivationType get_activation_type() const;
};
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_FUSION_MAT_MUL_FUSION_H_
