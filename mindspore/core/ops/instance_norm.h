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

#ifndef MINDSPORE_CORE_OPS_INSTANCE_NORM_H_
#define MINDSPORE_CORE_OPS_INSTANCE_NORM_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameInstanceNorm = "InstanceNorm";
/// \brief InstanceNorm defined the InstanceNorm operator prototype.
class MIND_API InstanceNorm : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(InstanceNorm);
  /// \brief Constructor.
  InstanceNorm() : BaseOperator(kNameInstanceNorm) {}

  /// \brief Method to init the op's attributes
  ///
  /// \param[in] epsilon Define a value added to the denominator for numerical stability.
  void Init(const float epsilon = 0.00001);

  /// \brief Method to set epsilon attribute.
  ///
  /// \param[in] epsilon Define a value added to the denominator for numerical stability.
  void set_epsilon(const float epsilon);

  /// \brief Method to get epsilon attribute.
  ///
  /// \return a value.
  float get_epsilon() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_INSTANCE_NORM_H_
