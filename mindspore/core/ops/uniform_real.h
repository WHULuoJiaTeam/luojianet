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

#ifndef MINDSPORE_CORE_OPS_RANDOM_UNIFORM_REAL_H_
#define MINDSPORE_CORE_OPS_RANDOM_UNIFORM_REAL_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameUniformReal = "UniformReal";
/// \brief Produces random floating-point values i, uniformly distributed to the interval [0, 1).
/// Refer to Python API @ref mindspore.ops.UniformReal for more details.

class MIND_API UniformReal : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(UniformReal);
  /// \brief Constructor.
  UniformReal() : BaseOperator(kNameUniformReal) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.UniformReal for the inputs.
  void Init(int64_t seed, int64_t seed2);
  /// \brief Set seed.
  void set_seed(int64_t seed);
  /// \brief Set seed2.
  void set_seed2(int64_t seed2);
  /// \brief Get seed.
  ///
  /// \return seed.
  int64_t get_seed() const;
  /// \brief Get seed2.
  ///
  /// \return seed2.
  int64_t get_seed2() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_RANDOM_UNIFORM_REAL_H_
