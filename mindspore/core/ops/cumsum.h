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

#ifndef MINDSPORE_CORE_OPS_CUMSUM_H_
#define MINDSPORE_CORE_OPS_CUMSUM_H_
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameCumSum = "CumSum";
/// \brief Computes the cumulative sum of input tensor along axis.
/// Refer to Python API @ref mindspore.ops.CumSum for more details.
class MIND_API CumSum : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(CumSum);
  /// \brief Constructor.
  CumSum() : BaseOperator(kNameCumSum) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.CumSum for the inputs.
  void Init(const bool exclusive, const bool reverse);
  /// \brief Set exclusive.
  void set_exclusive(const bool exclusive);
  /// \brief Set reverse.
  void set_reverse(const bool reverse);
  /// \brief Get exclusive.
  ///
  /// \return exclusive.
  bool get_exclusive() const;
  /// \brief Get reverse.
  ///
  /// \return reverse.
  bool get_reverse() const;
};
abstract::AbstractBasePtr CumSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimCumSum = std::shared_ptr<CumSum>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CUMSUM_H_
