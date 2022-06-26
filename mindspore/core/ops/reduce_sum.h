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

#ifndef MINDSPORE_CORE_OPS_REDUCE_SUM_H_
#define MINDSPORE_CORE_OPS_REDUCE_SUM_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/reduce.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReduceSum = "ReduceSum";
/// \brief Reduces a dimension of a tensor by summing all elements in the dimension, by default.
/// Refer to Python API @ref mindspore.ops.ReduceSum for more details.
class MIND_API ReduceSum : public Reduce {
 public:
  MIND_API_BASE_MEMBER(ReduceSum);
  /// \brief Constructor.
  ReduceSum() : Reduce(kNameReduceSum) { InitIOName({"x", "axis"}, {"y"}); }
  /// \brief Init.
  void Init() const {}
};
abstract::AbstractBasePtr ReduceSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REDUCE_SUM_H_
