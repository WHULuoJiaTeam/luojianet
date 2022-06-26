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

#ifndef MINDSPORE_CORE_OPS_REVERSE_V2_H_
#define MINDSPORE_CORE_OPS_REVERSE_V2_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReverseV2 = "ReverseV2";
/// \brief Reverses specific dimensions of a tensor.
/// Refer to Python API @ref mindspore.ops.ReverseV2 for more details.
class MIND_API ReverseV2 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ReverseV2);
  /// \brief Constructor.
  ReverseV2() : BaseOperator(kNameReverseV2) {}
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.ReverseV2 for the inputs.
  void Init(const std::vector<int64_t> &axis);
  /// \brief Set axis.
  void set_axis(const std::vector<int64_t> &axis);
  /// \brief Get axis.
  ///
  /// \return axis.
  std::vector<int64_t> get_axis() const;
};

abstract::AbstractBasePtr ReverseV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REVERSE_V2_H_
