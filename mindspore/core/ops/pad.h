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

#ifndef MINDSPORE_CORE_OPS_PAD_H_
#define MINDSPORE_CORE_OPS_PAD_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePad = "Pad";
/// \brief Pads the input tensor according to the paddings. Refer to Python API @ref mindspore.ops.Pad for more details.
class MIND_API Pad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Pad);
  /// \brief Constructor.
  Pad() : BaseOperator(kNamePad) { InitIOName({"x"}, {"y"}); }
  explicit Pad(const std::string k_name) : BaseOperator(k_name) { InitIOName({"x"}, {"y"}); }
  /// \brief Init. Refer to the parameters of Python API @ref mindspore.ops.Pad for the inputs.
  void Init(const std::vector<std::vector<int64_t>> &paddings);
  /// \brief Set paddings.
  void set_paddings(const std::vector<std::vector<int64_t>> &paddings);
  /// \brief Get paddings.
  ///
  /// \return paddings.
  std::vector<std::vector<int64_t>> get_paddings() const;
};
abstract::AbstractBasePtr PadInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PAD_H_
