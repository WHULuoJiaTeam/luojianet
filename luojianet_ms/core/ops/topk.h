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

#ifndef LUOJIANET_MS_CORE_OPS_TOPK_H_
#define LUOJIANET_MS_CORE_OPS_TOPK_H_
#include <vector>
#include <memory>
#include <string>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameTopK = "TopK";
/// \brief Finds values and indices of the k largest entries along the last dimension.
/// Refer to Python API @ref luojianet_ms.ops.TopK for more details.
class MIND_API TopK : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(TopK);
  /// \brief Constructor.
  explicit TopK(const std::string &k_name = kNameTopK) : BaseOperator(k_name) {
    InitIOName({"input", "k"}, {"values", "indices"});
  }
  /// \brief Init. Refer to the parameters of Python API @ref luojianet_ms.ops.TopK for the inputs.
  void Init(const bool sorted = false);
  /// \brief Set sorted.
  void set_sorted(const bool sorted);
  /// \brief Get sorted.
  ///
  /// \return sorted.
  bool get_sorted() const;
};
abstract::AbstractBasePtr TopKInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_TOPK_H_
