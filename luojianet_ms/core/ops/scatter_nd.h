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

#ifndef LUOJIANET_MS_CORE_OPS_SCATTER_ND_H_
#define LUOJIANET_MS_CORE_OPS_SCATTER_ND_H_
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameScatterNd = "ScatterNd";
/// \brief Scatters a tensor into a new tensor depending on the specified indices.
/// Refer to Python API @ref luojianet_ms.ops.ScatterNd for more details.
class MIND_API ScatterNd : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ScatterNd);
  /// \brief Constructor.
  ScatterNd() : BaseOperator(kNameScatterNd) { InitIOName({"indices", "update", "shape"}, {"output"}); }
  /// \brief Init.
  void Init() const {}
};
abstract::AbstractBasePtr ScatterNdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_SCATTER_ND_H_
