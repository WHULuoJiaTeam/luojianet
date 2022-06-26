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

#ifndef MINDSPORE_CORE_OPS_SCATTER_ND_UPDATE_H_
#define MINDSPORE_CORE_OPS_SCATTER_ND_UPDATE_H_
#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameScatterNdUpdate = "ScatterNdUpdate";
/// \brief Updates tensor values by using input indices and value.
/// Refer to Python API @ref mindspore.ops.ScatterNdUpdate for more details.
class MIND_API ScatterNdUpdate : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(ScatterNdUpdate);
  /// \brief Constructor.
  ScatterNdUpdate() : BaseOperator(kNameScatterNdUpdate) { InitIOName({"input_x", "indices", "update"}, {"output"}); }
  /// \brief Init.
  void Init() const {}
};
abstract::AbstractBasePtr ScatterNdUpdateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
using kPrimScatterNdUpdatePtr = std::shared_ptr<ScatterNdUpdate>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SCATTER_ND_UPDATE_H_
