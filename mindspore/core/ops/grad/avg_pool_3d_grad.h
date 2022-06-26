/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_AVG_POOL_3D_GRAD_H_
#define MINDSPORE_CORE_OPS_AVG_POOL_3D_GRAD_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAvgPool3DGrad = "AvgPool3DGrad";

class MIND_API AvgPool3DGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(AvgPool3DGrad);
  AvgPool3DGrad() : BaseOperator(kNameAvgPool3DGrad) { InitIOName({"origin_input_size", "grad"}, {"output"}); }
};

abstract::AbstractBasePtr AvgPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_AVG_POOL_3D_GRAD_H_
