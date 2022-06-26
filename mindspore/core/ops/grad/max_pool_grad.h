/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_MAX_POOL_GRAD_H_
#define MINDSPORE_CORE_OPS_MAX_POOL_GRAD_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/grad/pool_grad.h"
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMaxPoolGrad = "MaxPoolGrad";
class MIND_API MaxPoolGrad : public PoolGrad {
 public:
  MIND_API_BASE_MEMBER(MaxPoolGrad);
  MaxPoolGrad() : PoolGrad(kNameMaxPoolGrad) { InitIOName({"x_origin", "out_origin", "grad"}, {"output"}); }
};

abstract::AbstractBasePtr MaxPoolGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimMaxPoolGradPtr = std::shared_ptr<MaxPoolGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MAX_POOL_GRAD_H_
