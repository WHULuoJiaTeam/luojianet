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
 * limitations under the License
 */

#ifndef LUOJIANET_MS_CORE_OPS_CHOLESKYINVERSE_H_
#define LUOJIANET_MS_CORE_OPS_CHOLESKYINVERSE_H_

#include <map>
#include <set>
#include <memory>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameCholeskyInverse = "CholeskyInverse";
class MIND_API CholeskyInverse : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(CholeskyInverse);
  CholeskyInverse() : BaseOperator(kNameCholeskyInverse) { InitIOName({"x"}, {"y"}); }
};
abstract::AbstractBasePtr CholeskyInverseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimCholeskyInversePtr = std::shared_ptr<CholeskyInverse>;
}  // namespace ops
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CORE_OPS_CHOLESKYINVERSE_H_
