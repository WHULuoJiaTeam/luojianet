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

#ifndef MINDSPORE_CORE_OPS_LARS_V2_UPDATE_H_
#define MINDSPORE_CORE_OPS_LARS_V2_UPDATE_H_
#include <vector>
#include <memory>
#include <string>
#include <set>
#include <map>
#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameLARSUpdate = "LARSUpdate";
class MIND_API LARSUpdate : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(LARSUpdate);
  explicit LARSUpdate(const std::string &name = kNameLARSUpdate) : BaseOperator(name) {}
};
abstract::AbstractBasePtr LARSUpdateInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<abstract::AbstractBasePtr> &input_args);
using PrimLARSUpdatePtr = std::shared_ptr<LARSUpdate>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_LARS_V2_UPDATE_H_
