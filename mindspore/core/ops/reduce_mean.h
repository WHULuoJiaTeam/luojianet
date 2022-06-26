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

#ifndef MINDSPORE_CORE_OPS_REDUCE_MEAN_H_
#define MINDSPORE_CORE_OPS_REDUCE_MEAN_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/reduce.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameReduceMean = "ReduceMean";
/// \brief Reduces a dimension of a tensor by averaging all elements in the dimension.
/// Refer to Python API @ref mindspore.ops.ReduceMean for more details.
class MIND_API ReduceMean : public Reduce {
 public:
  MIND_API_BASE_MEMBER(ReduceMean);
  /// \brief Constructor.
  ReduceMean() : Reduce(kNameReduceMean) { InitIOName({"input_x", "axis"}, {"y"}); }
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_REDUCE_MEAN_H_
