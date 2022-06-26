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

#ifndef LUOJIANET_MS_CORE_OPS_REDUCE_ASUM_H_
#define LUOJIANET_MS_CORE_OPS_REDUCE_ASUM_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/reduce.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameReduceASum = "ReduceASum";
class MIND_API ReduceASum : public Reduce {
 public:
  MIND_API_BASE_MEMBER(ReduceASum);
  ReduceASum() : Reduce(kNameReduceASum) { InitIOName({"input_x", "axis"}, {"y"}); }
  void Init() const {}
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_REDUCE_ASUM_H_
