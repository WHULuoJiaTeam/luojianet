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

#ifndef LUOJIANET_MS_CORE_OPS_SCALE_H_
#define LUOJIANET_MS_CORE_OPS_SCALE_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameScale = "Scale";
/// \brief Scale defined Scale operator prototype of lite.
class MIND_API Scale : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Scale);
  /// \brief Constructor.
  Scale() : BaseOperator(kNameScale) {}

  /// \brief Constructor.
  explicit Scale(const std::string k_name) : BaseOperator(k_name) {}

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axis Define the first axis of input[0] along which to apply input[1], can be negative to index from
  ///            the end. Default -1.
  void Init(const int64_t axis = -1);

  /// \brief Method to set axis attribute.
  ///
  /// \param[in] axis Define the first axis of input[0] along which to apply input[1], can be negative to index from
  ///            the end. Default -1.
  void set_axis(const int64_t axis);

  /// \brief Method to get axis attribute.
  ///
  /// \return axis attribute.
  int64_t get_axis() const;
};
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_SCALE_H_
