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

#ifndef LUOJIANET_MS_CORE_OPS_LP_NORMALIZATION_H_
#define LUOJIANET_MS_CORE_OPS_LP_NORMALIZATION_H_
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameLpNormalization = "LpNormalization";
/// \brief LpNormalization defined LpNormalization operator prototype of lite.
class MS_CORE_API LpNormalization : public PrimitiveC {
 public:
  /// \brief Constructor.
  LpNormalization() : PrimitiveC(kNameLpNormalization) {}

  /// \brief Destructor.
  ~LpNormalization() = default;

  MS_DECLARE_PARENT(LpNormalization, PrimitiveC);

  /// \brief Method to init the op's attributes.
  ///
  /// \param[in] axis Define the dim to do normalization.
  /// \param[in] p Define the norm series.
  void Init(const int64_t axis, const int64_t p);

  /// \brief Method to set axis attribute.
  ///
  /// \param[in] axis Define the dim to do normalization.
  void set_axis(const int64_t axis);

  /// \brief Method to set p attribute.
  ///
  /// \param[in] p Define the norm series.
  void set_p(const int64_t p);

  /// \brief Method to get axis attribute.
  ///
  /// \return the dim to do normalization.
  int64_t get_axis() const;

  /// \brief Method to get p attribute.
  ///
  /// \return the norm series.
  int64_t get_p() const;
};

}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_LP_NORMALIZATION_H_
