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

#ifndef LUOJIANET_MS_CORE_OPS_DEPTH_TO_SPACE_H_
#define LUOJIANET_MS_CORE_OPS_DEPTH_TO_SPACE_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameDepthToSpace = "DepthToSpace";
/// \brief Rearrange blocks of depth data into spatial dimensions.
/// Refer to Python API @ref luojianet_ms.ops.DepthToSpace for more details.
class MS_CORE_API DepthToSpace : public PrimitiveC {
 public:
  /// \brief Constructor.
  DepthToSpace() : PrimitiveC(kNameDepthToSpace) { InitIOName({"x"}, {"y"}); }
  /// \brief Destructor.
  ~DepthToSpace() = default;
  MS_DECLARE_PARENT(DepthToSpace, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref luojianet_ms.ops.DepthToSpace for the inputs.
  void Init(const int64_t block_size, const Format &format = NCHW);
  /// \brief Set block_size.
  void set_block_size(const int64_t block_size);
  /// \brief Get block_size.
  ///
  /// \return block_size.
  int64_t get_block_size() const;
  /// \brief Set format.
  void set_format(const Format &format);
  /// \brief Get format.
  ///
  /// \return format.
  Format get_format() const;
};

AbstractBasePtr DepthToSpaceInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CORE_OPS_DEPTH_TO_SPACE_H_
