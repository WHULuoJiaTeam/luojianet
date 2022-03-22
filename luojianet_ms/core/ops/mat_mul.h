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

#ifndef LUOJIANET_MS_CORE_OPS_MAT_MUL_H_
#define LUOJIANET_MS_CORE_OPS_MAT_MUL_H_
#include <vector>
#include <memory>
#include <string>
#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"

namespace luojianet_ms {
namespace ops {
constexpr auto kNameMatMul = "MatMul";
/// \brief Multiplies matrix a and matrix b. Refer to Python API @ref luojianet_ms.ops.MatMul for more details.
class MS_CORE_API MatMul : public PrimitiveC {
 public:
  /// \brief Constructor.
  MatMul() : PrimitiveC(kNameMatMul) { InitIOName({"x1", "x2"}, {"output"}); }
  explicit MatMul(const std::string k_name) : PrimitiveC(k_name) { InitIOName({"x", "x2"}, {"output"}); }
  /// \brief Destructor.
  ~MatMul() = default;
  MS_DECLARE_PARENT(MatMul, PrimitiveC);
  /// \brief Init. Refer to the parameters of Python API @ref luojianet_ms.ops.MatMul for the inputs.
  void Init(bool transpose_a = false, bool transpose_b = false);
  /// \brief Set transpose_a.
  void set_transpose_a(bool transpose_a);
  /// \brief Set transpose_b.
  void set_transpose_b(bool transpose_b);
  /// \brief Get transpose_a.
  ///
  /// \return transpose_a.
  bool get_transpose_a() const;
  /// \brief Get transpose_b.
  ///
  /// \return transpose_b.
  bool get_transpose_b() const;
};
}  // namespace ops
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CORE_OPS_MAT_MUL_H_
