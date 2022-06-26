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
#ifndef LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_ATTENTION_H_
#define LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_ATTENTION_H_
#include <map>
#include <vector>
#include <string>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameAttention = "Attention";
/// \brief MultiHead-Attention op in MindIR.
class MIND_API Attention : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(Attention);
  /// \brief Constructor.
  Attention() : BaseOperator(kNameAttention) {
    InitIOName(
      {"q", "k", "v", "weight_q", "weight_k", "weight_v", "weight_o", "bias_q", "bias_k", "bias_v", "bias_o", "mask"},
      {"output"});
  }
  /// \brief Initialize Attention op.
  void Init() const {}
};
}  // namespace ops
}  // namespace mindspore
#endif  // LITE_MINDSPORE_LITE_TOOLS_CONVERTER_OPS_ATTENTION_H_
