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

#ifndef LUOJIANET_MS_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H
#define LUOJIANET_MS_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H
#include <cstddef>
#include <utility>
#include <set>
#include <string>
#include "ir/anf.h"
#include "base/core_ops.h"
#include "utils/check_convert_utils.h"

namespace luojianet_ms::lite::quant {
class QuantStrategy {
 public:
  QuantStrategy(size_t min_quant_weight_size, size_t min_quant_weight_channel, std::set<std::string> skip_node)
      : min_quant_weight_size_(min_quant_weight_size),
        min_quant_weight_channel_(min_quant_weight_channel),
        skip_node_(std::move(skip_node)) {}

  ~QuantStrategy() = default;

  bool CanOpFullQuantized(const AnfNodePtr &node, const std::set<PrimitivePtr> &support_int8_ops,
                          const std::set<PrimitivePtr> &skip_check_dtype_ops,
                          const std::set<luojianet_ms::ActivationType> &support_activation);
  bool CanTensorQuantized(const CNodePtr &cnode, const AnfNodePtr &input_node, int preferred_dim);
  bool IsSkipOp(const std::string &skip_node_name);

 private:
  size_t min_quant_weight_size_;
  size_t min_quant_weight_channel_;
  std::set<std::string> skip_node_;
};
}  // namespace luojianet_ms::lite::quant

#endif  // LUOJIANET_MS_LITE_TOOLS_CONVERTER_QUANTIZER_QUANT_STRATEGY_H
