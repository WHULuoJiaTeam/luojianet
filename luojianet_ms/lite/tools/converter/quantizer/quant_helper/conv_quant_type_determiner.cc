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

#define USE_DEPRECATED_API
#include "tools/converter/quantizer/quant_helper/conv_quant_type_determiner.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "luojianet_ms/core/utils/log_adapter.h"
#include "luojianet_ms/core/ir/dtype/type_id.h"
namespace luojianet_ms::lite {
bool ConvQuantTypeDeterminer::DetermineQuantWeight(const luojianet_ms::schema::MetaGraphT &graph,
                                                   luojianet_ms::schema::CNodeT *node) {
  MS_CHECK_TRUE_MSG(node != nullptr, false, "node is nullptr.");
  MS_ASSERT(node->inputIndex.size() >= kInputIndexTwo);
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->inputIndex.at(kInputIndex), false, "Out of vector range.");
  auto &input_tensor = graph.allTensors.at(node->inputIndex.at(kInputIndex));
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->inputIndex.at(kWeightIndex), false, "Out of vector range.");
  auto &weight_tensor = graph.allTensors.at(node->inputIndex.at(kWeightIndex));
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->outputIndex.at(kOutputIndex), false, "Out of vector range.");
  auto &output_tensor = graph.allTensors.at(node->outputIndex.at(kOutputIndex));
  if ((!quant::TensorQuantParamsInited(*input_tensor) || !quant::TensorQuantParamsInited(*output_tensor)) &&
      quant::TensorQuantParamsInited(*weight_tensor)) {
    node->quantType = schema::QuantType_QUANT_WEIGHT;
    return true;
  }
  return false;
}
}  // namespace luojianet_ms::lite
