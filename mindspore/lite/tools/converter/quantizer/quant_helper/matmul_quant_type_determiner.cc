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

#define USE_DEPRECATED_API
#include "tools/converter/quantizer/quant_helper/matmul_quant_type_determiner.h"
#include "tools/converter/quantizer/quantize_util.h"
#include "mindspore/core/utils/log_adapter.h"
#include "mindspore/core/ir/dtype/type_id.h"
namespace mindspore::lite {
bool MatmulQuantTypeDeterminer::DetermineQuantWeight(const mindspore::schema::MetaGraphT &graph,
                                                     mindspore::schema::CNodeT *node) {
  MS_CHECK_TRUE_MSG(node != nullptr, false, "node is nullptr.");
  MS_ASSERT(node->inputIndex.size() >= kInputIndexTwo);
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->inputIndex.at(kInputIndex), false, "Out of vector range.");
  auto &input_tensor1 = graph.allTensors.at(node->inputIndex.at(kInputIndex));
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->inputIndex.at(kWeightIndex), false, "Out of vector range.");
  auto &input_tensor2 = graph.allTensors.at(node->inputIndex.at(kWeightIndex));
  MS_CHECK_TRUE_MSG(graph.allTensors.size() > node->outputIndex.at(kOutputIndex), false, "Out of vector range.");
  auto &output_tensor = graph.allTensors.at(node->outputIndex.at(kOutputIndex));
  if (((!quant::TensorQuantParamsInited(*input_tensor1) && !quant::TensorQuantParamsInited(*input_tensor2)) ||
       (!quant::TensorQuantParamsInited(*input_tensor1) && !quant::TensorQuantParamsInited(*input_tensor2))) &&
      quant::TensorQuantParamsInited(*output_tensor)) {
    node->quantType = schema::QuantType_QUANT_WEIGHT;
    return true;
  }
  return false;
}
}  // namespace mindspore::lite
