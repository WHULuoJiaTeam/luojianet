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

#include "tools/converter/parser/tflite/tflite_split_v_parser.h"
#include <vector>
#include <memory>
#include <map>
#include "ops/split.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteSplitVParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_GE(tflite_op->inputs.size(), kInputSize2, nullptr);
  auto prim = std::make_unique<ops::Split>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  const auto &tflite_attr = tflite_op->builtin_options.AsSplitVOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get op splitv attr failed";
    return nullptr;
  }
  prim->set_output_num(tflite_attr->num_splits);

  std::vector<int64_t> size_splits;
  if (GetTfliteData(tflite_op->inputs[SECOND_INPUT], tflite_subgraph->tensors, tflite_model->buffers, &size_splits)) {
    MS_LOG(ERROR) << "get spliteV -> sizeSplits failed";
    return nullptr;
  }
  prim->set_size_splits(size_splits);

  const auto &tensor = tflite_subgraph->tensors.at(tflite_op->inputs.at(FIRST_INPUT));
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "tensor_shape is null";
    return nullptr;
  }
  auto tensor_shape = tensor->shape;
  std::vector<int64_t> axes;
  auto ret = GetTfliteData(tflite_op->inputs.at(THIRD_INPUT), tflite_subgraph->tensors, tflite_model->buffers, &axes);
  if (ret != RET_OK && ret != RET_NO_CHANGE) {
    MS_LOG(ERROR) << "get axes value failed.";
    return nullptr;
  }
  if (axes.size() < 1) {
    MS_LOG(ERROR) << "invalid axes param";
    return nullptr;
  }
  auto axis = axes[0];
  if (axis < 0) {
    axis += tensor_shape.size();
  }
  if (axis >= static_cast<int32_t>(tensor_shape.size())) {
    MS_LOG(ERROR) << "axis value is too large";
    return nullptr;
  }
  prim->set_axis(static_cast<int64_t>(axis));

  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteSplitVParser(tflite::BuiltinOperator_SPLIT_V, new TfliteSplitVParser());
}  // namespace lite
}  // namespace mindspore
