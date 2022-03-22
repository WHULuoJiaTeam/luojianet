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

#ifndef DPICO_PARSER_ONNX_ONNX_ROI_ALIGN_PARSER_H_
#define DPICO_PARSER_ONNX_ONNX_ROI_ALIGN_PARSER_H_

#include "include/registry/node_parser.h"

namespace luojianet_ms {
namespace lite {
class OnnxRoiAlignParser : public converter::NodeParser {
 public:
  OnnxRoiAlignParser() : NodeParser() {}
  ~OnnxRoiAlignParser() override = default;

  ops::PrimitiveC *Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};
}  // namespace lite
}  // namespace luojianet_ms
#endif  // DPICO_PARSER_ONNX_ONNX_ROI_ALIGN_PARSER_H_
