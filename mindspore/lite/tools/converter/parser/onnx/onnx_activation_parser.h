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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_RELU_PARSER_H
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_RELU_PARSER_H
#define USE_DEPRECATED_API

#include "tools/converter/parser/onnx/onnx_node_parser.h"
#include "tools/converter/parser/onnx/onnx_node_parser_registry.h"

namespace mindspore {
namespace lite {
class OnnxReluParser : public OnnxNodeParser {
 public:
  OnnxReluParser() : OnnxNodeParser("Relu") {}
  ~OnnxReluParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxLeakyReluParser : public OnnxNodeParser {
 public:
  OnnxLeakyReluParser() : OnnxNodeParser("LeakyRelu") {}
  ~OnnxLeakyReluParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxPReluParser : public OnnxNodeParser {
 public:
  OnnxPReluParser() : OnnxNodeParser("Prelu") {}
  ~OnnxPReluParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxEluParser : public OnnxNodeParser {
 public:
  OnnxEluParser() : OnnxNodeParser("Elu") {}
  ~OnnxEluParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxTanhParser : public OnnxNodeParser {
 public:
  OnnxTanhParser() : OnnxNodeParser("Tanh") {}
  ~OnnxTanhParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxSigmoidParser : public OnnxNodeParser {
 public:
  OnnxSigmoidParser() : OnnxNodeParser("Sigmoid") {}
  ~OnnxSigmoidParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxHardSigmoidParser : public OnnxNodeParser {
 public:
  OnnxHardSigmoidParser() : OnnxNodeParser("HardSigmoid") {}
  ~OnnxHardSigmoidParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

class OnnxSoftPlusParser : public OnnxNodeParser {
 public:
  OnnxSoftPlusParser() : OnnxNodeParser("Softplus") {}
  ~OnnxSoftPlusParser() override = default;

  PrimitiveCPtr Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) override;
};

}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_ONNX_RELU_PARSER_H
