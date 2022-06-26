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

#include "tools/converter/parser/onnx/onnx_argmax_parser.h"
#include <memory>
#include "ops/fusion/arg_max_fusion.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxArgMaxParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::ArgMaxFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "axis") {
      prim->set_axis(onnx_node_attr.i());
    } else if (attribute_name == "keepdims") {
      prim->set_keep_dims(static_cast<bool>(onnx_node_attr.i()));
    }
  }

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxArgMaxParser("ArgMax", new OnnxArgMaxParser());
}  // namespace lite
}  // namespace mindspore
