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

#include "tools/converter/parser/onnx/onnx_transpose_parser.h"
#include <memory>
#include <vector>
#include "ops/transpose.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxTransposeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Transpose>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  std::vector<int32_t> perm;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "axes" || attribute_name == "perm") {
      perm.resize(onnx_node_attr.ints_size());
      for (int i = 0; i < onnx_node_attr.ints_size(); ++i) {
        perm[i] = onnx_node_attr.ints(i);
      }
    }
  }
  auto perm_value = MakeValue(perm);
  MS_CHECK_TRUE_MSG(perm_value != nullptr, nullptr, "MakeValue failed");
  prim_c->AddAttr("perm", perm_value);
  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxTransposeParser("Transpose", new OnnxTransposeParser());
OnnxNodeRegistrar g_onnxInt8TransposeParser("Int8Transpose", new OnnxTransposeParser());
}  // namespace lite
}  // namespace mindspore
