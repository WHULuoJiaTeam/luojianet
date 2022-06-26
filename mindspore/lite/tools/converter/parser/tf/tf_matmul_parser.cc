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
#include "tools/converter/parser/tf/tf_matmul_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/fusion/mat_mul_fusion.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFMatMulParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::MatMulFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  tensorflow::AttrValue attr_value;
  if (TensorFlowUtils::FindAttrValue(tf_op, "transpose_a", &attr_value)) {
    prim->set_transpose_a(attr_value.b());
  }
  if (TensorFlowUtils::FindAttrValue(tf_op, "transpose_b", &attr_value)) {
    prim->set_transpose_b(attr_value.b());
  }
  prim->set_activation_type(mindspore::NO_ACTIVATION);

  *output_size = 1;
  if (AddOpInput(tf_op, 0, inputs) != RET_OK || AddOpInput(tf_op, 1, inputs) != RET_OK) {
    MS_LOG(ERROR) << "add op input failed";
    return nullptr;
  }

  return prim->GetPrim();
}

TFNodeRegistrar g_tfMatMulParser("MatMul", new TFMatMulParser());
}  // namespace lite
}  // namespace mindspore
