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
#include "tools/converter/parser/tf/tf_concat_parser.h"
#include <string>
#include <memory>
#include <map>
#include <vector>
#include "tools/converter/parser/tf/tf_node_parser_registry.h"
#include "ops/concat.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TFConcatParser::Parse(const tensorflow::NodeDef &tf_op,
                                    const std::map<string, const tensorflow::NodeDef *> &tf_node_map,
                                    std::vector<std::string> *inputs, int *output_size) {
  auto prim = std::make_unique<ops::Concat>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  auto axis_node = GetConstInputNode(tf_node_map, tf_op.input(tf_op.input_size() - 1));
  if (axis_node == nullptr) {
    MS_LOG(ERROR) << "get concat axis attr node failed";
    return nullptr;
  }
  tensorflow::AttrValue attr_value;
  if (!TensorFlowUtils::FindAttrValue(*axis_node, "value", &attr_value)) {
    MS_LOG(ERROR) << "The value attr should be specified";
    return nullptr;
  }
  auto tensor_proto = attr_value.tensor();
  prim->set_axis(tensor_proto.int_val(0));

  *output_size = 1;
  for (int i = 0; i < tf_op.input_size() - 1; ++i) {
    if (AddOpInput(tf_op, i, inputs) != RET_OK) {
      MS_LOG(ERROR) << "add op input failed";
      return nullptr;
    }
  }
  prim_c->AddAttr(ops::kOriginalOpName, MakeValue("ConcatV2"));
  return prim->GetPrim();
}

TFNodeRegistrar g_tfConcatV2Parser("ConcatV2", new TFConcatParser());
}  // namespace lite
}  // namespace mindspore
