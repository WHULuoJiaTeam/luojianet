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

#include "tools/converter/parser/onnx/onnx_slice_parser.h"
#include <functional>
#include <memory>
#include <numeric>
#include <algorithm>
#include <vector>
#include <string>
#include "ops/strided_slice.h"
#include "ops/op_utils.h"
#include "include/registry/converter_context.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr OnnxSliceParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::StridedSlice>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto prim_c = prim->GetPrim();
  MS_CHECK_TRUE_RET(prim_c != nullptr, nullptr);
  std::vector<int32_t> starts;
  std::vector<int32_t> ends;
  std::vector<int32_t> axes;
  std::vector<int32_t> steps;
  constexpr int64_t int_32_max = INT32_MAX;
  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "starts") {
      const int num = onnx_node_attr.ints_size();
      starts.clear();
      for (int i = 0; i < num; ++i) {
        starts.push_back(static_cast<int>(std::min(onnx_node_attr.ints()[i], int_32_max)));
      }
    } else if (attribute_name == "axes") {
      const int num = onnx_node_attr.ints_size();
      axes.clear();
      for (int i = 0; i < num; ++i) {
        axes.push_back(static_cast<int>(std::min(onnx_node_attr.ints()[i], int_32_max)));
      }
    } else if (attribute_name == "ends") {
      const int num = onnx_node_attr.ints_size();
      ends.clear();
      for (int i = 0; i < num; ++i) {
        ends.push_back(static_cast<int>(std::min(onnx_node_attr.ints()[i], int_32_max)));
      }
    } else if (attribute_name == "steps") {
      const int num = onnx_node_attr.ints_size();
      steps.clear();
      for (int i = 0; i < num; ++i) {
        steps.push_back(static_cast<int>(std::min(onnx_node_attr.ints()[i], int_32_max)));
      }
    }
  }
  int size = -1;
  if (!starts.empty()) {
    size = static_cast<int>(starts.size());
  } else if (!ends.empty()) {
    size = static_cast<int>(ends.size());
  } else if (!axes.empty()) {
    size = static_cast<int>(axes.size());
  } else if (!steps.empty()) {
    size = static_cast<int>(steps.size());
  }
  if (size == -1) {
    return prim->GetPrim();
  }
  if (axes.empty()) {
    for (size_t i = 0; i < starts.size(); ++i) {
      axes.push_back(i);
    }
  }
  if (steps.empty()) {
    steps.assign(starts.size(), 1);
  }

  prim_c->AddAttr("starts", MakeValue(starts));
  prim_c->AddAttr("axes", MakeValue(axes));
  prim_c->AddAttr("ends", MakeValue(ends));
  prim_c->AddAttr("steps", MakeValue(steps));
  int64_t fmk_type = converter::FmkType::kFmkTypeOnnx;
  prim_c->AddAttr(ops::kFmkType, MakeValue(fmk_type));

  return prim->GetPrim();
}

OnnxNodeRegistrar g_onnxSliceParser("Slice", new OnnxSliceParser());
OnnxNodeRegistrar g_onnxDynamicSliceParser("DynamicSlice", new OnnxSliceParser());
}  // namespace lite
}  // namespace mindspore
