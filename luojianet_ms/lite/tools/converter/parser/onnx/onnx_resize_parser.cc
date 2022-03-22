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

#include "tools/converter/parser/onnx/onnx_resize_parser.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include "ops/resize.h"
#include "ops/op_utils.h"
#include "nnacl/op_base.h"

namespace luojianet_ms {
namespace lite {
ops::PrimitiveC *OnnxResizeParser::Parse(const onnx::GraphProto &onnx_graph, const onnx::NodeProto &onnx_node) {
  auto prim = std::make_unique<ops::Resize>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->AddAttr(luojianet_ms::ops::kOriginalFormat, MakeValue<int64_t>(luojianet_ms::Format::NCHW));
  prim->set_nearest_mode(luojianet_ms::NearestMode::ROUND_HALF_DOWN);

  for (const auto &onnx_node_attr : onnx_node.attribute()) {
    const auto &attribute_name = onnx_node_attr.name();
    if (attribute_name == "coordinate_transformation_mode") {
      std::map<std::string, luojianet_ms::CoordinateTransformMode> transform_map = {
        {"half_pixel", luojianet_ms::CoordinateTransformMode::HALF_PIXEL},
        {"pytorch_half_pixel", luojianet_ms::CoordinateTransformMode::HALF_PIXEL},
        {"align_corners", luojianet_ms::CoordinateTransformMode::ALIGN_CORNERS},
        {"asymmetric", luojianet_ms::CoordinateTransformMode::ASYMMETRIC}};
      if (transform_map.find(onnx_node_attr.s()) != transform_map.end()) {
        prim->set_coordinate_transform_mode(transform_map[onnx_node_attr.s()]);
      } else {
        MS_LOG(ERROR) << "Unsupported coordinate transform mode: " << attribute_name;
        return nullptr;
      }
    } else if (attribute_name == "cubic_coeff_a") {
      prim->set_cubic_coeff(onnx_node_attr.f());
    } else if (attribute_name == "exclude_outside") {
      prim->set_exclude_outside(onnx_node_attr.i());
    } else if (attribute_name == "extrapolation_value") {
      prim->set_extrapolation_value(onnx_node_attr.f());
    } else if (attribute_name == "mode") {
      std::map<std::string, luojianet_ms::ResizeMethod> resize_mode = {
        {"nearest", luojianet_ms::ResizeMethod::NEAREST},
        {"linear", luojianet_ms::ResizeMethod::LINEAR},
        {"cubic", luojianet_ms::ResizeMethod::CUBIC},
      };
      if (resize_mode.find(onnx_node_attr.s()) != resize_mode.end()) {
        prim->set_method(resize_mode[onnx_node_attr.s()]);
      } else {
        MS_LOG(ERROR) << "Unsupported resize mode: " << attribute_name;
        return nullptr;
      }
    } else if (attribute_name == "nearest_mode") {
      std::map<std::string, luojianet_ms::NearestMode> nearest_mode = {
        {"round_prefer_floor", luojianet_ms::NearestMode::ROUND_HALF_DOWN},
        {"round_prefer_ceil", luojianet_ms::NearestMode::ROUND_HALF_UP},
        {"floor", luojianet_ms::NearestMode::FLOOR},
        {"ceil", luojianet_ms::NearestMode::CEIL},
      };
      if (nearest_mode.find(onnx_node_attr.s()) != nearest_mode.end()) {
        prim->set_nearest_mode(nearest_mode[onnx_node_attr.s()]);
      } else {
        MS_LOG(ERROR) << "Unsupported nearest mode: " << attribute_name;
        return nullptr;
      }
    }
  }

  return prim.release();
}

OnnxNodeRegistrar g_onnxResizeParser("Resize", new OnnxResizeParser());
}  // namespace lite
}  // namespace luojianet_ms
