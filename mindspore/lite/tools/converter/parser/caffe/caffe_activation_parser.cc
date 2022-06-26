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

#include "tools/converter/parser/caffe/caffe_activation_parser.h"
#include <memory>
#include "ops/fusion/activation.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr CaffeReluParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::RELU);
  prim->set_min_val(0);
  prim->set_max_val(FLT_MAX);

  if (proto.has_relu_param() && proto.relu_param().has_negative_slope()) {
    float negative_slope = proto.relu_param().negative_slope();
    if (negative_slope != 0) {
      prim->set_activation_type(mindspore::ActivationType::LEAKY_RELU);
      prim->set_alpha(negative_slope);
    }
    if (negative_slope > 0) {
      prim->set_min_val(-FLT_MAX);
    }
  }

  return prim->GetPrim();
}

PrimitiveCPtr CaffeRelu6Parser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::RELU6);
  prim->set_min_val(0);
  prim->set_max_val(kValueThreshold6);

  return prim->GetPrim();
}

PrimitiveCPtr CaffeSigmoidParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::SIGMOID);

  return prim->GetPrim();
}

PrimitiveCPtr CaffeTanhParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::TANH);

  return prim->GetPrim();
}

PrimitiveCPtr CaffeEluParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Activation>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_activation_type(mindspore::ActivationType::ELU);

  if (proto.has_elu_param()) {
    const caffe::ELUParameter &eluParameter = proto.elu_param();
    if (eluParameter.has_alpha()) {
      prim->set_alpha(eluParameter.alpha());
    }
  }

  return prim->GetPrim();
}

CaffeNodeRegistrar g_caffeReluParser("ReLU", new CaffeReluParser());
CaffeNodeRegistrar g_caffeRelu6Parser("ReLU6", new CaffeRelu6Parser());
CaffeNodeRegistrar g_caffeSigmoidParser("Sigmoid", new CaffeSigmoidParser());
CaffeNodeRegistrar g_caffeTanhParser("TanH", new CaffeTanhParser());
CaffeNodeRegistrar g_caffeEluParser("Elu", new CaffeEluParser());
}  // namespace lite
}  // namespace mindspore
