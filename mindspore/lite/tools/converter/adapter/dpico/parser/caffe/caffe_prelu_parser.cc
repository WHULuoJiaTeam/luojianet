/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "parser/caffe/caffe_prelu_parser.h"
#include <memory>
#include "ops/fusion/prelu_fusion.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffePReluParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::PReLUFusion>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  const caffe::PReLUParameter &prelu_param = proto.prelu_param();
  if (prelu_param.has_channel_shared()) {
    prim->set_channel_shared(prelu_param.channel_shared());
  } else {
    prim->set_channel_shared(false);
  }

  return prim;
}

CaffeNodeRegistrar g_caffePReluParser("PReLU", new CaffePReluParser());
}  // namespace lite
}  // namespace mindspore
