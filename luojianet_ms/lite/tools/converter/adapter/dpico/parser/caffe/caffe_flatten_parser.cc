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

#include "parser/caffe/caffe_flatten_parser.h"
#include <memory>
#include "common/op_attr.h"
#include "ops/flatten.h"

namespace luojianet_ms {
namespace lite {
ops::PrimitiveC *CaffeFlattenParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_unique<ops::Flatten>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  const caffe::FlattenParameter &flatten_param = proto.flatten_param();

  if (flatten_param.has_axis()) {
    prim->AddAttr(dpico::kStartAxis, MakeValue<int32_t>(flatten_param.axis()));
  }

  if (flatten_param.has_end_axis()) {
    prim->AddAttr(dpico::kEndAxis, MakeValue<int32_t>(flatten_param.end_axis()));
  }

  return prim.release();
}

CaffeNodeRegistrar g_CaffeFlattenParser("Flatten", new CaffeFlattenParser());
}  // namespace lite
}  // namespace luojianet_ms
