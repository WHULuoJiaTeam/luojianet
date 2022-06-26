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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_H_

#include <string>
#include <vector>
#include "google/protobuf/message.h"
#include "schema/inner/model_generated.h"
#include "proto/caffe.pb.h"
#include "tools/converter/parser/caffe/caffe_node_parser.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "ops/primitive_c.h"
#include "mindspore/core/utils/check_convert_utils.h"
#include "tools/converter/parser/parser_utils.h"

namespace mindspore {
namespace lite {
class CaffeNodeParser {
 public:
  explicit CaffeNodeParser(const std::string nodeName) : name(nodeName) {}

  virtual ~CaffeNodeParser() {}

  virtual PrimitiveCPtr Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
    return nullptr;
  }

 protected:
  const std::string name;
};

schema::TensorT *ConvertWeight(const caffe::BlobProto &proto);

STATUS ConvertShape(const caffe::BlobProto &proto, std::vector<int32_t> *shape);
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_NODE_PARSER_H_
