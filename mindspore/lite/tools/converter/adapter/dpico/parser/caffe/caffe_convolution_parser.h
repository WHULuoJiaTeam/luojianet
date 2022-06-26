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

#ifndef DPICO_PARSER_CAFFE_CONVOLUTION_PARSER_H_
#define DPICO_PARSER_CAFFE_CONVOLUTION_PARSER_H_

#include <utility>
#include <vector>
#include <string>
#include "parser/caffe/caffe_conv_base_parser.h"

namespace mindspore {
namespace lite {
class CaffeConvolutionParser : public CaffeNodeParser {
 public:
  CaffeConvolutionParser() : CaffeNodeParser("convolution") {}
  explicit CaffeConvolutionParser(std::string nodeName) : CaffeNodeParser(std::move(nodeName)) {}
  ~CaffeConvolutionParser() override = default;

  BaseOperatorPtr Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) override;
};
}  // namespace lite
}  // namespace mindspore

#endif  // DPICO_PARSER_CAFFE_CONVOLUTION_PARSER_H_
