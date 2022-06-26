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
#ifndef LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_MODEL_PARSER_H_
#define LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_MODEL_PARSER_H_

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <unordered_map>
#include "include/registry/model_parser.h"
#include "include/registry/model_parser_registry.h"
#include "proto/caffe.pb.h"
#include "ops/primitive_c.h"

using STATUS = int;
namespace luojianet_ms::lite {
class CaffeModelParser : public converter::ModelParser {
 public:
  CaffeModelParser();

  ~CaffeModelParser() override;

  api::FuncGraphPtr Parse(const converter::ConverterParameters &flag) override;

 private:
  STATUS InitOriginModel(const std::string &model_file, const std::string &weight_file);

  STATUS ConvertGraphInputs();

  STATUS ConvertGraphOutputs();

  STATUS ConvertLayers();

  static STATUS ConvertLayerQuantParams(const caffe::LayerParameter &layer, const caffe::LayerParameter &weight,
                                        PrimitiveCPtr primitive_c);

  STATUS ConvertBlobs(const caffe::LayerParameter &layer, std::vector<ParameterPtr> *const_parameters);

  STATUS ConvertBottom(const caffe::LayerParameter &layer, std::vector<AnfNodePtr> *input_nodes);

  STATUS ConvertTop(const caffe::LayerParameter &layer, const CNodePtr &cnode);

  std::string GetOriginLayerName(const std::string &layer_name);

  STATUS ConvertGraphInputsOfLayer();

  STATUS ConvertGraphInputsOfDim();

  STATUS ConvertGraphInputsOfShape();

  caffe::NetParameter caffe_model_;
  caffe::NetParameter caffe_weight_;
  std::unordered_map<std::string, caffe::LayerParameter> caffe_layers_;
  std::unordered_map<std::string, AnfNodePtr> nodes_;
};
}  // namespace luojianet_ms::lite

#endif  // LUOJIANET_MS_LITE_TOOLS_CONVERTER_PARSER_CAFFE_CAFFE_MODEL_PARSER_H_
