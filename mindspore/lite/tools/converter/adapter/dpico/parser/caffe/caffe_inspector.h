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

#ifndef DPICO_PARSER_CAFFE_INSPECTOR_H_
#define DPICO_PARSER_CAFFE_INSPECTOR_H_

#include <set>
#include <string>
#include <unordered_map>
#include <memory>
#include "./pico_caffe.pb.h"
#include "include/errorcode.h"

namespace mindspore {
namespace lite {
class CaffeInspector {
 public:
  CaffeInspector() = default;
  ~CaffeInspector() = default;

  STATUS InspectModel(const caffe::NetParameter &proto);
  void ParseInput();
  void FindGraphInputsAndOutputs();
  void SetLayerTopsAndBottoms();

  std::set<std::string> GetGraphInput() { return graphInput; }
  std::set<std::string> GetGraphOutput() { return graphOutput; }

 private:
  caffe::NetParameter net;

  std::set<std::string> layerTops;
  std::set<std::string> layerBottoms;

  std::set<std::string> graphInput;
  std::set<std::string> graphOutput;
};

using CaffeInspectorPtr = std::shared_ptr<CaffeInspector>;
}  // namespace lite
}  // namespace mindspore

#endif  // DPICO_PARSER_CAFFE_INSPECTOR_H_
