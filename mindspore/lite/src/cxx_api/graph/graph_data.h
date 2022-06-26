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

#ifndef MINDSPORE_LITE_SRC_CXX_API_GRAPH_GRAPH_DATA_H_
#define MINDSPORE_LITE_SRC_CXX_API_GRAPH_GRAPH_DATA_H_

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/graph.h"
#include "include/api/types.h"
#include "src/lite_model.h"

namespace mindspore {
class Graph::GraphData {
 public:
  GraphData() : lite_model_(nullptr) {}

  explicit GraphData(std::shared_ptr<lite::Model> model) : lite_model_(model) {}

  ~GraphData() = default;

  std::shared_ptr<lite::Model> lite_model() { return lite_model_; }

  bool IsTrainModel() const { return true; }

 private:
  std::shared_ptr<lite::Model> lite_model_ = nullptr;
  std::string file_name_ = "";
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_GRAPH_GRAPH_DATA_H_
