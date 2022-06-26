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
#ifndef LUOJIANET_MS_CCSRC_CXX_API_GRAPH_GRAPH_DATA_H
#define LUOJIANET_MS_CCSRC_CXX_API_GRAPH_GRAPH_DATA_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "include/api/graph.h"
#include "include/api/types.h"
#include "include/dataset/execute.h"
#include "ir/func_graph.h"

namespace luojianet_ms {
class Graph::GraphData {
 public:
  GraphData();

  explicit GraphData(const FuncGraphPtr &func_graph, enum ModelType model_type = kMindIR);

  GraphData(const Buffer &om_data, enum ModelType model_type);

  ~GraphData();

  enum ModelType ModelType() const { return model_type_; }

  FuncGraphPtr GetFuncGraph() const;

  Buffer GetOMData() const;

  void SetPreprocess(const std::vector<std::shared_ptr<dataset::Execute>> &data_graph);

  std::vector<std::shared_ptr<dataset::Execute>> GetPreprocess() { return data_graph_; }

 private:
  FuncGraphPtr func_graph_;
  Buffer om_data_;
  enum ModelType model_type_;
  std::vector<std::shared_ptr<dataset::Execute>> data_graph_;
};
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_CXX_API_GRAPH_GRAPH_DATA_H
