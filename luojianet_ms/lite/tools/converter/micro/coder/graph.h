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

#ifndef LUOJIANET_MS_LITE_MICRO_CODER_GRAPH_H_
#define LUOJIANET_MS_LITE_MICRO_CODER_GRAPH_H_

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include "tools/converter/micro/coder/config.h"
#include "include/context.h"
#include "include/model.h"
#include "schema/inner/model_generated.h"
#include "src/common/graph_util.h"
#include "src/tensor.h"

namespace luojianet_ms::lite::micro {
class CoderGraph {
 public:
  explicit CoderGraph(Model *model) : model_(model) {}
  ~CoderGraph();

  int ConvertTensors();
  int InitGraphInOutTensors();

  void SetAllTensors(const std::vector<Tensor *> &all_tensors);

  void InitInputs();
  void InitOutputs();

  void SetInputIndices(const std::vector<uint32_t> &input_indices);

  void SetOutputIndices(const std::vector<uint32_t> &output_indices);

  void AddInputMap(const std::string &node_id, Tensor *input_tensor);

  void AddOutputMap(const std::string &node_id, Tensor *output_tensor);

  std::vector<uint32_t> input_indices() const;

  std::vector<uint32_t> output_indices() const;

  std::vector<Tensor *> input_tensors() const;

  std::vector<Tensor *> output_tensors() const;

  std::vector<Tensor *> all_tensors() const;

  const std::map<NODE_ID, std::vector<Tensor *>> &GetOutputsMap() const;

  const Model *model() const { return this->model_; }

  void DumpUnSupportLayer(Target target);

 private:
  // graph_inputs && weight && bias is value_node
  // others are parameter_node
  std::vector<Tensor *> all_tensors_;

  std::vector<Tensor *> input_tensors_;

  std::vector<Tensor *> output_tensors_;

  std::vector<uint32_t> input_indices_;

  std::vector<uint32_t> output_indices_;

  std::map<std::string, std::vector<Tensor *>> inputs_map_;

  std::map<std::string, std::vector<Tensor *>> outputs_map_;

  Model *model_{nullptr};
};
}  // namespace luojianet_ms::lite::micro
#endif  // LUOJIANET_MS_LITE_MICRO_CODER_GRAPH_H_
