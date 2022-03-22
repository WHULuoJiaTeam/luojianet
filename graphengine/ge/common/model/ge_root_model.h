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
#include <map>
#include "graph/compute_graph.h"
#include "common/model/ge_model.h"

#ifndef GE_MODEL_GE_ROOT_MODEL_H_
#define GE_MODEL_GE_ROOT_MODEL_H_

namespace ge {
class GeRootModel {
 public:
  GeRootModel() = default;
  explicit GeRootModel(ComputeGraphPtr &root_graph) : root_graph_(root_graph), model_id_(INVALID_MODEL_ID) {};
  ~GeRootModel() = default;

  void SetSubgraphInstanceNameToModel(string instance_name, GeModelPtr ge_model);
  const std::map<std::string, GeModelPtr> &GetSubgraphInstanceNameToModel() const {
    return subgraph_instance_name_to_model_;
  };

  const ComputeGraphPtr &GetRootGraph() const { return root_graph_; }
  void SetModelId(uint32_t model_id) {
    model_id_ = model_id;
    // cached for removement
    model_ids_.emplace_back(model_id);
  }
  uint32_t GetModelId() const { return model_id_; }

  void SetIsSpecificStream(bool is_specific_stream) { is_specific_stream_ = is_specific_stream; }

  bool IsSpecificStream() const { return is_specific_stream_; }

  void SetModelName(const std::string &model_name) { model_name_ = model_name; }

  const std::string &GetModelName() const { return model_name_; }

  std::vector<uint32_t> GetAllModelId() const { return model_ids_; }

  void ClearAllModelId() { model_ids_.clear(); }

  Status CheckIsUnknownShape(bool &is_dynamic_shape);

  void SetRootGraph(ComputeGraphPtr graph) { root_graph_ = graph; }

  void SetTrainFlag(bool flag) { train_flag_ = flag; }

  bool GetTrainFlag() const { return train_flag_; }

 private:
  ComputeGraphPtr root_graph_ = nullptr;
  std::map<std::string, GeModelPtr> subgraph_instance_name_to_model_;
  uint32_t model_id_ = 0;
  // In multithread online secenario, same graph can owns different davinci_model for for concurrency
  std::vector<uint32_t> model_ids_;
  bool train_flag_ = false;
  std::string model_name_;
  bool is_specific_stream_ = false;
};
}  // namespace ge
using GeRootModelPtr = std::shared_ptr<ge::GeRootModel>;
#endif  // GE_MODEL_GE_ROOT_MODEL_H_
