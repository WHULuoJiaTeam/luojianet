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
#ifndef MINDSPORE_INCLUDE_API_GRAPH_H
#define MINDSPORE_INCLUDE_API_GRAPH_H

#include <cstddef>
#include <vector>
#include <map>
#include <memory>
#include "include/api/status.h"
#include "include/api/types.h"

namespace mindspore {
class MS_API Graph {
 public:
  class GraphData;
  Graph();
  explicit Graph(const std::shared_ptr<GraphData> &graph_data);
  explicit Graph(std::shared_ptr<GraphData> &&graph_data);
  explicit Graph(std::nullptr_t);
  ~Graph();

  enum ModelType ModelType() const;
  bool operator==(std::nullptr_t) const;
  bool operator!=(std::nullptr_t) const;

 private:
  friend class GraphCell;
  friend class ModelImpl;
  std::shared_ptr<GraphData> graph_data_;
};
}  // namespace mindspore
#endif  // MINDSPORE_INCLUDE_API_GRAPH_H
