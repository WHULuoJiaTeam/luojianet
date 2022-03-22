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

#ifndef GE_GRAPH_PASSES_LABEL_MAKER_H_
#define GE_GRAPH_PASSES_LABEL_MAKER_H_

#include <vector>

#include "graph/node.h"
#include "graph/label/label_maker_factory.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
class LabelMaker {
 public:
  LabelMaker(const ComputeGraphPtr &graph, const NodePtr &owner) : parent_node_(owner), parent_graph_(graph) {}

  virtual ~LabelMaker() {
    parent_node_ = nullptr;
    parent_graph_ = nullptr;
  }

  virtual Status Run(uint32_t &label_index) = 0;

  NodePtr AddStreamActive(const ComputeGraphPtr &graph, const std::string &name);

  NodePtr AddLabelSetEnter(const ComputeGraphPtr &graph, const std::string &name, uint32_t index,
                           NodePtr &stream_active);
  NodePtr AddLabelSetLeave(const ComputeGraphPtr &graph, const std::string &name, uint32_t index);

  NodePtr AddLabelGotoEnter(const ComputeGraphPtr &graph, const std::string &name, uint32_t index);
  NodePtr AddLabelGotoLeave(const ComputeGraphPtr &graph, const std::string &name, uint32_t index);

  NodePtr AddLabelSwitchEnter(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                              const std::vector<uint32_t> &labels);
  NodePtr AddLabelSwitchLeave(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                              const std::vector<uint32_t> &labels);

  NodePtr AddLabelSwitchIndex(const ComputeGraphPtr &graph, const std::string &name, const GeTensorDesc &desc,
                              const NodePtr &sw_node, uint32_t parent_index);

  LabelMaker &operator=(const LabelMaker &model) = delete;
  LabelMaker(const LabelMaker &model) = delete;

 protected:
  NodePtr parent_node_;
  ComputeGraphPtr parent_graph_;

 private:
  void LinkToGraphHead(const ComputeGraphPtr &graph, const NodePtr &node);
  void LinkToGraphTail(const ComputeGraphPtr &graph, const NodePtr &node);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_LABEL_MAKER_H_
