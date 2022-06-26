/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_H_
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "ir/func_graph.h"
#include "backend/common/optimizer/pass.h"

namespace mindspore::graphkernel {
struct StitchInfo {
  std::vector<std::string> stitch_ops;
  std::vector<std::string> stitch_atomic_ops;
};

class SplitNodesDecoder {
 public:
  SplitNodesDecoder() {}
  ~SplitNodesDecoder() = default;
  static bool DecodeSplitNodes(const nlohmann::json &kernel_json,
                               const std::map<std::string, AnfNodePtr> &address_node_map, AnfNodePtrList *res_graphs);
};

class SplitSchemer {
 public:
  SplitSchemer() = default;
  virtual ~SplitSchemer() = default;
  virtual bool Split(const FuncGraphPtr &func_graph) = 0;
  virtual bool NeedInline(size_t) const { return false; }
  const std::vector<AnfNodePtrList> &split_plan() const { return split_plan_; }

 protected:
  std::vector<AnfNodePtrList> split_plan_;
};

class GraphKernelSplitter : public opt::Pass {
 public:
  GraphKernelSplitter() : Pass("graph_kernel_splitter") {}
  ~GraphKernelSplitter() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
  bool TrySplit(const CNodePtr &sub_root_cnode);
  virtual std::shared_ptr<SplitSchemer> GetSplitSchema(const std::string &processor);
};
using GraphKernelSplitterPtr = std::shared_ptr<GraphKernelSplitter>;
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_CORE_GRAPH_KERNEL_SPLITTER_H_
