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

#ifndef GE_GRAPH_PASSES_COMPILE_NODES_PASS_H_
#define GE_GRAPH_PASSES_COMPILE_NODES_PASS_H_

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "inc/graph_pass.h"
#include "init/gelib.h"

namespace ge {
///
/// compile nodes
///
class CompileNodesPass : public GraphPass {
 public:
  CompileNodesPass() {}
  virtual ~CompileNodesPass() {}

  graphStatus Run(ComputeGraphPtr graph) override;

 private:
  graphStatus GetSupportedKernel(const NodePtr &node, const std::shared_ptr<GELib> instance, string &kernel_lib_name);
  bool CheckAccuracySupport(const OpsKernelInfoStorePtr &kernel_info, const std::shared_ptr<GELib> instance,
                            const NodePtr &node, string& unsupported_reason);
  graphStatus CompileNodes(const std::shared_ptr<GELib> instance,
                           std::unordered_map<string, vector<NodePtr>> &kernel_to_compile_nodes);
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_COMPILE_NODES_PASS_H_
