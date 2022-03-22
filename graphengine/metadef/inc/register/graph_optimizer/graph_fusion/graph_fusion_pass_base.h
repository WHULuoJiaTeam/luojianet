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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_PASS_BASE_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_PASS_BASE_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include "register/graph_optimizer/graph_fusion/graph_pass.h"

using std::initializer_list;
using std::map;
using std::string;
using std::vector;

using namespace std;

namespace fe {
enum GraphFusionPassType {
  BUILT_IN_GRAPH_PASS = 0,
  BUILT_IN_VECTOR_CORE_GRAPH_PASS,
  CUSTOM_AI_CORE_GRAPH_PASS,
  CUSTOM_VECTOR_CORE_GRAPH_PASS,
  SECOND_ROUND_BUILT_IN_GRAPH_PASS,
  BUILT_IN_BEFORE_TRANSNODE_INSERTION_GRAPH_PASS,
  GRAPH_FUSION_PASS_TYPE_RESERVED
};
class PatternFusionBasePassImpl;
using PatternFusionBasePassImplPtr = std::shared_ptr<PatternFusionBasePassImpl>;

/** Pass based on pattern
 * @ingroup FUSION_PASS_GROUP
 * @note New virtual methods should be append at the end of this class
 */
class GraphFusionPassBase : public GraphPass {
 public:
  using OpDesc = FusionPattern::OpDesc;
  using Mapping = std::map<const std::shared_ptr<OpDesc>, std::vector<ge::NodePtr>>;
  using Mappings = std::vector<Mapping>;

  GraphFusionPassBase();
  virtual ~GraphFusionPassBase();

  /** execute pass
   *
   * @param [in] graph, the graph waiting for pass level optimization
   * @return SUCCESS, successfully optimized the graph by the pass
   * @return NOT_CHANGED, the graph did not change
   * @return FAILED, fail to modify graph
   */
  Status Run(ge::ComputeGraph &graph) override;

 protected:
  /** define pattern
   *
   * @return NA
   */
  virtual std::vector<FusionPattern *> DefinePatterns() = 0;

  /** do fusion according to nodes matched
   *
   * @param graph the graph waiting for pass level optimization
   * @param new_nodes fusion result node(s)
   * @return SUCCESS, successfully optimized the graph by the pass
   * @return NOT_CHANGED, the graph did not change
   * @return FAILED, fail to modify graph
   */
  virtual Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, std::vector<ge::NodePtr> &new_nodes) = 0;

  /** get nodes from matched result
   *
   * @param mapping match result
   * @return nodes result
   */
  static ge::NodePtr GetNodeFromMapping(const std::string &id, const Mapping &mapping);

 private:
  /** match all nodes in graph according to pattern
   *
   * @param pattern fusion pattern defined
   * @param mappings match result
   * @return SUCCESS, successfully add edge
   * @return FAILED, fail
   */
  bool MatchAll(ge::ComputeGraph &graph, const FusionPattern &pattern, Mappings &mappings);

  Status RunOnePattern(ge::ComputeGraph &graph, const FusionPattern &pattern, bool &changed);  // lint !e148

  /** Internal implement class ptr */
  std::shared_ptr<PatternFusionBasePassImpl> pattern_fusion_base_pass_impl_ptr_;
};

}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_FUSION_PASS_BASE_H_
