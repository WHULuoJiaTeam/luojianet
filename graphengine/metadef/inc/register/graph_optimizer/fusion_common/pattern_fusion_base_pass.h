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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_PATTERN_FUSION_BASE_PASS_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_PATTERN_FUSION_BASE_PASS_H_

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "common/opskernel/ops_kernel_info_store.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include "register/graph_optimizer/graph_fusion/graph_pass.h"
#include "register/graph_optimizer/graph_fusion/connection_matrix.h"

using std::initializer_list;
using std::map;
using std::string;
using std::vector;

using namespace std;

namespace fe {
using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
class PatternFusionBasePassImpl;
using PatternFusionBasePassImplPtr = std::shared_ptr<PatternFusionBasePassImpl>;

/** Pass based on pattern
 * @ingroup FUSION_PASS_GROUP
 * @note New virtual methods should be append at the end of this class
 */
class PatternFusionBasePass : public GraphPass {
 public:
  using OpDesc = FusionPattern::OpDesc;
  using Mapping = std::map<const std::shared_ptr<OpDesc>, std::vector<ge::NodePtr>>;
  using Mappings = std::vector<Mapping>;

  PatternFusionBasePass();
  virtual ~PatternFusionBasePass();

  /** execute pass
   *
   * @param [in] graph, the graph waiting for pass level optimization
   * @return SUCCESS, successfully optimized the graph by the pass
   * @return NOT_CHANGED, the graph did not change
   * @return FAILED, fail to modify graph
   */
  Status Run(ge::ComputeGraph &graph) override;

  /** execute pass
   *
   * @param [in] graph, the graph waiting for pass level optimization
   * @param [ops_kernel_info_store_ptr, OP info kernel instance
   * @return SUCCESS, successfully optimized the graph by the pass
   * @return NOT_CHANGED, the graph did not change
   * @return FAILED, fail to modify graph
   */
  virtual Status Run(ge::ComputeGraph &graph, OpsKernelInfoStorePtr ops_kernel_info_store_ptr);

  /* Detect whether there are cycles in graph
   * after fusing all nodes in param fusion_nodes.
   *
   * Compared with Cycle Detection
   * @param fusion_nodes: each vector in fusion_nodes
   * will be fused into an entity(which could contains
   * more than one node). The caller should put all original
   * nodes which are expected to be fused into one larger node
   * into each sub-vector of fusion_nodes.
   *
   * This function can tell whether there are a cycle after
   * fusing all nodes in fusion_nodes. Each vector in 2-d
   * vector fusion_nodes will be fused into an entity.
   *
   *
   * This interface cannot detect whether there are cycles
   * inside the fused nodes.
   *
   * e.g. {a, b, c, d} -> {e, f}
   * Because the edge information is not given for e and f
   * so this function we cannot tell if e and f are in a
   * cycle.
   * */
  bool CycleDetection(const ge::ComputeGraph &graph,
                      const std::vector<std::vector<ge::NodePtr>> &fusion_nodes);

 protected:
  virtual std::vector<FusionPattern *> DefinePatterns() = 0;
  virtual Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, std::vector<ge::NodePtr> &new_nodes) = 0;

  std::vector<ge::NodePtr> GetNodesFromMapping(const Mapping &mapping);
  ge::NodePtr GetNodeFromMapping(const std::string &id, const Mapping &mapping);

  void RecordOutputAnchorMap(ge::NodePtr output_node);
  void ClearOutputAnchorMap();

  Status SetDataDumpAttr(std::vector<ge::NodePtr> &original_nodes, std::vector<ge::NodePtr> &fus_nodes);

  bool CheckOpSupported(const ge::OpDescPtr &op_desc_ptr);

  bool CheckOpSupported(const ge::NodePtr &node);

  /** check whether the input graph is Cyclic
  *
  *  @param graph need to be checked
  *  @return false or true
  */
  bool CheckGraphCycle(ge::ComputeGraph &graph);

  void EnableNetworkAnalysis();

  void DumpMapping(const FusionPattern &pattern, const Mapping &mapping);
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

  /* Check whether there are cycles after fusing scope_nodes as an
   * entity. The algorithm is:
   * If one of the output node of scope nodes has an edged linked to
   * the scope nodes again, there will be a cycle.
   * e.g.
   *               A
   *             /  \
   *            B    \
   *           /      \
   *          D------->C
   *          |        |
   * After fusion A/B/C, the graph looks like:
   *              <---
   *             /    \
   *           ABC--->D
   * There obviously a cycle in the fused graph.
   *             */
  bool DetectOneScope(const std::vector<ge::NodePtr> &scope_nodes);

  bool CheckEachPeerOut(const ge::NodePtr &node,
                        const std::unordered_set<ge::NodePtr> &scope_nodes_set,
                        const std::vector<ge::NodePtr> &scope_nodes);

  /** Internal implement class ptr */
  std::shared_ptr<PatternFusionBasePassImpl> pattern_fusion_base_pass_impl_ptr_;

  std::unordered_map<ge::NodePtr, std::map<ge::InDataAnchorPtr, ge::OutDataAnchorPtr>> origin_op_anchors_map_;

  /* For detecting cycles, we will only build connectivity once.
   * One time generation of connectivity needs O(n+e) where n is
   * total number of nodes and e is total number of edges, which is
   * not tolerable. And this requires one pass only executed once.
   * */
  std::shared_ptr<ConnectionMatrix> connectivity_{nullptr};

  bool enable_network_analysis_ = false;
};
}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_PATTERN_FUSION_BASE_PASS_H_
