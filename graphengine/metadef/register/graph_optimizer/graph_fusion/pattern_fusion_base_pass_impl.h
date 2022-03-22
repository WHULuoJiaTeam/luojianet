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

#ifndef FE_PATTERN_FUSION_BASE_PASS_IMPL_H
#define FE_PATTERN_FUSION_BASE_PASS_IMPL_H

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/opskernel/ops_kernel_info_store.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"

using std::initializer_list;
using std::map;
using std::string;
using std::vector;

using namespace std;

namespace fe {

using OpDesc = FusionPattern::OpDesc;
using Mapping = map<const std::shared_ptr<OpDesc>, vector<ge::NodePtr>>;
using Mappings = std::vector<Mapping>;
using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;

/** Base pattern impl
 * @ingroup FUSION_PASS_GROUP
 * @note New virtual methods should be append at the end of this class
 */
class PatternFusionBasePassImpl {
 public:
  PatternFusionBasePassImpl();

  virtual ~PatternFusionBasePassImpl();

  void GetPatterns(vector<FusionPattern *> &patterns);

  void SetPatterns(vector<FusionPattern *> &patterns);

  void SetOpsKernelInfoStore(OpsKernelInfoStorePtr ops_kernel_info_store_ptr);

  PatternFusionBasePassImpl &operator=(const PatternFusionBasePassImpl &) = delete;

  bool CheckOpSupported(const ge::OpDescPtr &op_desc_ptr);

  bool CheckOpSupported(const ge::NodePtr &node);

  bool IsNodesExist(ge::NodePtr current_node, std::vector<ge::NodePtr> &nodes);

  bool IsMatched(std::shared_ptr<OpDesc> op_desc, const ge::NodePtr node, const Mapping &mapping);

  void DumpMappings(const FusionPattern &pattern, const Mappings &mappings);

  bool IsOpTypeExist(const string &type, const vector<string> &types);

  bool MatchFromOutput(ge::NodePtr output_node, std::shared_ptr<OpDesc> output_op_desc, Mapping &mapping);

  std::string GetNodeType(ge::NodePtr node);

  bool GetMatchOutputNodes(ge::ComputeGraph &graph, const FusionPattern &pattern,
                           vector<ge::NodePtr> &matched_output_nodes);

 private:
  vector<FusionPattern *> patterns_;

  OpsKernelInfoStorePtr ops_kernel_info_store_ptr_;

  bool MatchFromOutput(vector<ge::NodePtr> &candidate_nodes, vector<std::shared_ptr<OpDesc>> &candidate_op_descs,
                       Mapping &mapping);

  bool MatchAllEdges(const size_t &input_size, const std::unique_ptr<bool[]> &usage_flags);

  void GetInDataAnchors(const ge::NodePtr &node, std::vector<ge::InDataAnchorPtr> &in_anchor_vec);
};

}  // namespace fe

#endif  // FE_PATTERN_FUSION_BASE_PASS_H
