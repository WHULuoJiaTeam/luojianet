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

#ifndef REGISTER_SCOPE_SCOPE_PASS_IMPL_H_
#define REGISTER_SCOPE_SCOPE_PASS_IMPL_H_

#include "external/register/scope/scope_fusion_pass_register.h"

namespace ge {
class ScopesResult::ScopesResultImpl {
 public:
  void SetScopes(const std::vector<Scope *> &scopes) { scopes_ = scopes; }
  const std::vector<Scope *> &GetScopes() const { return scopes_; }
  void SetNodes(const std::vector<ge::OperatorPtr> &nodes) { nodes_ = nodes; }
  const std::vector<ge::OperatorPtr> &GetNodes() const { return nodes_; }

 private:
  std::vector<Scope *> scopes_;  // multiple scopes
  std::vector<ge::OperatorPtr> nodes_;  // op outside of scope
};

class ScopeBasePass::ScopeBasePassImpl {
 public:
  explicit ScopeBasePassImpl(ScopeBasePass *parent) : parent_(parent) {}
  virtual ~ScopeBasePassImpl();

  Status Run(std::shared_ptr<ScopeGraph> &scope_graph);

 private:
  Status AddFusionScopesResultToScopeGraph(std::shared_ptr<ScopeGraph> &scope_graph,
                                           std::vector<ScopesResult> &scope_results);
  // Match rules one by one, support multiple sets of matching rules, and finally output a single scope
  // Note: This function does not have to be rewritten.
  //       In order to match the fusion rules designed by you better,
  //       you can implement your specific versions separately.
  bool MatchAllBatches(const ScopeTree *scope_tree, std::vector<Scope *> &results);

  bool MatchOneBatch(const ScopeTree *scope_tree, const std::vector<ScopePattern *> &patternlist,
                     std::vector<Scope *> &results);
  bool MatchOneScope(const ScopePattern *pattern, Scope *scope, std::vector<Scope *> &results);
  Status PrintFusionScopeInfo(std::shared_ptr<ScopeGraph> &scope_graph);

 private:
  std::vector<ScopeFusionPatterns> patterns_;
  ScopeBasePass *parent_;
};
}  // namespace ge
#endif  // REGISTER_SCOPE_SCOPE_PASS_IMPL_H_