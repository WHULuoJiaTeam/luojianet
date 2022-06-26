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

#ifndef LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_OPT_H_
#define LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_OPT_H_

#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <functional>

#include "base/base.h"
#include "ir/manager.h"
#include "utils/hash_map.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "frontend/optimizer/optimizer_caller.h"
#include "frontend/operator/ops.h"

namespace luojianet_ms {
/* namespace to support opt */
namespace opt {

// Define the interaction mode between an Optimize pass and Renormalize pass
// FORCE_RENORM: if the pass modified the graph then the next Renormalize will be executed
// CHECK_RENORM: check if the new node is un-typed to decide if the next Renormalize will be executted
enum RenormAction : int64_t { FORCE_RENORM = 0, CHECK_RENORM };

class Substitution {
 public:
  OptimizerCallerPtr transform_;
  std::string name_;
  PredicateFuncType predicate_{nullptr};
  // An enum to mark this Substitution relation to renormalize pass.
  RenormAction renorm_action_;
  // Determine whether it is a priority substitution, that is, some patterns need to be matched prior to others.
  bool has_priority_pattern_{false};

  Substitution(const OptimizerCallerPtr &transform, const std::string &name, const PredicateFuncType &predicate,
               const RenormAction &renorm_action, bool has_priority_pattern)
      : transform_(transform),
        name_(name),
        predicate_(predicate),
        renorm_action_(renorm_action),
        has_priority_pattern_(has_priority_pattern) {}
  ~Substitution() = default;
  AnfNodePtr operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node);
};

using SubstitutionPtr = std::shared_ptr<Substitution>;

SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name, const PrimitivePtr &prim,
                                 const RenormAction &action_renorm = CHECK_RENORM, bool has_priority_pattern = false);
SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name,
                                 const std::vector<PrimitivePtr> &prims,
                                 const RenormAction &action_renorm = CHECK_RENORM, bool has_priority_pattern = false);
SubstitutionPtr MakeSubstitution(const OptimizerCallerPtr &transform, const std::string &name,
                                 const PredicateFuncType &predicate, const RenormAction &action_renorm = CHECK_RENORM,
                                 bool has_priority_pattern = false);

enum OptTraverseSubstitutionsMode { kOptTraverseFromIRToSubstitutions = 0, kOptTraverseFromSubstitutionsToIR };

class SubstitutionList {
 public:
  explicit SubstitutionList(const std::vector<SubstitutionPtr> &patterns, bool is_once = false,
                            bool global_sensitive = false)
      : list_(patterns), is_once_(is_once), global_sensitive_(global_sensitive) {}
  ~SubstitutionList() = default;

  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) const;

 private:
  bool ApplyIRToSubstitutions(const OptimizerPtr &optimizer, const FuncGraphPtr &func_graph) const;
  bool ApplySubstitutionToIR(const OptimizerPtr &optimizer, const FuncGraphPtr &func_graph,
                             const SubstitutionPtr &sub) const;
  bool ApplySubstitutionsToIR(const OptimizerPtr &optimizer, const FuncGraphPtr &func_graph) const;
  void DisplayStatusOfSubstitution(const luojianet_ms::HashMap<std::string, std::vector<bool>> &status,
                                   const OptimizerPtr &optimizer, size_t space) const;

  std::vector<SubstitutionPtr> list_;
  // a flag to mark this list of Substitution can only be executed only once
  bool is_once_{false};
  bool global_sensitive_{false};
};

// SimpleRewriter simply rewrites a graph according to the node rewriter defined by derived class.
class SimpleRewriter {
 public:
  SimpleRewriter(const FuncGraphPtr &root_graph, const FuncGraphManagerPtr &manager)
      : root_graph_(root_graph), manager_(manager) {}
  virtual ~SimpleRewriter() = default;
  bool Run();

 protected:
  virtual AnfNodePtr NodeRewrite(const AnfNodePtr &node) = 0;
  FuncGraphPtr root_graph_;
  FuncGraphManagerPtr manager_;
};
}  // namespace opt
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_FRONTEND_OPTIMIZER_OPT_H_
