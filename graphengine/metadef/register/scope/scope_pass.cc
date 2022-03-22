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

#include "register/scope/scope_pass_impl.h"
#include <memory>
#include <stack>
#include "register/scope/scope_graph_impl.h"
#include "register/scope/scope_pattern_impl.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_util.h"

namespace ge {
ScopesResult::ScopesResult() {
  impl_ = std::unique_ptr<ScopesResultImpl>(new (std::nothrow) ScopesResultImpl);
}

ScopesResult::ScopesResult(ScopesResult const &result) {
  impl_ = std::unique_ptr<ScopesResultImpl>(new (std::nothrow) ScopesResultImpl);
  if (impl_ == nullptr || result.impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "ScopesResult is not properly initialized.");
    return;
  }
  const std::vector<Scope *> &scopes = result.impl_->GetScopes();
  const std::vector<ge::OperatorPtr> &nodes = result.impl_->GetNodes();
  impl_->SetScopes(scopes);
  impl_->SetNodes(nodes);
}
ScopesResult &ScopesResult::operator=(ScopesResult const &result) {
  if (&result == this) {
    return *this;
  }
  if (impl_ == nullptr || result.impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "ScopesResult is not properly initialized.");
    return *this;
  }
  const std::vector<Scope *> &scopes = result.impl_->GetScopes();
  const std::vector<ge::OperatorPtr> &nodes = result.impl_->GetNodes();
  impl_->SetScopes(scopes);
  impl_->SetNodes(nodes);
  return *this;
}

ScopesResult::~ScopesResult() {}

void ScopesResult::SetScopes(std::vector<Scope *> &scopes) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetScopes(), ScopesResult is not properly initialized.");
    return;
  }

  impl_->SetScopes(scopes);
}

void ScopesResult::SetNodes(std::vector<ge::OperatorPtr> &nodes) {
  if (impl_ == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "Failed to invoke SetNodes(), ScopesResult is not properly initialized.");
    return;
  }

  impl_->SetNodes(nodes);
}

ScopeBasePass::ScopeBasePassImpl::~ScopeBasePassImpl() {
  for (auto &scope_patterns : patterns_) {
    for (auto &batch_patterns : scope_patterns) {
      for (auto &pattern : batch_patterns) {
        if (pattern != nullptr) {
          delete pattern;
          pattern = nullptr;
        }
      }
    }
  }
}

Status ScopeBasePass::ScopeBasePassImpl::AddFusionScopesResultToScopeGraph(std::shared_ptr<ScopeGraph> &scope_graph,
                                                                           std::vector<ScopesResult> &scope_results) {
  for (auto &rlt : scope_results) {
    FusionScopesResult *fusion_rlt = new (std::nothrow) FusionScopesResult();
    if (fusion_rlt == nullptr) {
      GELOGE(FAILED, "Alloc fusion_rlt failed.");
      return FAILED;
    }
    if (fusion_rlt->Init() != SUCCESS) {
      GELOGE(FAILED, "Init fusion_rlt failed.");
      delete fusion_rlt;
      fusion_rlt = nullptr;
      return FAILED;
    }
    auto &impl_fusion_rlt = fusion_rlt->impl_;
    auto &impl_scope_rlt = rlt.impl_;
    if (impl_scope_rlt == nullptr) {
      GELOGE(ge::MEMALLOC_FAILED, "ScopesResult is not properly initialized.");
      delete fusion_rlt;
      fusion_rlt = nullptr;
      continue;
    }

    impl_fusion_rlt->AddNodes(impl_scope_rlt->GetNodes());
    impl_fusion_rlt->AddScopes(impl_scope_rlt->GetScopes());

    parent_->GenerateFusionResult(impl_scope_rlt->GetScopes(), fusion_rlt);
    if (impl_fusion_rlt->Type() == kScopeInvalidType) {
      GELOGE(FAILED, "Failed to set inner node for fusion op %s.", impl_fusion_rlt->Type().c_str());
      delete fusion_rlt;
      return FAILED;
    }
    auto &impl_scope_graph = scope_graph->impl_;
    impl_scope_graph->AddFusionScopesResult(fusion_rlt);
  }

  return SUCCESS;
}

Status ScopeBasePass::ScopeBasePassImpl::Run(std::shared_ptr<ScopeGraph> &scope_graph) {
  GE_CHECK_NOTNULL(scope_graph);
  const ScopeTree *const scope_tree = scope_graph->GetScopeTree();
  GE_CHECK_NOTNULL(scope_tree);
  GE_CHECK_NOTNULL(parent_);
  patterns_ = parent_->DefinePatterns();
  std::vector<Scope *> results;
  if (!MatchAllBatches(scope_tree, results)) {
    GELOGI("[scope_fusion] Scope pass %s's patterns is not matched and ignored.", parent_->PassName().c_str());
    return domi::SCOPE_NOT_CHANGED;
  }
  GELOGI("[scope_fusion] Scope pass %s's patterns is matched.", parent_->PassName().c_str());

  std::vector<ScopesResult> scope_results;
  Status ret = parent_->LastMatchScopesAndOPs(scope_graph, scope_results);
  if (ret != SUCCESS) {
    for (auto &result : results) {
      GE_CHECK_NOTNULL(result);
      auto &impl_scope = result->impl_;
      impl_scope->ClearTypeAndSubType();
    }
    GELOGW("[ScopeFusion][RunPass] Scope pass %s's patterns is ignored, because LastMatchScopesAndOPs failed.",
           parent_->PassName().c_str());
    return domi::SCOPE_NOT_CHANGED;
  }

  if (!results.empty()) {
    ret = AddFusionScopesResultToScopeGraph(scope_graph, scope_results);
    if (ret != SUCCESS) {
      GELOGE(FAILED, "Scope pass %s add fusion scopes result to scope graph failed.", parent_->PassName().c_str());
      return FAILED;
    }
  } else {
    GELOGI("[scope_fusion] Scope pass %s not match any scope.", parent_->PassName().c_str());
  }

  ret = PrintFusionScopeInfo(scope_graph);
  if (ret != SUCCESS) {
    GELOGI("[scope_fusion] Print scope pass %s fusion info failed.", parent_->PassName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

bool ScopeBasePass::ScopeBasePassImpl::MatchAllBatches(const ScopeTree *scope_tree, std::vector<Scope *> &results) {
  if (scope_tree == nullptr) {
    GELOGE(PARAM_INVALID, "Input param [scope_tree] is nullptr.");
    return false;
  }

  for (auto &scope_patterns : patterns_) {
    std::vector<Scope *> tmp_results;
    std::vector<Scope *> last_results;
    uint32_t batch_num = 0U;
    for (auto &batch_patterns : scope_patterns) {
      ++batch_num;
      std::vector<Scope *> one_results;
      const bool is_matched = MatchOneBatch(scope_tree, batch_patterns, one_results);
      if (!is_matched) {
        break;
      }
      if (batch_num == scope_patterns.size()) {
        (void)last_results.insert(last_results.end(), one_results.begin(), one_results.end());
      } else {
        (void)tmp_results.insert(tmp_results.end(), one_results.begin(), one_results.end());
      }
    }
    for (auto &tmp : tmp_results) {
      bool rollback = true;
      for (auto &result : last_results) {
        if ((result->Name().length() <= tmp->Name().length()) && (tmp->Name().find(result->Name()) == 0U)) {
          rollback = false;
          break;
        }
      }
      if (rollback) {
        auto &impl = tmp->impl_;
        impl->SetSubType("");
      }
    }
    (void)results.insert(results.end(), last_results.begin(), last_results.end());
  }

  return !(results.empty());
}

bool ScopeBasePass::ScopeBasePassImpl::MatchOneBatch(const ScopeTree *scope_tree,
                                                     const std::vector<ScopePattern *> &patternlist,
                                                     std::vector<Scope *> &results) {
  if (scope_tree == nullptr) {
    GELOGE(PARAM_INVALID, "Input param [scope_tree] is nullptr");
    return false;
  }

  int32_t find = 0;
  auto &impl_scope_tree = scope_tree->impl_;
  const Scope *const root = impl_scope_tree->Root();
  if (root != nullptr) {
    auto &impl_scope = root->impl_;
    const std::unordered_map<std::string, Scope *> &sub_scopes = impl_scope->GetSubScopes();
    for (auto &pattern : patternlist) {
      for (auto &scope : sub_scopes) {
        if (MatchOneScope(pattern, scope.second, results)) {
          ++find;
        }
      }
    }
  }

  return (find > 0) ? true : false;
}

bool ScopeBasePass::ScopeBasePassImpl::MatchOneScope(const ScopePattern *pattern, Scope *scope,
                                                     std::vector<Scope *> &results) {
  if ((pattern == nullptr) || (scope == nullptr)) {
    GELOGE(PARAM_INVALID, "Input param is nullptr");
    return false;
  }
  auto &impl_scope_pattern = pattern->impl_;
  if (impl_scope_pattern == nullptr) {
    GELOGE(ge::MEMALLOC_FAILED, "ScopePattern is not properly initialized.");
    return false;
  }
  if (impl_scope_pattern->Match(scope)) {
    auto &scope_impl = scope->impl_;
    scope_impl->SetSubType(impl_scope_pattern->SubType());
    results.push_back(scope);
    return true;
  }
  int32_t find = 0;
  std::stack<Scope *> scopes;
  scopes.push(scope);
  while (!scopes.empty()) {
    const Scope *const current_scope = scopes.top();
    scopes.pop();
    auto &current_scope_impl = current_scope->impl_;
    const std::unordered_map<std::string, Scope *> &sub_scopes = current_scope_impl->GetSubScopes();
    for (auto &sub_scope : sub_scopes) {
      if (impl_scope_pattern->Match(sub_scope.second)) {
        auto &sub_scope_impl = sub_scope.second->impl_;
        sub_scope_impl->SetSubType(impl_scope_pattern->SubType());
        results.push_back(sub_scope.second);
        ++find;
      } else {
        scopes.push(sub_scope.second);
      }
    }
  }
  return (find > 0) ? true : false;
}

Status ScopeBasePass::ScopeBasePassImpl::PrintFusionScopeInfo(std::shared_ptr<ScopeGraph> &scope_graph) {
  if (scope_graph == nullptr) {
    GELOGE(PARAM_INVALID, "Input param scope_graph is nullptr.");
    return PARAM_INVALID;
  }
  auto &impl_scope_graph = scope_graph->impl_;
  const std::unordered_map<std::string, FusionScopesResult *> &final_results = impl_scope_graph->FusionScopesResults();
  for (auto &result : final_results) {
    if (result.second == nullptr) {
       GELOGE(PARAM_INVALID, "Fusion scope is nullptr.");
       return PARAM_INVALID;
    }
    GELOGI("FusionScope:%s", result.second->Name().c_str());
    auto &impl = result.second->impl_;
    const std::map<std::string, std::vector<int32_t>> &inputs = impl->GetInputs();
    for (auto &input : inputs) {
      const std::vector<int32_t> indexs = input.second;
      for (const int32_t index : indexs) {
        GELOGI("FusionScope input node:%s,%d", input.first.c_str(), index);
      }
    }

    const std::map<std::string, std::vector<int32_t>> &outputs = impl->GetOutputs();
    for (auto &output : outputs) {
      const std::vector<int32_t> indexs = output.second;
      for (const int32_t index : indexs) {
        GELOGI("FusionScope output node:%s,%d", output.first.c_str(), index);
      }
    }

    for (auto &scope : impl->Scopes()) {
      if (scope == nullptr) {
        GELOGE(PARAM_INVALID, "Scope in fusion scope is nullptr.");
        return PARAM_INVALID;
      }
      GELOGI("FusionScope GetScope:%s", scope->Name().c_str());
    }

    for (auto &node : result.second->Nodes()) {
      if (node == nullptr) {
        GELOGE(PARAM_INVALID, "Node in scope is nullptr.");
        return PARAM_INVALID;
      }
      GELOGI("FusionScope Node:%s", node->GetName().c_str());
    }
  }
  return SUCCESS;
}

ScopeBasePass::ScopeBasePass() {
  impl_ = std::unique_ptr<ScopeBasePassImpl>(new (std::nothrow) ScopeBasePassImpl(this));
}

ScopeBasePass::~ScopeBasePass() {}
}  // namespace ge
