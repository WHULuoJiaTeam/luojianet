/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_SPECIALIZE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_SPECIALIZE_H_

#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unordered_map>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/anf.h"
#include "ir/func_graph_cloner.h"
#include "pipeline/jit/static_analysis/evaluator.h"

namespace mindspore {
namespace abstract {
enum SpecializeStatusCode {
  kSpecializeSuccess = 0,
  kSpecializeDead = 1,  // Dead Node
  kSpecializePoly = 2,  // Poly Node
  kSpecializeFailure = 0xFF
};

class FuncGraphSpecializer;
using BaseFuncGraphEvaluatorPtr = std::shared_ptr<BaseFuncGraphEvaluator>;

// Specialize a func graph using analyzed abstract values.
class ProgramSpecializer {
 public:
  explicit ProgramSpecializer(const std::shared_ptr<AnalysisEngine> &engine) : engine_(engine), top_context_(nullptr) {
    mng_ = engine_->func_graph_manager();
  }
  ~ProgramSpecializer() = default;
  // Run the program specializer on the topmost graph in the given context.
  FuncGraphPtr Run(const FuncGraphPtr &fg, const AnalysisContextPtr &context);
  const mindspore::HashSet<AnfNodePtr> &seen() const { return seen_; }
  void AddSeen(const AnfNodePtr &node) { (void)seen_.insert(node); }

  std::shared_ptr<FuncGraphSpecializer> GetFuncGraphSpecializer(const AnalysisContextPtr &context);
  // Specialze one FuncGraph in a given context.
  FuncGraphPtr SpecializeFuncGraph(const FuncGraphPtr &fg, const AnalysisContextPtr &context);

  std::shared_ptr<AnalysisEngine> engine() { return engine_; }

  const AnalysisContextPtr &top_context() const { return top_context_; }
  void PutSpecializedAbstract(const CNodePtr &cnode, const AnfNodePtr &func, const AbstractFunctionPtr &old_abs_func,
                              const AbstractFunctionPtr &new_abs_func);
  AbstractBasePtr GetSpecializedAbstract(const AbstractFunctionPtr &old_abs_func);
  void SpecializeCNodeInput0FuncGraph();

  std::vector<std::pair<AbstractSequencePtr, AnfNodePtr>> &sequence_abstract_list() { return sequence_abstract_list_; }
  std::vector<std::pair<AnfNodePtr, size_t>> &dead_node_list() { return dead_node_list_; }

 private:
  std::shared_ptr<AnalysisEngine> engine_;
  mindspore::HashSet<AnfNodePtr> seen_;
  FuncGraphManagerPtr mng_;
  std::unordered_map<AnalysisContextPtr, std::shared_ptr<FuncGraphSpecializer>, ContextHasher, ContextEqual>
    specializations_;
  AnalysisContextPtr top_context_;
  // The list to purify tuple/list elements.
  std::vector<std::pair<AbstractSequencePtr, AnfNodePtr>> sequence_abstract_list_;
  // The list to erase the DeadNode in tuple/list elements.
  std::vector<std::pair<AnfNodePtr, size_t>> dead_node_list_;
  // Map for unspecialized abstract function to specialized abstract;
  std::unordered_map<AbstractFunctionPtr, AbstractBasePtr, AbstractFunctionHasher, AbstractFunctionEqual>
    specialized_abs_map_;

  AbstractBasePtr SpecializeAbstractFuncRecursively(const AbstractFunctionPtr &old_abs_func);
};

class FuncGraphSpecializer : public std::enable_shared_from_this<FuncGraphSpecializer> {
 public:
  FuncGraphSpecializer(ProgramSpecializer *const s, const FuncGraphPtr &fg, const AnalysisContextPtr &context);
  virtual ~FuncGraphSpecializer() { specializer_ = nullptr; }
  void Run();
  FuncGraphPtr specialized_func_graph() { return specialized_func_graph_; }

  std::shared_ptr<FuncGraphSpecializer> GetTopSpecializer(const AnfNodePtr &node);

 private:
  ProgramSpecializer *specializer_;
  FuncGraphPtr func_graph_;
  FuncGraphPtr specialized_func_graph_;
  AnalysisContextPtr context_;
  std::shared_ptr<FuncGraphSpecializer> parent_;
  std::shared_ptr<AnalysisEngine> engine_;
  ClonerPtr cloner_;
  std::vector<AnfNodePtr> todo_;
  mindspore::HashSet<AnfNodePtr> marked_;
  mindspore::HashMap<EvaluatorPtr, EvaluatorCacheMgrPtr> eval_cache_;

  void FirstPass();
  void SecondPass();
  void ProcessNode(const AnfNodePtr &node);
  void ProcessCNode(const CNodePtr &node);

  void EliminateUnusedSequenceItem(const CNodePtr &cnode);

  const NodeToNodeMap &cloned_nodes() const { return cloner_->cloned_nodes(); }

  inline AnfNodeConfigPtr MakeConfig(const AnfNodePtr &node) {
    return engine_->MakeConfig(node, context_, func_graph_);  // 'func_graph_' is dummy here.
  }

  inline AnalysisContextPtr MakeContext(const AnalysisEnginePtr &engine, const BaseFuncGraphEvaluatorPtr &evaluator,
                                        const AbstractBasePtrList &args_spec_list) {
    AbstractBasePtrList normalized_args_spec_list = evaluator->NormalizeArgs(args_spec_list);
    FuncGraphPtr fg = evaluator->GetFuncGraph(engine, normalized_args_spec_list);
    MS_EXCEPTION_IF_NULL(evaluator->parent_context());
    AnalysisContextPtr new_context = evaluator->parent_context()->NewContext(fg, normalized_args_spec_list);
    return new_context;
  }

  inline void AddTodoItem(const AnfNodePtr &node) { todo_.push_back(node); }
  inline void AddTodoItem(const std::vector<AnfNodePtr> &nodes) {
    (void)todo_.insert(todo_.end(), nodes.cbegin(), nodes.cend());
  }
  // Get node replicated by Cloner.
  AnfNodePtr GetReplicatedNode(const AnfNodePtr &node);
  // Replicated node which is not used directly by a func graph, so it's not searchable from it's return node
  // (disconnected).
  AnfNodePtr ReplicateDisconnectedNode(const AnfNodePtr &node);

  // Build a value node from parameter if the function graph has special flag to hint it can be done.
  AnfNodePtr BuildSpecializedParameterNode(const CNodePtr &node);
  // Build a value node if ival is a function.
  AnfNodePtr BuildValueNodeForAbstractFunction(const AnfNodePtr &origin_node, const AbstractBasePtr &ival,
                                               const AttrValueMapPtr &attrs, const AnfNodePtr &cnode,
                                               const AbstractFunctionPtr &abs);
  // Build a value node if ival is constant and not any-value
  AnfNodePtr BuildPossibleValueNode(const AnfNodePtr &origin_node, const AbstractBasePtr &ival,
                                    const AttrValueMapPtr &attrs, const AnfNodePtr &cnode = nullptr);
  // Build a replaceable node for iconf->node; it may be a replicated forwarded CNode in static analysis or just a
  // replicated node.
  AnfNodePtr BuildReplacedNode(const AnfNodeConfigPtr &conf);
  // Build a specialized node from given argvals;
  AnfNodePtr BuildSpecializedNode(const CNodePtr &cnode, const AnfNodePtr &node, const AbstractBasePtr &abs,
                                  const AbstractBasePtrList &argvals);
  AnfNodePtr BuildSpecializedNodeInner(const CNodePtr &cnode, const AnfNodePtr &node, const AbstractBasePtr &abs,
                                       const AbstractFunctionPtr &func, const AbstractBasePtrList &args,
                                       SpecializeStatusCode *errcode);

  // Find the unique argument values which can be used to specialize a primitive or graph function.
  SpecializeStatusCode AcquireUniqueEvalVal(const AbstractFunctionPtr &fn, const EvaluatorPtr &eval,
                                            const AbstractBasePtrList &argvals,
                                            std::pair<AbstractBasePtrList, AbstractBasePtr> *result);
  // Get cache, it may be eval's cache or cache built from broaded argument values.
  const EvaluatorCacheMgrPtr GetEvalCache(const EvaluatorPtr &eval);
  // Try to build unique argvals from the broaded arg vals if it is unique.
  std::pair<AbstractBasePtrList, AbstractBasePtr> BuildFromBroadedArgsVal(const EvaluatorPtr &eval);
  void UpdateNewCNodeInputs(const AnfNodePtr &node, const AnfNodePtr &new_node);
};
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_STATIC_ANALYSIS_SPECIALIZE_H_
