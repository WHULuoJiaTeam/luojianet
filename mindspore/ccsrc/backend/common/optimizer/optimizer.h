/**
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_OPTIMIZER_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_OPTIMIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "backend/common/optimizer/pass_manager.h"
#include "backend/common/optimizer/pattern_engine.h"
#include "ir/graph_utils.h"
#include "utils/ms_utils.h"
#include "backend/common/optimizer/helper.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
using PatternListType = std::initializer_list<BaseRef>;

class BACKEND_EXPORT PatternProcessPass : public NodePass {
 public:
  explicit PatternProcessPass(const std::string &name = "", bool multigraph = true);
  ~PatternProcessPass() override = default;
  virtual const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const = 0;
  virtual const BaseRef DefinePattern() const;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;
  CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs, const FuncGraphPtr &fg) const;
  CNodePtr NewCNode(const CNodePtr &cnode, const KernelGraphPtr &fg) const;

 protected:
  virtual std::vector<AnfNodePtr> GetOrigNodes() const;

 private:
  void Build();

  AnfNodePtr pattern_ = nullptr;
  bool multigraph_ = true;
  PatternEngine pattern_engine_;
  PrimitiveVarMapPtr primitive_vars_;
  EquivPtr equiv_;
};

class MultipleOutputPatternProcessPass : public PatternProcessPass {
 public:
  explicit MultipleOutputPatternProcessPass(const std::string &name = "", bool multigraph = true)
      : PatternProcessPass(name, multigraph),
        child_pattern_engine_(PatternEngine(std::make_shared<Visitor>())),
        child_primitive_vars_(std::make_shared<PrimitiveVarMap>()),
        child_equiv_(std::make_shared<Equiv>()) {}
  ~MultipleOutputPatternProcessPass() override = default;
  virtual BaseRef DefineAnotherPattern() const = 0;
  // check two patterns whether share the same nodes or not
  virtual bool IsShareNodes(const EquivPtr &equiv1, const EquivPtr &equiv2) const = 0;

 protected:
  bool MatchAnotherPattern(const AnfNodePtr &node, const EquivPtr &equiv) const;
  std::vector<AnfNodePtr> GetOrigNodes() const override;
  PatternEngine child_pattern_engine_;
  PrimitiveVarMapPtr child_primitive_vars_;
  EquivPtr child_equiv_;
};

class BACKEND_EXPORT GraphOptimizer {
 public:
  explicit GraphOptimizer(const std::string &name = "graph_optimizer") : name_(name) {}
  virtual ~GraphOptimizer() = default;

  void AddPassManager(const PassManagerPtr &pass_manager);
  FuncGraphPtr Optimize(const FuncGraphPtr &func_graph, bool run_only_once = true);

 private:
  const std::string name_ = "graph_optimizer";
  std::vector<PassManagerPtr> pass_managers_{};
  bool run_only_once_ = true;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_COMMON_OPTIMIZER_H_
