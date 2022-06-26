/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_MULTIPLE_PATTERN_PROCESS_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_MULTIPLE_PATTERN_PROCESS_PASS_H_

#include <string>
#include <utility>
#include <memory>
#include <unordered_map>
#include "backend/common/optimizer/node_pass.h"
#include "backend/common/optimizer/pattern_engine.h"
#include "backend/common/optimizer/helper.h"

namespace mindspore {
namespace opt {
class MultiplePatternProcessPass : public NodePass {
 public:
  explicit MultiplePatternProcessPass(const std::string &name = "", bool multigraph = true);
  ~MultiplePatternProcessPass() override = default;
  virtual AnfNodePtr Process(const std::string &, const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const = 0;
  virtual std::unordered_map<std::string, VectorRef> DefinePatterns() const = 0;
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const AnfNodePtr &node) override;

 private:
  std::unordered_map<std::string, BaseRef> patterns_;
  std::unordered_map<std::string, PrimitiveVarMapPtr> primitive_var_maps_;
  bool multigraph_ = true;
  PatternEngine pattern_engine_;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_COMMON_MULTIPLE_PATTERN_PROCESS_PASS_H_
