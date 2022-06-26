/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PASS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PASS_H_
#include <string>
#include <memory>

#include "utils/hash_map.h"
#include "ir/anf.h"
#include "frontend/optimizer/pattern.h"
#include "include/common/pybind_api/api_register.h"
#include "pybind_api/export_flags.h"

namespace mindspore {
namespace opt {
namespace python_pass {
class PythonPass;
using PythonPassPtr = std::shared_ptr<PythonPass>;
using NodeEquiv = mindspore::HashMap<std::string, AnfNodePtr>;
using NodeEquivPtr = std::shared_ptr<NodeEquiv>;

class PythonPass {
 public:
  explicit PythonPass(const std::string &name, const PatternPtr &src, const PatternPtr &dst, bool run_only_once = false)
      : src_pattern_(src), dst_pattern_(dst), name_(name), run_only_once_(run_only_once) {}
  ~PythonPass() = default;
  bool Run(const FuncGraphPtr &func_graph, const MatchResultPtr &res);
  std::string name() const { return name_; }
  AnfNodePtr Run(const FuncGraphPtr &func_graph, const FuncGraphPtr &top_graph, const AnfNodePtr &node,
                 const MatchResultPtr &res);
  PatternPtr src_pattern() { return src_pattern_; }
  PatternPtr dst_pattern() { return dst_pattern_; }

 private:
  PatternPtr src_pattern_;
  PatternPtr dst_pattern_;
  const std::string name_;
  bool run_only_once_;
};

using PythonPassPtr = std::shared_ptr<PythonPass>;
}  // namespace python_pass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_PASS_H_
