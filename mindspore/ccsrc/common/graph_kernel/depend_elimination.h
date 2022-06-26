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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_ELIMINATION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_ELIMINATION_H_

#include "backend/common/optimizer/pass.h"

namespace mindspore::graphkernel {
// Optimize cases like %1 = Depend(%0, %0). This depend statement is necessary in frontend, but not here.
class DependElimination : public opt::Pass {
 public:
  DependElimination() : Pass("depend_elimination") {}
  ~DependElimination() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};
}  // namespace mindspore::graphkernel
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_DEPEND_ELIMINATION_H_
