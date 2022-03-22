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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PASS_MANAGER_H_
#define LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PASS_MANAGER_H_

#include <utility>
#include <vector>
#include <string>
#include <memory>

#include "utils/context/graph_kernel_flags.h"
#include "backend/optimizer/common/pass_manager.h"

namespace luojianet_ms::graphkernel {
using opt::PassManager;
class GraphKernelPassManager : public PassManager {
 public:
  GraphKernelPassManager(size_t stage, const std::string &name)
      : PassManager(name, true), stage_(stage), flags_(GraphKernelFlags::GetInstance()) {}
  ~GraphKernelPassManager() = default;

  // Add graph pass, the pass object will be freed when pass manager freed.
  virtual void AddPass(const opt::PassPtr &pass, unsigned int pass_level, bool default_enable = true);

  // Run passes on the func_graph
  bool Run(const FuncGraphPtr &func_graph) const override;

 protected:
  std::string GetPassFullname(size_t pass_id, const opt::PassPtr &pass) const override;

  size_t stage_;
  std::vector<bool> enabled_;
  const GraphKernelFlags &flags_;
};
}  // namespace luojianet_ms::graphkernel
#endif  // LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_PASS_MANAGER_H_
