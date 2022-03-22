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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REWRITE_OUTPUT_SHAPE_H_
#define LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REWRITE_OUTPUT_SHAPE_H_
#include <string>
#include "ir/func_graph.h"
#include "backend/optimizer/common/pass.h"

namespace luojianet_ms::graphkernel {
class SaveOutputShape : public opt::Pass {
 public:
  explicit SaveOutputShape(const std::string &pass_name = "save_output_shape") : Pass(pass_name) {}
  ~SaveOutputShape() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;
};

class RewriteOutputShape : public opt::Pass {
 public:
  explicit RewriteOutputShape(const std::string &pass_name = "rewrite_output_shape") : Pass(pass_name) {}
  ~RewriteOutputShape() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  void Process(const AnfNodePtr &node, size_t index, const AbstractBasePtr &abstract);
};
}  // namespace luojianet_ms::graphkernel
#endif  // LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_GRAPH_KERNEL_REWRITE_OUTPUT_SHAPE_H_
