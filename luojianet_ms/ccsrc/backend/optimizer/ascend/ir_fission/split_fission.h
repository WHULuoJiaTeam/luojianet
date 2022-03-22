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
#ifndef LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_SPLIT_FISSION_H_
#define LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_SPLIT_FISSION_H_

#include <string>
#include "backend/optimizer/common/optimizer.h"

namespace luojianet_ms {
namespace opt {
constexpr int64_t kSplitOutputsDivisor = 63;
class SplitFission : public PatternProcessPass {
 public:
  explicit SplitFission(const std::string name = "split_fission", bool multigraph = true,
                        int64_t divisor = kSplitOutputsDivisor)
      : PatternProcessPass(name, multigraph), outputs_divisor_(divisor) {}
  ~SplitFission() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 protected:
  AnfNodePtr DoFission(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int64_t num_split, int64_t divisor,
                       int64_t split_dim) const;
  CNodePtr CreateSplitVNode(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node) const;
  CNodePtr CreateBaseSplitVNode(const FuncGraphPtr &func_graph, const CNodePtr &origin_cnode) const;
  int64_t outputs_divisor_;
};
}  // namespace opt
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FISSION_SPLIT_FISSION_H_
