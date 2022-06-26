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

#ifndef MINDSPORE_LITE_SRC_PASS_FUSION_CONV_BIASADD_FUSION_H_
#define MINDSPORE_LITE_SRC_PASS_FUSION_CONV_BIASADD_FUSION_H_

#include "backend/common/optimizer/optimizer.h"
#include "tools/converter/converter_context.h"

namespace mindspore {
namespace opt {
class ConvBiasaddFusion : public PatternProcessPass {
 public:
  explicit ConvBiasaddFusion(bool multigraph = true) : PatternProcessPass("ConvBiasaddFusion", multigraph) {}
  ~ConvBiasaddFusion() override = default;

 private:
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  bool CheckCanFusion(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  int DoFuison(const FuncGraphPtr &func_graph, const AnfNodePtr &node) const;
  CNodePtr GetAddCnode(const AnfNodePtr &node) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_PASS_FUSION_CONV_BIASADD_FUSION_H_
