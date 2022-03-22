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

#ifndef LUOJIANET_MS_LITE_EXAMPLES_CONVERTER_EXTEND_PASS_PASS_REGISTRY_TUTORIAL_H
#define LUOJIANET_MS_LITE_EXAMPLES_CONVERTER_EXTEND_PASS_PASS_REGISTRY_TUTORIAL_H

#include "include/registry/pass_base.h"

namespace luojianet_ms {
namespace opt {
class PassTutorial : public registry::PassBase {
 public:
  PassTutorial() : PassBase("PassTutorial") {}

  ~PassTutorial() = default;

  bool Execute(const api::FuncGraphPtr &func_graph) override;

 private:
  AnfNodePtr CreateCustomOp(const api::FuncGraphPtr func_graph, const CNodePtr &cnode);
};
}  // namespace opt
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_LITE_EXAMPLES_CONVERTER_EXTEND_PASS_PASS_REGISTRY_TUTORIAL_H
