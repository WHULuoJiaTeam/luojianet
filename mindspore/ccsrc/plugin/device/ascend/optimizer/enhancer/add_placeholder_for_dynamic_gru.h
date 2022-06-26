/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_PLACEHOLDER_FOR_DYNAMIC_GRU_H_
#define MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_PLACEHOLDER_FOR_DYNAMIC_GRU_H_

#include <memory>
#include <vector>
#include "backend/common/optimizer/optimizer.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"

namespace mindspore {
namespace opt {
class InsertPlaceholderForDynamicGRUV2 : public PatternProcessPass {
 public:
  explicit InsertPlaceholderForDynamicGRUV2(bool multigraph = true)
      : PatternProcessPass("add_placeholder_for_dynamic_gru", multigraph) {}
  ~InsertPlaceholderForDynamicGRUV2() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PRE_ACTIVATE_ASCEND_ENHANCER_ADD_PLACEHOLDER_FOR_DYNAMIC_GRU_H_
