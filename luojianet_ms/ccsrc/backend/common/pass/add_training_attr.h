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

#ifndef LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_PASS_ADD_TRAINING_ATTR_H
#define LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_PASS_ADD_TRAINING_ATTR_H
#include <string>

#include "ir/anf.h"
#include "include/common/utils/convert_utils.h"
#include "backend/common/optimizer/optimizer.h"

namespace luojianet_ms {
namespace opt {
class AddTrainingAttr : public PatternProcessPass {
 public:
  explicit AddTrainingAttr(bool multigraph = true) : PatternProcessPass("add_training_attr", multigraph) {}
  ~AddTrainingAttr() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_BACKEND_OPTIMIZER_PASS_ADD_TRAINING_ATTR_H
