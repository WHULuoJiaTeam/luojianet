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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUTS_ADJUST_PASS_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUTS_ADJUST_PASS_H_

#include <vector>
#include <string>
#include "tools/optimizer/common/gllo_utils.h"
#include "include/errorcode.h"

using mindspore::lite::STATUS;
namespace mindspore::lite {
class InputAdjust {
 public:
  InputAdjust() {}
  ~InputAdjust() = default;

  STATUS AddAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode, int input_num,
                        const std::string &attr_name, int flag);
  bool Run(const FuncGraphPtr &func_graph);
};
}  // namespace mindspore::lite
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_INPUTS_ADJUST_PASS_H_
