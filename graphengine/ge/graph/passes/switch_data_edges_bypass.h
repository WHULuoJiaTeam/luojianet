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
#ifndef GE_SWITCH_DATA_EDGES_BYPASS_H_
#define GE_SWITCH_DATA_EDGES_BYPASS_H_

#include "inc/graph_pass.h"

namespace ge {
class SwitchDataEdgesBypass : public GraphPass {
 public:
  Status Run(ComputeGraphPtr graph) override;
 private:
  Status BypassSwitch(const NodePtr &node);
};
}  // namespace ge

#endif  //GE_SWITCH_DATA_EDGES_BYPASS_H_