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

#ifndef GE_GRAPH_PASSES_WHILE_OP_LABEL_PASS_H_
#define GE_GRAPH_PASSES_WHILE_OP_LABEL_PASS_H_

#include "graph/node.h"
#include "graph/label/label_maker.h"
/***********************************************************************************************************************
                                            +------------+      Step0: DavinciModel::InitNodes
                                            |    Node    |
                                            +------------+          rtLabelCreateExV2
                                            |    Node    |
                                            +------------+
                                            |    Node    |
                                            +------------+
                                            |    While   |
                                            +------------+
            +-----------+                                       Step1: TaskInfo::Init
            |   Node    |                   +------------+
            +-----------+                   | LabelSet(0)|\         LabelSetTaskInfo --> id=0
            |   Node    |                   +------------+ \
            +-----------+                   |StreamActive|  \       If active_stream_list empty, not task.
            |   Node    |                   +------------+   A
            +-----------+                   |     c      |   |
            |   While   |                   +------------+   |
            +-----------+                   |     o      |   |
            |   Node    |                   +------------+   |
            +-----------+                   |     n      |   |
            |   Node    |                   +------------+   |
            +-----------+                   |     d      |   |
            |   Node    |                   +------------+   |
            +-----------+                  /|SwitchByIdx |   |      LabelSwitchByIndexTaskInfo --> rtLabelListCpy({1,2})
                                          / +------------+   |
                                 ====>   /                   |
                                        | \ +------------+   |
                                        |  \| LabelSet(1)|   |      LabelSetTaskInfo --> id=1
                                        |   +------------+   |
                                        |   |StreamActive|   |      If active_stream_list empty, not task.
                                        |   +------------+   |
  +-----------+   +-----------+         |   |     b      |   |
  |     c     |   |     b     |         |   +------------+   |
  +-----------+   +-----------+         |   |     o      |   |
  |     o     |   |     o     |         |   +------------+   |
  +-----------+   +-----------+         |   |     d      |   |
  |     n     |   |     d     |         |   +------------+   |
  +-----------+   +-----------+         |   |     y      |  /
  |     d     |   |     y     |         V   +------------+ /
  +-----------+   +-----------+          \  | LabelGoto  |/         LabelGotoExTaskInfo --> GetLabelGotoAddr(id=0)
                                          \ +------------+
                                           \| LabelSet(2)|          LabelSetTaskInfo --> id=2
                                            +------------+
                                                                Step2: TaskInfo::Distribute
                                            +------------+
                                            |    Node    |          LabelSetTaskInfo           --> rtLabelSet
                                            +------------+          LabelSwitchByIndexTaskInfo --> rtLabelSwitchByIndex
                                            |    Node    |          LabelSetTaskInfo           --> rtLabelSet
                                            +------------+          LabelGotoExTaskInfo        --> rtLabelSwitchByIndex
                                            |    Node    |          LabelSetTaskInfo           --> rtLabelSet
                                            +------------+
***********************************************************************************************************************/
namespace ge {
class WhileOpLabelMaker : public LabelMaker {
 public:
  WhileOpLabelMaker(const ComputeGraphPtr &graph, const NodePtr &owner) : LabelMaker(graph, owner) {}

  ~WhileOpLabelMaker() override {}

  virtual Status Run(uint32_t &label_index);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_WHILE_OP_LABEL_PASS_H_
