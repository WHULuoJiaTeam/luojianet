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

#ifndef GE_GRAPH_PASSES_CASE_OP_LABEL_PASS_H_
#define GE_GRAPH_PASSES_CASE_OP_LABEL_PASS_H_

#include "graph/node.h"
#include "graph/label/label_maker.h"
/*******************************************************************************
                                                                +------------+
                                                                |    Node    |
                                                                +------------+
                                                                |    Node    |
                                                                +------------+
                                                                |    Case    |
                                                                +------------+
               +-----------+
               |   Node    |                                    +------------+
               +-----------+                                   /|SwitchByIdx |
               |   Node    |                                  A +------------+
               +-----------+                                 / \|LabelSet(0) |
               |   Case    |                                |   +------------+
               +-----------+                                |   |StreamActive|
               |   Node    |                                |   +------------+
               +-----------+                                |   |     c      |
               |   Node    |                                |   +------------+
               +-----------+                                |   |     a      |
               |   Node    |                                |   +------------+
               +-----------+                                |   |     s      |
               |   Node    |                                |   +------------+
               +-----------+                                |   |     e      |
                                                            |   +------------+
                                                   ====>    |   | LabelGoto  |\
                                                            V   +------------+ \
                                                            |\                  \
                                                            | \ +------------+   |
   +-----------+   +-----------+   +-----------+            |  \|LabelSet(1) |   |
   |     c     |   |     c     |   |     c     |            |   +------------+   |
   +-----------+   +-----------+   +-----------+            |   |StreamActive|   |
   |     a     |   |     a     |   |     a     |            |   +------------+   |
   +-----------+   +-----------+   +-----------+            |   |     c      |   |
   |     s     |   |     s     |   |     s     |            |   +------------+   |
   +-----------+   +-----------+   +-----------+            |   |     a      |   |
   |     e     |   |     e     |   |     e     |            |   +------------+   |
   +-----------+   +-----------+   +-----------+            |   |     s      |   |
                                                            |   +------------+   |
                                                            |   |     e      |   |
                                                            |   +------------+   V
                                                            |   | LabelGoto  |\  |
                                                            V   +------------+ \ |
                                                             \                  \|
                                                              \ +------------+   |
                                                               \|LabelSet(2) |   |
                                                                +------------+   |
                                                                |StreamActive|   |
                                                                +------------+   |
                                                                |     c      |   |
                                                                +------------+   |
                                                                |     a      |   |
                                                                +------------+   |
                                                                |     s      |   |
                                                                +------------+   V
                                                                |     e      |  /
                                                                +------------+ /
                                                                | LabelSet   |/
                                                                +------------+

                                                                +------------+
                                                                |    Node    |
                                                                +------------+
                                                                |    Node    |
                                                                +------------+
                                                                |    Node    |
                                                                +------------+
*******************************************************************************/
namespace ge {
class CaseOpLabelMaker : public LabelMaker {
 public:
  CaseOpLabelMaker(const ComputeGraphPtr &graph, const NodePtr &owner) : LabelMaker(graph, owner) {}

  ~CaseOpLabelMaker() override {}

  virtual Status Run(uint32_t &label_index);
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_CASE_OP_LABEL_PASS_H_
