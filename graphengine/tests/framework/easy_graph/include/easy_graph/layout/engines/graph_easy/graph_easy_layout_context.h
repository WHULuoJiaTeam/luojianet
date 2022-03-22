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

#ifndef HB8CC77BE_6A2E_4EB4_BE59_CA85DE56C027
#define HB8CC77BE_6A2E_4EB4_BE59_CA85DE56C027

#include "easy_graph/eg.h"
#include <string>
#include <deque>

EG_NS_BEGIN

struct GraphEasyOption;
struct Graph;

struct GraphEasyLayoutContext {
  GraphEasyLayoutContext(const GraphEasyOption &);

  const Graph *GetCurrentGraph() const;

  void EnterGraph(const Graph &);
  void ExitGraph();

  void LinkBegin();
  void LinkEnd();

  bool InLinking() const;

  std::string GetGroupPath() const;
  const GraphEasyOption &GetOptions() const;

 private:
  std::deque<const Graph *> graphs_;
  const GraphEasyOption &options_;
  bool is_linking_{false};
};

EG_NS_END

#endif
