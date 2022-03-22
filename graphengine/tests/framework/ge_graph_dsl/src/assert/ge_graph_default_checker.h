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
#ifndef BCF4D96BE9FC48938DE7B7E93B551C54
#define BCF4D96BE9FC48938DE7B7E93B551C54

#include "ge_graph_dsl/ge.h"
#include "ge_graph_checker.h"
#include "graph/compute_graph.h"

GE_NS_BEGIN

using GraphCheckFun = std::function<void(const ::GE_NS::ComputeGraphPtr &)>;

struct GeGraphDefaultChecker : GeGraphChecker {
  GeGraphDefaultChecker(const std::string &, const GraphCheckFun &);

 private:
  const std::string &PhaseId() const override;
  void Check(const ge::ComputeGraphPtr &graph) const override;

 private:
  const std::string phase_id_;
  const GraphCheckFun check_fun_;
};

GE_NS_END

#endif