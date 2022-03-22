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

#ifndef HF900DC04_D202_42ED_992A_35DD7C940CE6
#define HF900DC04_D202_42ED_992A_35DD7C940CE6

#include "easy_graph/infra/status.h"
#include "external/graph/gnode.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/vistor/ge_graph_vistor.h"

GE_NS_BEGIN

struct GeSubgraphVisitor : ::EG_NS::GraphVisitor {
  GeSubgraphVisitor(ComputeGraphPtr &, const ::EG_NS::Node &);
  ::EG_NS::Status BuildGraphRelations();

 private:
  ::EG_NS::Status Visit(const ::EG_NS::Graph &) override;
  ::EG_NS::Status Visit(const ::EG_NS::Node &) override;
  ::EG_NS::Status Visit(const ::EG_NS::Edge &) override;

 private:
  ::EG_NS::Status BuildGraphRelations(OpDescPtr &);

 private:
  ComputeGraphPtr &root_graph_;
  const ::EG_NS::Node &node_;
  GeGraphVisitor cur_graph_vistor_;
  std::vector<ComputeGraphPtr> subgraphs_;
};

GE_NS_END

#endif /* HF900DC04_D202_42ED_992A_35DD7C940CE6 */
