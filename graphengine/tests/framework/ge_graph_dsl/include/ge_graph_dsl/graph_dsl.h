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

#ifndef H7C82E219_BDEF_4480_A2D9_30F0590C8AC5
#define H7C82E219_BDEF_4480_A2D9_30F0590C8AC5

#include "easy_graph/graph/graph.h"
#include "easy_graph/builder/graph_dsl.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/op_desc/op_desc_node_builder.h"
#include "external/graph/graph.h"

GE_NS_BEGIN

Graph ToGeGraph(const ::EG_NS::Graph &graph);
ComputeGraphPtr ToComputeGraph(const ::EG_NS::Graph &graph);

#define DATA_EDGE(...) Data(__VA_ARGS__)
#define CTRL_EDGE(...) Ctrl(__VA_ARGS__)
#define NODE(...) Node(::GE_NS::OpDescNodeBuild(__VA_ARGS__))
#define EDGE(...) DATA_EDGE(__VA_ARGS__)

GE_NS_END

#endif
