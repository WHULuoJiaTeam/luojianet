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

#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_graph_dsl/ge.h"

GE_NS_BEGIN

REGISTER_OPTYPE_DEFINE(DATA, "Data");
REGISTER_OPTYPE_DEFINE(HCOMALLGATHER, "HcomAllGather");
REGISTER_OPTYPE_DEFINE(VARIABLE, "Variable");
REGISTER_OPTYPE_DEFINE(CONSTANT, "Const");
REGISTER_OPTYPE_DEFINE(CONSTANTOP, "Constant");
REGISTER_OPTYPE_DEFINE(LESS, "Less");
REGISTER_OPTYPE_DEFINE(MUL, "Mul");
REGISTER_OPTYPE_DEFINE(NETOUTPUT, "NetOutput");
REGISTER_OPTYPE_DEFINE(ADD, "Add");
REGISTER_OPTYPE_DEFINE(WHILE, "While");
REGISTER_OPTYPE_DEFINE(ENTER, "Enter");
REGISTER_OPTYPE_DEFINE(MERGE, "Merge");
REGISTER_OPTYPE_DEFINE(LOOPCOND, "Loopcond");
REGISTER_OPTYPE_DEFINE(SWITCH, "Switch");
REGISTER_OPTYPE_DEFINE(EXIT, "Exit");
REGISTER_OPTYPE_DEFINE(NEXTITERATION, "Nextiteration");

GE_NS_END
