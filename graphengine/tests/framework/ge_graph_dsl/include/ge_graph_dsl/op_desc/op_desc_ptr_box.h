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

#ifndef HCFDD0816_CC46_4264_9363_9E8C6934F43E
#define HCFDD0816_CC46_4264_9363_9E8C6934F43E

#include "easy_graph/eg.h"
#include "easy_graph/graph/node_id.h"
#include "graph/op_desc.h"
#include "ge_graph_dsl/ge.h"
#include "ge_graph_dsl/op_desc/op_box.h"

GE_NS_BEGIN

struct OpDescPtrBox : OpBox {
  OpDescPtrBox(const OpDescPtr &op) : op_(op) {}

 private:
  OpDescPtr Build(const ::EG_NS::NodeId &id) const override;
  const OpDescPtr op_;
};

GE_NS_END

#endif /* HCFDD0816_CC46_4264_9363_9E8C6934F43E */
