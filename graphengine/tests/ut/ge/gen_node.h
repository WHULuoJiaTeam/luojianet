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

#ifndef UT_GE_Gen_Node_H_
#define UT_GE_Gen_Node_H_

#include <gtest/gtest.h>

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/compute_graph.h"
#include "graph/op_desc.h"
#include "graph/optimize/common/params.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "register/op_registry.h"

static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr op_desc);

static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr op_desc) {
  if (!op_desc) {
    return nullptr;
  }
  static auto g = std::make_shared<ge::ComputeGraph>("g");
  return g->AddNode(std::move(op_desc));
}

static void AddInputDesc(ge::OpDescPtr op_desc, int num) {
  for (int i = 0; i < num; ++i) {
    ge::GeTensorDesc tensor;
    tensor.SetFormat(ge::FORMAT_NCHW);
    tensor.SetShape(ge::GeShape({1, 1, 1, 1}));
    tensor.SetDataType(ge::DT_FLOAT);
    ge::TensorUtils::SetRealDimCnt(tensor, 4);
    op_desc->AddInputDesc(tensor);
  }
}

#endif  // UT_GE_Gen_Node_H_
