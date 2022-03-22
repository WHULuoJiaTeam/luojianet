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

#include "graph/passes/dimension_compute_pass.h"

#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "common/types.h"
#include "graph/passes/base_pass.h"
#include "graph_builder_utils.h"
#include "inc/kernel.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const char *AddNYes = "AddNYes";
const char *AddNNo = "AddNNo";
const char *HuberLossYes = "HuberLossYes";
const char *ShapeNo = "ShapeNo";
const char *ShapeYes = "ShapeYes";
const char *DataNo = "dataNo";
}  // namespace

class UtestShapeYesKernel : public Kernel {
 public:
  Status Compute(const NodePtr &node, std::vector<GeTensorPtr> &v_output) override {
    auto output = std::make_shared<GeTensor>();
    std::vector<uint8_t> data{1, 2, 3};
    std::vector<int64_t> shape{3};
    output->MutableTensorDesc().SetShape(GeShape(shape));
    output->SetData(data);
    output->MutableTensorDesc().SetDataType(DT_UINT8);
    v_output.push_back(output);
    return SUCCESS;
  }
};
REGISTER_KERNEL(ShapeYes, UtestShapeYesKernel);

class UtestGraphPassesDimensionAdjustPass : public testing::Test {
 protected:
  UtestGraphPassesDimensionAdjustPass() = default;
};

namespace {

///     netoutput1
///        |
///      shapeNo1
///       |
///     addnNo1
///    /    \.
///  /       \.
/// const1   const2
ComputeGraphPtr BuildGraph8() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto const2 = builder.AddNode("const2", CONSTANT, 0, 1);
  auto addn1 = builder.AddNode("addn1", AddNNo, 2, 1);
  auto shape1 = builder.AddNode("shape1", ShapeNo, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(const1, 0, addn1, 0);
  builder.AddDataEdge(const2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0, shape1, 0);
  builder.AddDataEdge(shape1, 0, netoutput1, 0);

  return builder.GetGraph();
}

///     netoutput1
///        |
///      shapeNo1
///       |
///     addnYes1
///      /    \.
///     /      \.
///const1   data1
ComputeGraphPtr BuildGraph9() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto data1 = builder.AddNode("data1", DataNo, 0, 1);
  auto addn1 = builder.AddNode("addn1", AddNYes, 2, 1);
  auto shape1 = builder.AddNode("shape1", ShapeNo, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(const1, 0, addn1, 0);
  builder.AddDataEdge(data1, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0, shape1, 0);
  builder.AddDataEdge(shape1, 0, netoutput1, 0);

  return builder.GetGraph();
}

///     netoutput1
///        |
///      shapeYes1
///       |
///     addnNo1
ComputeGraphPtr BuildGraph1() {
  auto builder = ut::GraphBuilder("test");
  auto addnNo1 = builder.AddNode("addnNo1", AddNNo, 2, 1);
  auto shapeYes1 = builder.AddNode("shapeYes1", ShapeYes, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

  builder.AddDataEdge(addnNo1, 0, shapeYes1, 0);
  builder.AddDataEdge(shapeYes1, 0, netoutput1, 0);

  return builder.GetGraph();
}
}  // namespace

TEST_F(UtestGraphPassesDimensionAdjustPass, not_changed_no_kernel) {
  auto graph = BuildGraph8();
  NamesToPass names_to_pass;
  names_to_pass.push_back({"Test", new DimensionComputePass});

  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(graph->GetAllNodes().size(), 5);

  for (auto &name_to_pass : names_to_pass) {
    delete name_to_pass.second;
  }
}

TEST_F(UtestGraphPassesDimensionAdjustPass, not_changed_no_compute_kernel) {
  auto graph = BuildGraph9();
  NamesToPass names_to_pass;
  names_to_pass.push_back({"Test", new DimensionComputePass});

  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(graph->GetAllNodes().size(), 5);

  for (auto &name_to_pass : names_to_pass) {
    delete name_to_pass.second;
  }
}

TEST_F(UtestGraphPassesDimensionAdjustPass, success) {
  auto graph = BuildGraph1();
  NamesToPass names_to_pass;
  names_to_pass.push_back({"Test", new DimensionComputePass});

  GEPass pass(graph);
  EXPECT_EQ(pass.Run(names_to_pass), SUCCESS);

  EXPECT_EQ(graph->GetAllNodes().size(), 2);

  for (auto &name_to_pass : names_to_pass) {
    delete name_to_pass.second;
  }
}
}  // namespace ge
