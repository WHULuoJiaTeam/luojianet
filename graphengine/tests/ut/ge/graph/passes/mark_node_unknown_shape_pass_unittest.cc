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

#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <string>

#define private public
#include "graph/passes/mark_node_unknown_shape_pass.h"

#include "common/ge_inner_error_codes.h"
#include "inc/pass_manager.h"
#include "common/local_context.h"
#undef private

namespace ge {
class UtestMarkNodeUnknownShapePass : public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
public:
  NodePtr MakeNode(const ComputeGraphPtr &graph, int in_num, int out_num, string name, string type) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    return graph->AddNode(op_desc);
  }
///    netoutput1
///        |
///       conv1
///     \       /
///        data
  void make_graph(const ComputeGraphPtr &graph) {
    GetLocalOmgContext().fuzz_compile_flag = true;
    auto conv2d_node = MakeNode(graph, 2, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 1, "data", "Data");
      GeTensorDesc tensor_desc(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
    }

    conv2d_node->GetOpDesc()->SetOpKernelLibName("AIcoreEngine");
    AttrUtils::SetBool(conv2d_node->GetOpDesc(), ATTR_NAME_FUZZ_BUILD_RES_ATTRS, true);
    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }
};

TEST_F(UtestMarkNodeUnknownShapePass, test_run_with_GE_kernel) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  op_desc->SetOpKernelLibName("GE");
  graph->AddNode(op_desc);
  PassManager pass;
  pass.AddPass("MarkNodeUnknownShapePass", new (std::nothrow) MarkNodeUnknownShapePass);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
}

TEST_F(UtestMarkNodeUnknownShapePass, test_run_without_fuzz_attrs) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  op_desc->SetOpKernelLibName("AIcoreEngine");
  graph->AddNode(op_desc);
  GetLocalOmgContext().fuzz_compile_flag = true;
  PassManager pass;
  pass.AddPass("MarkNodeUnknownShapePass", new (std::nothrow) MarkNodeUnknownShapePass);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
}

TEST_F(UtestMarkNodeUnknownShapePass, test_run_with_fuzz_attrs) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_graph(graph);
  PassManager pass;
  pass.AddPass("MarkNodeUnknownShapePass", new (std::nothrow) MarkNodeUnknownShapePass);
  EXPECT_EQ(pass.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 3);
  for (const auto &node : graph->GetAllNodes()) {
    if (node->GetName() == "conv1") {
      auto op_desc = node->GetOpDesc();
      EXPECT_NE(op_desc, nullptr);
      for (size_t i = 0; i < op_desc->GetAllInputsSize(); ++i) {
        auto input_desc = op_desc->MutableInputDesc(static_cast<uint32_t>(i));
        EXPECT_TRUE(input_desc->GetShape().GetDim(0) == -2);
      }
      for (auto &output_desc : op_desc->GetAllOutputsDescPtr()) {
        EXPECT_NE(output_desc, nullptr);
        EXPECT_TRUE(output_desc->GetShape().GetDim(0) == -2);
      }
    }
  }
}

}  // namespace ge
