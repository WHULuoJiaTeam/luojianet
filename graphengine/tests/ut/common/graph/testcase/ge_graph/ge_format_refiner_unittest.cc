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
#include <stdlib.h>
#include <iostream>
#include <unordered_map>

#include "graph_builder_utils.h"

#define private public
#define protected public
#include "format_refiner.h"
#undef private
#undef protected

namespace ge {
class UtestFormatRefiner : public testing::Test {
 protected:
  void SetUp() {
    char *which_op = getenv("WHICH_OP");
    if (which_op != nullptr) {
      is_set_env = true;
      return;
      ;
    }
    int ret = setenv("WHICH_OP", "GEOP", 0);
  }

  void TearDown() {
    if (!is_set_env) {
      unsetenv("WHICH_OP");
    }
  }

 private:
  bool is_set_env{false};
};

namespace {

///
///            netoutput1
///               |
///             relu1
///              |
///            conv1
///            /   \.
///        var1  var2
///
ut::GraphBuilder BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNDNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNDNode("var2", "Variable", 0, 1);
  auto conv1 = builder.AddNDNode("conv1", "Conv2D", 2, 1);
  auto conv_data = conv1->GetOpDesc()->GetInputDesc(0);
  conv_data.SetFormat(FORMAT_NCHW);
  conv_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  conv1->GetOpDesc()->UpdateInputDesc(0, conv_data);
  auto weight = conv1->GetOpDesc()->GetInputDesc(1);
  weight.SetFormat(FORMAT_HWCN);
  weight.SetShape(GeShape(std::vector<int64_t>({1, 1, 3, 256})));
  conv1->GetOpDesc()->UpdateInputDesc(1, weight);
  auto conv_out = conv1->GetOpDesc()->GetOutputDesc(0);
  conv_out.SetFormat(FORMAT_NCHW);
  conv_out.SetShape(GeShape(std::vector<int64_t>({1, 256, 224, 224})));
  conv1->GetOpDesc()->UpdateOutputDesc(0, conv_out);
  auto relu1 = builder.AddNDNode("relu1", "Relu", 1, 1);
  auto netoutput1 = builder.AddNDNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, conv1, 0);
  builder.AddDataEdge(var2, 0, conv1, 1);
  builder.AddDataEdge(conv1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput1, 0);
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  return builder;
}

///
///             netoutput1
///               |
///             relu1
///              |
///             bn1 -----------------
///             |   \    \     \     \.
///           conv1 var3 var4  var5  var6
///           |   \.
///         var1  var2
///
ut::GraphBuilder BuildGraph2() {
  auto builder = ut::GraphBuilder("g2");
  auto var1 = builder.AddNDNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNDNode("var2", "Variable", 0, 1);
  auto var3 = builder.AddNDNode("var3", "Variable", 0, 1);
  auto var4 = builder.AddNDNode("var4", "Variable", 0, 1);
  auto var5 = builder.AddNDNode("var5", "Variable", 0, 1);
  auto var6 = builder.AddNDNode("var6", "Variable", 0, 1);
  auto conv1 = builder.AddNDNode("conv1", "Conv2D", 2, 1);
  auto conv_data = conv1->GetOpDesc()->GetInputDesc(0);
  conv_data.SetFormat(FORMAT_NHWC);
  conv_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  conv1->GetOpDesc()->UpdateInputDesc(0, conv_data);
  auto weight = conv1->GetOpDesc()->GetInputDesc(1);
  weight.SetFormat(FORMAT_HWCN);
  weight.SetShape(GeShape(std::vector<int64_t>({1, 1, 3, 256})));
  conv1->GetOpDesc()->UpdateInputDesc(1, weight);
  auto conv_out = conv1->GetOpDesc()->GetOutputDesc(0);
  conv_out.SetFormat(FORMAT_NHWC);
  conv_out.SetShape(GeShape(std::vector<int64_t>({1, 256, 224, 224})));
  conv1->GetOpDesc()->UpdateOutputDesc(0, conv_out);
  auto bn1 = builder.AddNDNode("bn1", "BatchNorm", 5, 1);
  auto relu1 = builder.AddNDNode("relu1", "Relu", 1, 1);
  auto netoutput1 = builder.AddNDNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, conv1, 0);
  builder.AddDataEdge(var2, 0, conv1, 1);
  builder.AddDataEdge(conv1, 0, bn1, 0);
  builder.AddDataEdge(var3, 0, bn1, 1);
  builder.AddDataEdge(var4, 0, bn1, 2);
  builder.AddDataEdge(var5, 0, bn1, 3);
  builder.AddDataEdge(var6, 0, bn1, 4);
  builder.AddDataEdge(bn1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput1, 0);
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  return builder;
}

///
///             netoutput1
///               |
///              conv2
///              |   \.
///             relu1   var3
///              |
///            conv1
///           /   \.
///        var1  var2
///
ut::GraphBuilder BuildGraph3() {
  auto builder = ut::GraphBuilder("g3");
  auto var1 = builder.AddNDNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNDNode("var2", "Variable", 0, 1);
  auto var3 = builder.AddNDNode("var3", "Variable", 0, 1);
  auto conv1 = builder.AddNDNode("conv1", "Conv2D", 2, 1);
  auto conv_data = conv1->GetOpDesc()->GetInputDesc(0);
  conv_data.SetFormat(FORMAT_NCHW);
  conv_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  conv1->GetOpDesc()->UpdateInputDesc(0, conv_data);
  auto weight = conv1->GetOpDesc()->GetInputDesc(1);
  weight.SetFormat(FORMAT_HWCN);
  weight.SetShape(GeShape(std::vector<int64_t>({1, 1, 3, 256})));
  conv1->GetOpDesc()->UpdateInputDesc(1, weight);
  auto conv_out = conv1->GetOpDesc()->GetOutputDesc(0);
  conv_out.SetFormat(FORMAT_NCHW);
  conv_out.SetShape(GeShape(std::vector<int64_t>({1, 256, 224, 224})));
  conv1->GetOpDesc()->UpdateOutputDesc(0, conv_out);
  auto relu1 = builder.AddNDNode("relu1", "Relu", 1, 1);
  auto conv2 = builder.AddNDNode("conv2", "Conv2D", 2, 1);
  conv_data = conv2->GetOpDesc()->GetInputDesc(0);
  conv_data.SetFormat(FORMAT_NHWC);
  conv_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  conv2->GetOpDesc()->UpdateInputDesc(0, conv_data);
  weight = conv2->GetOpDesc()->GetInputDesc(1);
  weight.SetFormat(FORMAT_HWCN);
  weight.SetShape(GeShape(std::vector<int64_t>({1, 1, 3, 256})));
  conv2->GetOpDesc()->UpdateInputDesc(1, weight);
  conv_out = conv2->GetOpDesc()->GetOutputDesc(0);
  conv_out.SetFormat(FORMAT_NHWC);
  conv_out.SetShape(GeShape(std::vector<int64_t>({1, 256, 224, 224})));
  conv2->GetOpDesc()->UpdateOutputDesc(0, conv_out);
  auto netoutput1 = builder.AddNDNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, conv1, 0);
  builder.AddDataEdge(var2, 0, conv1, 1);
  builder.AddDataEdge(conv1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, conv2, 0);
  builder.AddDataEdge(var3, 0, conv2, 1);
  builder.AddDataEdge(conv2, 0, netoutput1, 0);
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  return builder;
}

///
///             netoutput1
///               |
///              conv2
///              |   \.
///             relu1   var3
///              |
///              bn1
///              |
///            conv1
///           /   \.
///        var1  var2
///
ut::GraphBuilder BuildGraph4() {
  auto builder = ut::GraphBuilder("g4");
  auto var1 = builder.AddNDNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNDNode("var2", "Variable", 0, 1);
  auto var3 = builder.AddNDNode("var3", "Variable", 0, 1);
  auto conv1 = builder.AddNDNode("conv1", "Conv2D", 2, 1);
  auto conv_data = conv1->GetOpDesc()->GetInputDesc(0);
  conv_data.SetFormat(FORMAT_NHWC);
  conv_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  conv1->GetOpDesc()->UpdateInputDesc(0, conv_data);
  auto weight = conv1->GetOpDesc()->GetInputDesc(1);
  weight.SetFormat(FORMAT_HWCN);
  weight.SetShape(GeShape(std::vector<int64_t>({1, 1, 3, 256})));
  conv1->GetOpDesc()->UpdateInputDesc(1, weight);
  auto conv_out = conv1->GetOpDesc()->GetOutputDesc(0);
  conv_out.SetFormat(FORMAT_NHWC);
  conv_out.SetShape(GeShape(std::vector<int64_t>({1, 256, 224, 224})));
  conv1->GetOpDesc()->UpdateOutputDesc(0, conv_out);
  auto bn1 = builder.AddNDNode("bn1", "BatchNorm", 1, 1);
  auto relu1 = builder.AddNDNode("relu1", "Relu", 1, 1);
  auto conv2 = builder.AddNDNode("conv2", "Conv2D", 2, 1);
  conv_data = conv2->GetOpDesc()->GetInputDesc(0);
  conv_data.SetFormat(FORMAT_NHWC);
  conv_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  conv2->GetOpDesc()->UpdateInputDesc(0, conv_data);
  weight = conv2->GetOpDesc()->GetInputDesc(1);
  weight.SetFormat(FORMAT_HWCN);
  weight.SetShape(GeShape(std::vector<int64_t>({1, 1, 3, 256})));
  conv2->GetOpDesc()->UpdateInputDesc(1, weight);
  conv_out = conv2->GetOpDesc()->GetOutputDesc(0);
  conv_out.SetFormat(FORMAT_NHWC);
  conv_out.SetShape(GeShape(std::vector<int64_t>({1, 256, 224, 224})));
  conv2->GetOpDesc()->UpdateOutputDesc(0, conv_out);
  auto netoutput1 = builder.AddNDNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, conv1, 0);
  builder.AddDataEdge(var2, 0, conv1, 1);
  builder.AddDataEdge(conv1, 0, bn1, 0);
  builder.AddDataEdge(bn1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, conv2, 0);
  builder.AddDataEdge(var3, 0, conv2, 1);
  builder.AddDataEdge(conv2, 0, netoutput1, 0);
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  return builder;
}

///
///              netoutput1
///                 |
///                apply1
///               /  \.
///  relug1 --> bng1  \.
///     \    /  | \    \.
///      relu1  | |     \.
///           \|  |     |
///           |   |     |
///           bn1 |     |
///             \ |     |
///              conv1  |
///              /    \|
///             /     |
///           data1  var1
///
ut::GraphBuilder BuilderGraph5() {
  auto builder = ut::GraphBuilder("g5");
  auto data1 = builder.AddNDNode("data1", "Data", 0, 1);
  auto var1 = builder.AddNDNode("var1", "Variable", 0, 1);
  auto conv1 = builder.AddNDNode("conv1", "Conv2D", 2, 1);
  auto conv_data = conv1->GetOpDesc()->GetInputDesc(0);
  conv_data.SetFormat(FORMAT_NHWC);
  conv_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  conv1->GetOpDesc()->UpdateInputDesc(0, conv_data);
  auto weight = conv1->GetOpDesc()->GetInputDesc(1);
  weight.SetFormat(FORMAT_HWCN);
  weight.SetShape(GeShape(std::vector<int64_t>({1, 1, 3, 256})));
  conv1->GetOpDesc()->UpdateInputDesc(1, weight);
  auto conv_out = conv1->GetOpDesc()->GetOutputDesc(0);
  conv_out.SetFormat(FORMAT_NHWC);
  conv_out.SetShape(GeShape(std::vector<int64_t>({1, 256, 224, 224})));
  conv1->GetOpDesc()->UpdateOutputDesc(0, conv_out);
  auto bn1 = builder.AddNDNode("bn1", "BatchNorm", 1, 1);
  auto relu1 = builder.AddNDNode("relu1", "Relu", 1, 1);
  auto relug1 = builder.AddNDNode("relug1", "ReluGrad", 1, 1);
  auto bng1 = builder.AddNDNode("bng1", "BatchNormGrad", 4, 1);
  auto apply1 = builder.AddNDNode("apply1", "ApplyMomentum", 2, 1);
  auto netoutput1 = builder.AddNDNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(data1, 0, conv1, 0);
  builder.AddDataEdge(var1, 0, conv1, 1);
  builder.AddDataEdge(var1, 0, apply1, 1);
  builder.AddDataEdge(conv1, 0, bn1, 0);
  builder.AddDataEdge(conv1, 0, bng1, 3);
  builder.AddDataEdge(bn1, 0, relu1, 0);
  builder.AddDataEdge(bn1, 0, bng1, 2);
  builder.AddDataEdge(relu1, 0, relug1, 0);
  builder.AddDataEdge(relu1, 0, bng1, 1);
  builder.AddDataEdge(relug1, 0, bng1, 0);
  builder.AddDataEdge(bng1, 0, apply1, 0);
  builder.AddDataEdge(apply1, 0, netoutput1, 0);
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  return builder;
}

///
///             netoutput1
///              |
///            AddN
///           /   \          \.
///        L2Loss  GatherV2   Constant
///          /      \.
///        Data1   Data2
///
ut::GraphBuilder BuildGraph6() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNDNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNDNode("data2", "Data", 1, 1);
  auto loss = builder.AddNDNode("loss", "L2Loss", 1, 1);
  auto gather = builder.AddNDNode("gather", "GatherV2", 1, 1);
  auto addn = builder.AddNDNode("addN", "AddN", 3, 1);
  auto netoutput = builder.AddNDNode("netoutput", "NetOutput", 1, 0);
  auto constant = builder.AddNDNode("constant", "Constant", 0, 1);

  auto data1_input = data1->GetOpDesc()->GetInputDesc(0);
  data1_input.SetFormat(FORMAT_HWCN);
  data1->GetOpDesc()->UpdateInputDesc(0, data1_input);
  auto data1_output = data1->GetOpDesc()->GetOutputDesc(0);
  data1_output.SetFormat(FORMAT_HWCN);
  data1->GetOpDesc()->UpdateOutputDesc(0, data1_output);

  auto net_input = netoutput->GetOpDesc()->GetInputDesc(0);
  net_input.SetFormat(FORMAT_NCHW);
  netoutput->GetOpDesc()->UpdateInputDesc(0, net_input);

  auto data2_input = data2->GetOpDesc()->GetInputDesc(0);
  data2_input.SetFormat(FORMAT_HWCN);
  data2->GetOpDesc()->UpdateInputDesc(0, data2_input);
  auto data2_output = data2->GetOpDesc()->GetOutputDesc(0);
  data2_output.SetFormat(FORMAT_HWCN);
  data2->GetOpDesc()->UpdateOutputDesc(0, data2_output);

  builder.AddDataEdge(data1, 0, loss, 0);
  builder.AddDataEdge(data2, 0, gather, 0);
  builder.AddDataEdge(loss, 0, addn, 0);
  builder.AddDataEdge(gather, 0, addn, 1);
  builder.AddDataEdge(constant, 0, addn, 2);
  builder.AddDataEdge(addn, 0, netoutput, 0);

  //FormatRefiner::SetInferOrigineFormatFlag(true);

  return builder;
}

///
///             netoutput1
///              |
///            AddN
///           /   \          \.
///        L2Loss  GatherV2   Constant
///          /      \.
///        Data1   Data2
///
ut::GraphBuilder BuildGraph7() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNDNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNDNode("data2", "Data", 1, 1);
  auto loss = builder.AddNDNode("loss", "L2Loss", 1, 1);
  auto gather = builder.AddNDNode("gather", "GatherV2", 1, 1);
  auto addn = builder.AddNDNode("addN", "AddN", 3, 1);
  auto netoutput = builder.AddNDNode("netoutput", "NetOutput", 1, 0);
  auto constant = builder.AddNDNode("constant", "Constant", 0, 1);

  auto data1_input = data1->GetOpDesc()->GetInputDesc(0);
  data1->GetOpDesc()->UpdateInputDesc(0, data1_input);
  auto data1_output = data1->GetOpDesc()->GetOutputDesc(0);
  data1->GetOpDesc()->UpdateOutputDesc(0, data1_output);

  auto net_input = netoutput->GetOpDesc()->GetInputDesc(0);
  netoutput->GetOpDesc()->UpdateInputDesc(0, net_input);

  auto data2_input = data2->GetOpDesc()->GetInputDesc(0);
  data2->GetOpDesc()->UpdateInputDesc(0, data2_input);
  auto data2_output = data2->GetOpDesc()->GetOutputDesc(0);
  data2->GetOpDesc()->UpdateOutputDesc(0, data2_output);

  builder.AddDataEdge(data1, 0, loss, 0);
  builder.AddDataEdge(data2, 0, gather, 0);
  builder.AddDataEdge(loss, 0, addn, 0);
  builder.AddDataEdge(gather, 0, addn, 1);
  builder.AddDataEdge(constant, 0, addn, 2);
  builder.AddDataEdge(addn, 0, netoutput, 0);

  //FormatRefiner::SetInferOrigineFormatFlag(true);

  return builder;
}

///
///                  data2
///                   |
///          data1   relu
///                   |
///                  reshape
///             \   /
///             conv
///              |
///             netoutput
///
ut::GraphBuilder BuildGraph8() {
  auto builder = ut::GraphBuilder("g8");

  auto data1 = builder.AddNDNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNDNode("data2", "Data", 1, 1);
  auto relu = builder.AddNDNode("relu", "Relu", 1, 1);
  auto reshape = builder.AddNDNode("reshape", "Reshape", 1, 1);
  auto conv = builder.AddNDNode("conv", "Conv2D", 2, 1);
  auto netoutput = builder.AddNDNode("netoutput", "NetOutput", 1, 0);

  auto reshape_data = reshape->GetOpDesc()->GetInputDesc(0);
  reshape_data.SetFormat(FORMAT_ND);
  reshape_data.SetOriginFormat(FORMAT_ND);
  reshape_data.SetShape(GeShape(std::vector<int64_t>({224, 224})));
  reshape_data.SetShape(GeShape(std::vector<int64_t>({224, 224})));
  reshape->GetOpDesc()->UpdateInputDesc(0, reshape_data);
  reshape->GetOpDesc()->UpdateOutputDesc(0, reshape_data);

  auto conv_data = conv->GetOpDesc()->GetInputDesc(0);
  conv_data.SetFormat(FORMAT_NHWC);
  conv_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 224, 224})));
  conv->GetOpDesc()->UpdateInputDesc(0, conv_data);
  auto weight = conv->GetOpDesc()->GetInputDesc(1);
  weight.SetFormat(FORMAT_HWCN);
  weight.SetShape(GeShape(std::vector<int64_t>({1, 1, 3, 256})));
  conv->GetOpDesc()->UpdateInputDesc(1, weight);
  auto conv_out = conv->GetOpDesc()->GetOutputDesc(0);
  conv_out.SetFormat(FORMAT_NHWC);
  conv_out.SetShape(GeShape(std::vector<int64_t>({1, 256, 224, 224})));
  conv->GetOpDesc()->UpdateOutputDesc(0, conv_out);

  builder.AddDataEdge(data1, 0, conv, 0);
  builder.AddDataEdge(data2, 0, relu, 0);
  builder.AddDataEdge(relu, 0, reshape, 0);
  builder.AddDataEdge(reshape, 0, conv, 1);
  builder.AddDataEdge(conv, 0, netoutput, 0);
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  return builder;
}
}  // namespace
/*
TEST_F(UtestFormatRefiner, data_format) {
  auto builder = BuildGraph8();
  auto graph = builder.GetGraph();
  //FormatRefiner::SetInferOrigineFormatFlag(false);
  graph->SaveDataFormat(FORMAT_NCHW);
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto data2 = graph->FindNode("data2");
  auto relu = graph->FindNode("relu");
  EXPECT_EQ(data2->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(data2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  //FormatRefiner::SetInferOrigineFormatFlag(true);
}
*/
TEST_F(UtestFormatRefiner, constant_fail) {
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  auto builder = BuildGraph6();
  auto graph = builder.GetGraph();
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_FAILED);
}

TEST_F(UtestFormatRefiner, scalar_nodes_infer) {
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  auto builder = BuildGraph6();
  auto graph = builder.GetGraph();
  auto constant = graph->FindNode("constant");
  ge::GeTensorPtr value = std::make_shared<GeTensor>();
  AttrUtils::SetTensor(constant->GetOpDesc(), "value", value);
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
}

TEST_F(UtestFormatRefiner, forward_and_default_infer_func) {
  auto builder = BuildGraph1();
  auto graph = builder.GetGraph();
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto var2 = graph->FindNode("var2");
  EXPECT_EQ(var2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_HWCN);
  auto relu1 = graph->FindNode("relu1");
  EXPECT_EQ(relu1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto netoutput1 = graph->FindNode("netoutput1");
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto conv1 = graph->FindNode("conv1");
  EXPECT_EQ(conv1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(conv1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_HWCN);
  EXPECT_EQ(conv1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
}

TEST_F(UtestFormatRefiner, forward_and_specifed_infer_func) {
  auto builder = BuildGraph1();
  auto graph = builder.GetGraph();
  auto relu1 = graph->FindNode("relu1");
  relu1->GetOpDesc()->AddInferFormatFunc([](Operator &op) {
    auto output1 = op.GetOutputDesc(0);
    output1.SetOriginFormat(FORMAT_NHWC);
    op.UpdateOutputDesc("0", output1);
    return GRAPH_SUCCESS;
  });

  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto var2 = graph->FindNode("var2");
  EXPECT_EQ(var2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_HWCN);
  EXPECT_EQ(relu1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  auto netoutput1 = graph->FindNode("netoutput1");
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
}

TEST_F(UtestFormatRefiner, failed_when_infer) {
  auto builder = BuildGraph1();
  auto graph = builder.GetGraph();
  auto relu1 = graph->FindNode("relu1");
  relu1->GetOpDesc()->AddInferFormatFunc([](Operator &op) { return GRAPH_FAILED; });

  EXPECT_NE(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
}

TEST_F(UtestFormatRefiner, forward_backward) {
  auto builder = BuildGraph2();
  auto graph = builder.GetGraph();

  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto bn1 = graph->FindNode("bn1");
  EXPECT_EQ(bn1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(bn1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  for (auto name : {"var3", "var4", "var5", "var6"}) {
    auto node = graph->FindNode(name);
    EXPECT_EQ(node->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  }
}

TEST_F(UtestFormatRefiner, format_conflict) {
  auto builder = BuildGraph3();
  auto graph = builder.GetGraph();
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
}

TEST_F(UtestFormatRefiner, infer_stop_nd) {
  auto builder = BuildGraph1();
  auto graph = builder.GetGraph();
  auto relu1 = graph->FindNode("relu1");
  relu1->GetOpDesc()->AddInferFormatFunc([](Operator &op) { return GRAPH_SUCCESS; });
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto var2 = graph->FindNode("var2");
  EXPECT_EQ(var2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_HWCN);
  relu1 = graph->FindNode("relu1");
  EXPECT_EQ(relu1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_ND);
  auto netoutput1 = graph->FindNode("netoutput1");
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_ND);
  auto conv1 = graph->FindNode("conv1");
  EXPECT_EQ(conv1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(conv1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_HWCN);
  EXPECT_EQ(conv1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
}

TEST_F(UtestFormatRefiner, infer_stop_same_format) {
  auto builder = BuildGraph4();
  auto graph = builder.GetGraph();
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
}

TEST_F(UtestFormatRefiner, forward_multi_output) {
  auto builder = BuilderGraph5();
  auto graph = builder.GetGraph();
  auto apply1 = graph->FindNode("apply1");
  apply1->GetOpDesc()->AddInferFormatFunc([](Operator &op) {
    auto out = op.GetOutputDesc(0);
    out.SetOriginFormat(FORMAT_NHWC);
    op.UpdateOutputDesc("0", out);
    auto in0 = op.GetInputDesc(0);
    in0.SetOriginFormat(FORMAT_NHWC);
    op.UpdateInputDesc("0", in0);
    auto in1 = op.GetInputDesc(1);
    in1.SetOriginFormat(FORMAT_HWCN);
    op.UpdateInputDesc("1", in1);
    return GRAPH_SUCCESS;
  });

  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);

  auto data1 = graph->FindNode("data1");
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_HWCN);
  auto bn1 = graph->FindNode("bn1");
  EXPECT_EQ(bn1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(bn1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  auto relu1 = graph->FindNode("relu1");
  EXPECT_EQ(relu1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(relu1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  auto relug1 = graph->FindNode("relug1");
  EXPECT_EQ(relug1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(relug1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  auto bng1 = graph->FindNode("bng1");
  EXPECT_EQ(bng1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(bng1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(bng1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(bng1->GetOpDesc()->GetInputDesc(2).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(bng1->GetOpDesc()->GetInputDesc(3).GetOriginFormat(), FORMAT_NHWC);

  EXPECT_EQ(apply1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(apply1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(apply1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_HWCN);
}

TEST_F(UtestFormatRefiner, get_anchor_points_failed) {
  ge::ComputeGraphPtr graph = nullptr;
  std::vector<ge::NodePtr> anchor_points;
  std::vector<ge::NodePtr> data_nodes;
  std::unordered_map<ge::NodePtr, bool> node_status;
  auto status = FormatRefiner::GetAnchorPoints(graph, anchor_points, data_nodes, node_status);
  EXPECT_EQ(status, GRAPH_FAILED);
}

TEST_F(UtestFormatRefiner, anchor_process_failed) {
  ge::NodePtr anchor_node;
  std::unordered_map<ge::NodePtr, bool> node_status;
  auto status = FormatRefiner::AnchorProcess(anchor_node, node_status);
  EXPECT_EQ(status, GRAPH_FAILED);
}

TEST_F(UtestFormatRefiner, infer_origine_format_failed) {
  ge::ComputeGraphPtr graph = nullptr;
  auto status = FormatRefiner::InferOrigineFormat(graph);
  EXPECT_EQ(status, GRAPH_FAILED);
}

TEST_F(UtestFormatRefiner, save_format) {
  //FormatRefiner::SetInferOrigineFormatFlag(true);
  auto builder = BuildGraph6();
  auto graph = builder.GetGraph();
  graph->SaveDataFormat(FORMAT_NHWC);
  auto save_format = graph->GetDataFormat();
  EXPECT_EQ(save_format, FORMAT_NHWC);
  graph->SaveDataFormat(FORMAT_ND);
}
}  // namespace ge
