/**
 * Copyright 2019 Huawei Technologies Co., Ltd

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at

 * http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include <gtest/gtest.h>
#include <iostream>
#include <unordered_map>
#include <stdlib.h>
#include "graph_builder_utils.h"
#define private public
#define protected public
#include "graph/format_refiner.h"
#include "graph/ref_relation.h"
#undef private
#undef protected

namespace ge {
class UTEST_FormatRefiner : public testing::Test {
 protected:
  void SetUp()
  {
    char* which_op = getenv("WHICH_OP");
    if (which_op != nullptr) {
      is_set_env = true;
      return;;
    }
    int ret = setenv("WHICH_OP", "GEOP", 0);
  }

  void TearDown()
  {
    if (!is_set_env) {
      unsetenv("WHICH_OP");
    }
  }

private:
  bool  is_set_env{false};

};

namespace {
const string kIsGraphInferred = "_is_graph_inferred";

void SetFirstInferFlag(ComputeGraphPtr graph, bool is_first) {
  (void)AttrUtils::SetBool(graph, kIsGraphInferred, !is_first);
}

/*
 *              netoutput1
 *                |
 *              relu1
 *               |
 *             conv1
 *            /   \
 *         var1  var2
 */
ut::GraphBuilder BuildGraph1() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto conv1 = builder.AddNode("conv1", "Conv2D", 2, 1);
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
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, conv1, 0);
  builder.AddDataEdge(var2, 0, conv1, 1);
  builder.AddDataEdge(conv1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput1, 0);
  SetFirstInferFlag(builder.GetGraph(), true);
  return builder;
}

/*
 *              netoutput1
 *                |
 *              relu1
 *               |
 *              bn1 -----------------
 *              |   \    \     \     \
 *            conv1 var3 var4  var5  var6
 *            |   \
 *          var1  var2
 */
ut::GraphBuilder BuildGraph2() {
  auto builder = ut::GraphBuilder("g2");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto var3 = builder.AddNode("var3", "Variable", 0, 1);
  auto var4 = builder.AddNode("var4", "Variable", 0, 1);
  auto var5 = builder.AddNode("var5", "Variable", 0, 1);
  auto var6 = builder.AddNode("var6", "Variable", 0, 1);
  auto conv1 = builder.AddNode("conv1", "Conv2D", 2, 1);
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
  auto bn1 = builder.AddNode("bn1", "BatchNorm", 5, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, conv1, 0);
  builder.AddDataEdge(var2, 0, conv1, 1);
  builder.AddDataEdge(conv1, 0, bn1, 0);
  builder.AddDataEdge(var3, 0, bn1, 1);
  builder.AddDataEdge(var4, 0, bn1, 2);
  builder.AddDataEdge(var5, 0, bn1, 3);
  builder.AddDataEdge(var6, 0, bn1, 4);
  builder.AddDataEdge(bn1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, netoutput1, 0);
  SetFirstInferFlag(builder.GetGraph(), true);
  return builder;
}

/*
 *              netoutput1
 *                |
 *               conv2
 *               |   \
 *              relu1   var3
 *               |
 *             conv1
 *            /   \
 *         var1  var2
 */
ut::GraphBuilder BuildGraph3() {
  auto builder = ut::GraphBuilder("g3");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto var3 = builder.AddNode("var3", "Variable", 0, 1);
  auto conv1 = builder.AddNode("conv1", "Conv2D", 2, 1);
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
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto conv2 = builder.AddNode("conv2", "Conv2D", 2, 1);
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
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, conv1, 0);
  builder.AddDataEdge(var2, 0, conv1, 1);
  builder.AddDataEdge(conv1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, conv2, 0);
  builder.AddDataEdge(var3, 0, conv2, 1);
  builder.AddDataEdge(conv2, 0, netoutput1, 0);
  SetFirstInferFlag(builder.GetGraph(), true);
  return builder;
}

/*
 *              netoutput1
 *                |
 *               conv2
 *               |   \
 *              relu1   var3
 *               |
 *               bn1
 *               |
 *             conv1
 *            /   \
 *         var1  var2
 */
ut::GraphBuilder BuildGraph4() {
  auto builder = ut::GraphBuilder("g4");
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "Variable", 0, 1);
  auto var3 = builder.AddNode("var3", "Variable", 0, 1);
  auto conv1 = builder.AddNode("conv1", "Conv2D", 2, 1);
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
  auto bn1 = builder.AddNode("bn1", "BatchNorm", 1, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto conv2 = builder.AddNode("conv2", "Conv2D", 2, 1);
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
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  builder.AddDataEdge(var1, 0, conv1, 0);
  builder.AddDataEdge(var2, 0, conv1, 1);
  builder.AddDataEdge(conv1, 0, bn1, 0);
  builder.AddDataEdge(bn1, 0, relu1, 0);
  builder.AddDataEdge(relu1, 0, conv2, 0);
  builder.AddDataEdge(var3, 0, conv2, 1);
  builder.AddDataEdge(conv2, 0, netoutput1, 0);
  SetFirstInferFlag(builder.GetGraph(), true);
  return builder;
}

/*
 *               netoutput1
 *                  |
 *                 apply1
 *                /  \
 * relug1 --> bng1    \
 *      \    /  | \    \
 *       relu1  | |     \
 *            \|  |     |
 *            |   |     |
 *            bn1 |     |
 *              \ |     |
 *               conv1  |
 *               /    \|
 *              /     |
 *            data1  var1
 */
ut::GraphBuilder BuilderGraph5() {
  auto builder = ut::GraphBuilder("g5");
  auto data1 = builder.AddNode("data1", "Data", 0, 1);
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto conv1 = builder.AddNode("conv1", "Conv2D", 2, 1);
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
  auto bn1 = builder.AddNode("bn1", "BatchNorm", 1, 1);
  auto relu1 = builder.AddNode("relu1", "Relu", 1, 1);
  auto relug1 = builder.AddNode("relug1", "ReluGrad", 1, 1);
  auto bng1 = builder.AddNode("bng1", "BatchNormGrad", 4, 1);
  auto apply1 = builder.AddNode("apply1", "ApplyMomentum", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

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
  SetFirstInferFlag(builder.GetGraph(), true);
  return builder;
}
/*
 *              netoutput1
 *               |
 *             AddN
 *            /   \          \
 *         L2Loss  GatherV2   Constant
 *           /      \
 *         Data1   Data2
 *
 *
 */
ut::GraphBuilder BuildGraph6() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto loss = builder.AddNode("loss", "L2Loss", 1, 1);
  auto gather = builder.AddNode("gather", "GatherV2", 1, 1);
  auto addN = builder.AddNode("addN", "AddN", 3, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);
  auto constant = builder.AddNode("constant", "Constant", 0, 1);

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
  builder.AddDataEdge(loss, 0, addN, 0);
  builder.AddDataEdge(gather, 0, addN, 1);
  builder.AddDataEdge(constant, 0, addN, 2);
  builder.AddDataEdge(addN, 0, netoutput, 0);

  SetFirstInferFlag(builder.GetGraph(), true);

  return builder;
}
/*
 *              netoutput1
 *               |
 *             AddN
 *            /   \          \
 *         L2Loss  GatherV2   Constant
 *           /      \
 *         Data1   Data2
 *
 *
 */
ut::GraphBuilder BuildGraph7() {
  auto builder = ut::GraphBuilder("g1");
  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto loss = builder.AddNode("loss", "L2Loss", 1, 1);
  auto gather = builder.AddNode("gather", "GatherV2", 1, 1);
  auto addN = builder.AddNode("addN", "AddN", 3, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);
  auto constant = builder.AddNode("constant", "Constant", 0, 1);

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
  builder.AddDataEdge(loss, 0, addN, 0);
  builder.AddDataEdge(gather, 0, addN, 1);
  builder.AddDataEdge(constant, 0, addN, 2);
  builder.AddDataEdge(addN, 0, netoutput, 0);

  SetFirstInferFlag(builder.GetGraph(), true);

  return builder;
}
/*
 *                   data2
 *                    |
 *           data1   relu
 *                    |
 *                   reshape
 *              \   /
 *              conv
 *               |
 *              netoutput
 */

ut::GraphBuilder BuildGraph8() {
  auto builder = ut::GraphBuilder("g8");

  auto data1 = builder.AddNode("data1", "Data", 1, 1);
  auto data2 = builder.AddNode("data2", "Data", 1, 1);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);
  auto reshape = builder.AddNode("reshape", "Reshape", 1, 1);
  auto conv = builder.AddNode("conv", "Conv2D", 2, 1);
  auto netoutput = builder.AddNode("netoutput", "NetOutput", 1, 0);

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
  SetFirstInferFlag(builder.GetGraph(), true);
  return builder;
}

/*
 *            netoutput1
 *               |
 *             BiasAdd
 *               |
 *             square
 *               |
 *              var
 */
ut::GraphBuilder BuildGraph9() {
  auto builder = ut::GraphBuilder("g9");
  auto var = builder.AddNode("var", "Variable", 0, 1);
  auto square = builder.AddNode("square", "Square", 1, 1);
  auto biasadd = builder.AddNode("biasadd", "BiasAdd", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "NetOutput", 1, 0);

  auto biasadd_data = biasadd->GetOpDesc()->GetInputDesc(0);
  biasadd_data.SetFormat(FORMAT_NHWC);
  biasadd_data.SetOriginFormat(FORMAT_NHWC);
  biasadd_data.SetShape(GeShape(std::vector<int64_t>({1, 3, 3,224, 224})));
  biasadd->GetOpDesc()->UpdateInputDesc(0, biasadd_data);
  auto biasadd_out = biasadd->GetOpDesc()->GetOutputDesc(0);
  biasadd_out.SetFormat(FORMAT_NHWC);
  biasadd_out.SetOriginFormat(FORMAT_NHWC);
  biasadd_out.SetShape(GeShape(std::vector<int64_t>({1, 3, 256, 224, 224})));
  biasadd->GetOpDesc()->UpdateOutputDesc(0, biasadd_out);


  builder.AddDataEdge(var, 0, square, 0);
  builder.AddDataEdge(square, 0, biasadd, 0);
  builder.AddDataEdge(biasadd, 0, netoutput1, 0);
  SetFirstInferFlag(builder.GetGraph(), true);
  return builder;
}

/*
 *   netoutput1
 *       |      \
 *      sub      relu
 *     /   \     /
 * data1   data2
 */
ComputeGraphPtr BuildSubGraph(const std::string name,ge::Format to_be_set_format = FORMAT_ND) {
  ut::GraphBuilder builder(name);
  auto data1 = builder.AddNode(name + "data1", "Data", 1, 1);
  auto data2 = builder.AddNode(name + "data2", "Data", 1, 1);
  auto sub = builder.AddNode(name + "sub", "Sub", 2, 1, to_be_set_format);
  auto relu = builder.AddNode(name + "relu", "Relu", 1, 1);
  auto netoutput = builder.AddNode(name + "netoutput", "NetOutput", 2, 2);

  AttrUtils::SetInt(data1->GetOpDesc(), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(data2->GetOpDesc(), "_parent_node_index", static_cast<int>(1));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), "_parent_node_index", static_cast<int>(0));
  AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(1), "_parent_node_index", static_cast<int>(1));


  builder.AddDataEdge(data1, 0, sub, 0);
  builder.AddDataEdge(data2, 0, sub, 1);
  builder.AddDataEdge(sub, 0, netoutput, 0);
  builder.AddDataEdge(data2, 0, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput, 1);


  return builder.GetGraph();
}
/*
 *   netoutput relu
 *       |    /
 *      if
 *     /   \
 * data1   data2
 */
ComputeGraphPtr BuildMainGraphWithIf(string anchor_graph) {
  ut::GraphBuilder builder("main_graph");
  auto to_be_set_format = FORMAT_ND;
  auto to_be_set_format_of_sub = FORMAT_ND;
  if (anchor_graph == "main") {
    to_be_set_format = FORMAT_NHWC;
    to_be_set_format_of_sub = FORMAT_ND;
  } else {
    to_be_set_format = FORMAT_ND;
    to_be_set_format_of_sub = FORMAT_NHWC;
  }
  auto data1 = builder.AddNode("data1", "Data", 1, 1, to_be_set_format);
  auto data2 = builder.AddNode("data2", "Data", 1, 1, to_be_set_format);
  auto if1 = builder.AddNode("if", "If", 2, 2);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 2, 2);
  auto relu = builder.AddNode("relu", "Relu", 1, 1);

  builder.AddDataEdge(data1, 0, if1, 0);
  builder.AddDataEdge(data2, 0, if1, 1);
  builder.AddDataEdge(if1, 0, netoutput1, 0);
  builder.AddDataEdge(if1, 1, relu, 0);
  builder.AddDataEdge(relu, 0, netoutput1, 1);

  auto main_graph = builder.GetGraph();

  auto sub1 = BuildSubGraph("sub1", to_be_set_format_of_sub);
  sub1->SetParentGraph(main_graph);
  sub1->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub1");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(0, "sub1");
  main_graph->AddSubgraph("sub1", sub1);

  auto sub2 = BuildSubGraph("sub2");
  sub2->SetParentGraph(main_graph);
  sub2->SetParentNode(main_graph->FindNode("if"));
  main_graph->FindNode("if")->GetOpDesc()->AddSubgraphName("sub2");
  main_graph->FindNode("if")->GetOpDesc()->SetSubgraphInstanceName(1, "sub2");
  main_graph->AddSubgraph("sub2", sub2);

  return main_graph;
}
}
// Test BiasAdd special process
TEST_F(UTEST_FormatRefiner, biasadd_special_process) {
  auto builder = BuildGraph9();
  auto graph = builder.GetGraph();
  SetFirstInferFlag(graph, false);
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto square = graph->FindNode("square");
  auto biasadd = graph->FindNode("biasadd");
  EXPECT_EQ(square->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(square->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(biasadd->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NDHWC);
  EXPECT_EQ(biasadd->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NDHWC);
  SetFirstInferFlag(graph, true);
}
// only main graph own anchor point
TEST_F(UTEST_FormatRefiner, with_if_sub_graph_1) {
  auto main_graph = BuildMainGraphWithIf("main");
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(main_graph), GRAPH_SUCCESS);
  // check main graph format
  auto if1 = main_graph->FindNode("if");
  auto relu = main_graph->FindNode("relu");
  auto netoutput = main_graph->FindNode("netoutput");
  EXPECT_EQ(if1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(if1->GetOpDesc()->GetOutputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(if1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(if1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(netoutput->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  // check sub graph
  auto sub_graph_1 = main_graph->GetSubgraph("sub1");
  auto sub_graph_2 = main_graph->GetSubgraph("sub2");
  string prefix_1 = "sub1";
  string prefix_2 = "sub2";
  auto sub1_data_1 = sub_graph_1->FindNode(prefix_1 + "data1");
  auto sub1_data_2 = sub_graph_1->FindNode(prefix_1 + "data2");
  auto sub1_relu = sub_graph_1->FindNode(prefix_1 + "relu");
  auto sub1_sub = sub_graph_1->FindNode(prefix_1 + "sub");
  auto sub1_netoutput = sub_graph_1->FindNode(prefix_1 + "netoutput");
  auto sub2_data_1 = sub_graph_2->FindNode(prefix_2 + "data1");
  auto sub2_data_2 = sub_graph_2->FindNode(prefix_2 + "data2");
  auto sub2_relu = sub_graph_2->FindNode(prefix_2 + "relu");
  auto sub2_sub = sub_graph_2->FindNode(prefix_2 + "sub");
  auto sub2_netoutput = sub_graph_2->FindNode(prefix_2 + "netoutput");

  EXPECT_EQ(sub1_data_1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_data_1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_data_2->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_data_2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_relu->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_sub->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_sub->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_netoutput->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_netoutput->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_data_1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_data_1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_data_2->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_data_2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_relu->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_sub->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_sub->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_netoutput->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_netoutput->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
}

// only sub graph own anchor point
TEST_F(UTEST_FormatRefiner, with_if_sub_graph_2) {
  auto main_graph = BuildMainGraphWithIf("sub");
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(main_graph), GRAPH_SUCCESS);
  // check main graph format
  auto if1 = main_graph->FindNode("if");
  auto relu = main_graph->FindNode("relu");
  auto netoutput = main_graph->FindNode("netoutput");
  EXPECT_EQ(if1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(if1->GetOpDesc()->GetOutputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(if1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(if1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(netoutput->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  // check sub graph
  auto sub_graph_1 = main_graph->GetSubgraph("sub1");
  auto sub_graph_2 = main_graph->GetSubgraph("sub2");
  string prefix_1 = "sub1";
  string prefix_2 = "sub2";
  auto sub1_data_1 = sub_graph_1->FindNode(prefix_1 + "data1");
  auto sub1_data_2 = sub_graph_1->FindNode(prefix_1 + "data2");
  auto sub1_relu = sub_graph_1->FindNode(prefix_1 + "relu");
  auto sub1_sub = sub_graph_1->FindNode(prefix_1 + "sub");
  auto sub1_netoutput = sub_graph_1->FindNode(prefix_1 + "netoutput");
  auto sub2_data_1 = sub_graph_2->FindNode(prefix_2 + "data1");
  auto sub2_data_2 = sub_graph_2->FindNode(prefix_2 + "data2");
  auto sub2_relu = sub_graph_2->FindNode(prefix_2 + "relu");
  auto sub2_sub = sub_graph_2->FindNode(prefix_2 + "sub");
  auto sub2_netoutput = sub_graph_2->FindNode(prefix_2 + "netoutput");

  EXPECT_EQ(sub1_data_1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_data_1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_data_2->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_data_2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_relu->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_sub->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(sub1_sub->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NHWC);
  EXPECT_EQ(sub1_netoutput->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub1_netoutput->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_data_1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_data_1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_data_2->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_data_2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_relu->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_sub->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_sub->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_netoutput->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(sub2_netoutput->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
}

TEST_F(UTEST_FormatRefiner, data_format) {
  auto builder = BuildGraph8();
  auto graph = builder.GetGraph();
  SetFirstInferFlag(graph, false);
  graph->SaveDataFormat(FORMAT_NCHW);
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto data2 = graph->FindNode("data2");
  auto relu = graph->FindNode("relu");
  EXPECT_EQ(data2->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(data2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  SetFirstInferFlag(graph, true);
}

TEST_F(UTEST_FormatRefiner, constantFail) {
  auto builder = BuildGraph6();
  auto graph = builder.GetGraph();
  SetFirstInferFlag(graph, true);
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_FAILED);
}
TEST_F(UTEST_FormatRefiner, scalarNodesInfer) {
  auto builder = BuildGraph6();
  auto graph = builder.GetGraph();
  SetFirstInferFlag(graph, true);
  auto constant = graph->FindNode("constant");
  ge::GeTensorPtr value = std::make_shared<GeTensor>();
  AttrUtils::SetTensor(constant->GetOpDesc(), "value", value);
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
}

TEST_F(UTEST_FormatRefiner, ForwardAndDefaultInferFunc) {
  auto builder = BuildGraph1();
  auto graph = builder.GetGraph();
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto var2 = graph->FindNode("var2");
  EXPECT_EQ(var2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto relu1 = graph->FindNode("relu1");
  EXPECT_EQ(relu1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto netoutput1 = graph->FindNode("netoutput1");
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto conv1 = graph->FindNode("conv1");
  EXPECT_EQ(conv1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(conv1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(conv1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
}


TEST_F(UTEST_FormatRefiner, ForwardAndSpecifedInferFunc) {
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
  EXPECT_EQ(var2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto netoutput1 = graph->FindNode("netoutput1");
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
}

TEST_F(UTEST_FormatRefiner, FailedWhenInfer) {
  auto builder = BuildGraph1();
  auto graph = builder.GetGraph();
  auto relu1 = graph->FindNode("relu1");
  relu1->GetOpDesc()->AddInferFormatFunc([](Operator &op) {
    return GRAPH_FAILED;
  });

  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
}

TEST_F(UTEST_FormatRefiner, ForwardBackward) {
  auto builder = BuildGraph2();
  auto graph = builder.GetGraph();

  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto bn1 = graph->FindNode("bn1");
  EXPECT_EQ(bn1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(bn1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  for (auto name : {"var3", "var4", "var5", "var6"}) {
    auto node = graph->FindNode(name);
    EXPECT_EQ(node->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  }
}

TEST_F(UTEST_FormatRefiner, FormatConflict) {
  auto builder = BuildGraph3();
  auto graph = builder.GetGraph();
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
}

TEST_F(UTEST_FormatRefiner, InferStopND) {
  auto builder = BuildGraph1();
  auto graph = builder.GetGraph();
  auto relu1 = graph->FindNode("relu1");
  relu1->GetOpDesc()->AddInferFormatFunc([](Operator &op) {
    return GRAPH_SUCCESS;
  });
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto var2 = graph->FindNode("var2");
  EXPECT_EQ(var2->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  relu1 = graph->FindNode("relu1");
  EXPECT_EQ(relu1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto netoutput1 = graph->FindNode("netoutput1");
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto conv1 = graph->FindNode("conv1");
  EXPECT_EQ(conv1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(conv1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(conv1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
}

TEST_F(UTEST_FormatRefiner, InferStopSameFormat) {
  auto builder = BuildGraph4();
  auto graph = builder.GetGraph();
  EXPECT_EQ(FormatRefiner::InferOrigineFormat(graph), GRAPH_SUCCESS);

}

TEST_F(UTEST_FormatRefiner, ForwardMultiOutput) {
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
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto var1 = graph->FindNode("var1");
  EXPECT_EQ(var1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto bn1 = graph->FindNode("bn1");
  EXPECT_EQ(bn1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(bn1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto relu1 = graph->FindNode("relu1");
  EXPECT_EQ(relu1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relu1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto relug1 = graph->FindNode("relug1");
  EXPECT_EQ(relug1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(relug1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  auto bng1 = graph->FindNode("bng1");
  EXPECT_EQ(bng1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(bng1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(bng1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(bng1->GetOpDesc()->GetInputDesc(2).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(bng1->GetOpDesc()->GetInputDesc(3).GetOriginFormat(), FORMAT_NCHW);

  EXPECT_EQ(apply1->GetOpDesc()->GetOutputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(apply1->GetOpDesc()->GetInputDesc(0).GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(apply1->GetOpDesc()->GetInputDesc(1).GetOriginFormat(), FORMAT_NCHW);

}

TEST_F(UTEST_FormatRefiner, GetAnchorPointsFailed) {
  ge::ComputeGraphPtr graph = nullptr;
  std::vector<ge::NodePtr> anchor_points;
  std::vector<ge::NodePtr> data_nodes;
  auto status = FormatRefiner::GetAnchorPoints(graph, anchor_points, data_nodes);
  EXPECT_EQ(status, GRAPH_FAILED);
}

TEST_F(UTEST_FormatRefiner, AnchorProcessFailed) {
    ge::NodePtr anchor_node;
    auto status = FormatRefiner::AnchorProcess(anchor_node);
    EXPECT_EQ(status, GRAPH_FAILED);
}

TEST_F(UTEST_FormatRefiner, InferOrigineFormatFailed) {
  ge::ComputeGraphPtr graph = nullptr;
  auto status = FormatRefiner::InferOrigineFormat(graph);
  EXPECT_EQ(status, GRAPH_FAILED);
}
TEST_F(UTEST_FormatRefiner, SaveFormat) {
  auto builder = BuildGraph6();
  auto graph = builder.GetGraph();
  SetFirstInferFlag(graph, true);
  graph->SaveDataFormat(FORMAT_NHWC);
  auto save_format = graph->GetDataFormat();
  EXPECT_EQ(save_format, FORMAT_NHWC);
  graph->SaveDataFormat(FORMAT_ND);
}

}
