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
#include <iostream>

#define protected public
#define private public
#include "graph/op_desc.h"

#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/operator_factory.h"
#include "utils/op_desc_utils.h"
#undef protected
#undef private

using namespace std;
using namespace ge;

class UtestGeOpdesc : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeOpdesc, ge_test_opdesc_common) {
  string name = "Conv2d";
  string type = "Data";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  EXPECT_EQ(name, op_desc->GetName());
  EXPECT_EQ(type, op_desc->GetType());
  name = name + "_modify";
  type = type + "_modify";
  op_desc->SetName(name);
  op_desc->SetType(type);
  EXPECT_EQ(name, op_desc->GetName());
  EXPECT_EQ(type, op_desc->GetType());
}

TEST_F(UtestGeOpdesc, clear_all_output_desc) {
  auto g = std::make_shared<ge::ComputeGraph>("Test");

  // creat node
  ::ge::OpDescPtr desc = std::make_shared<ge::OpDesc>("", "");
  desc->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
  desc->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW));
  desc->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  auto node = g->AddNode(desc);
  bool ret = OpDescUtils::ClearOutputDesc(node);
  EXPECT_EQ(true, ret);
}

TEST_F(UtestGeOpdesc, clear_output_desc_by_index) {
  auto g = std::make_shared<ge::ComputeGraph>("Test");

  // creat node
  ::ge::OpDescPtr desc = std::make_shared<ge::OpDesc>("", "");
  desc->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW));
  desc->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW));
  desc->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  desc->AddOutputDesc("z", GeTensorDesc(GeShape({1, 1, 8, 8}), FORMAT_NCHW));
  auto node = g->AddNode(desc);
  bool ret = OpDescUtils::ClearOutputDesc(desc, 1);
  EXPECT_EQ(true, ret);
}

TEST_F(UtestGeOpdesc, ge_test_opdesc_inputs) {
  string name = "Conv2d";
  string type = "Data";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  GeTensorDesc te_desc1(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc(te_desc1));
  GeTensorDesc te_desc2(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc("w", te_desc2));
  GeTensorDesc te_desc3(GeShape({8, 9, 10, 11}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc("w", te_desc3));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc(1, te_desc3));
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddInputDesc(2, te_desc3));

  GeTensorDesc te_desc4(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(op_desc->UpdateInputDesc(1, te_desc4), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateInputDesc(4, te_desc4), GRAPH_FAILED);
  EXPECT_EQ(op_desc->UpdateInputDesc("w", te_desc4), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateInputDesc("weight", te_desc4), GRAPH_FAILED);

  GeTensorDesc get_te1 = op_desc->GetInputDesc(1);
  GeTensorDesc get_te2 = op_desc->GetInputDesc(4);
  GeTensorDesc get_te4 = op_desc->GetInputDesc("w");
  GeTensorDesc get_te3 = op_desc->GetInputDesc("weight");

  EXPECT_EQ(op_desc->GetInputNameByIndex(1), "w");
  EXPECT_EQ(op_desc->GetInputNameByIndex(3), "");

  auto vistor_in = op_desc->GetAllInputsDesc();
  EXPECT_EQ(vistor_in.size(), 3);

  auto input_size = op_desc->GetInputsSize();
  EXPECT_EQ(input_size, 3);
}

TEST_F(UtestGeOpdesc, ge_test_opdesc_outputs) {
  string name = "Conv2d";
  string type = "Data";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  GeTensorDesc te_desc1(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddOutputDesc(te_desc1));
  GeTensorDesc te_desc2(GeShape({4, 5, 6, 7}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_SUCCESS, op_desc->AddOutputDesc("w", te_desc2));
  GeTensorDesc te_desc3(GeShape({8, 9, 10, 11}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(GRAPH_FAILED, op_desc->AddOutputDesc("w", te_desc3));

  GeTensorDesc te_desc4(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  EXPECT_EQ(op_desc->UpdateOutputDesc(1, te_desc4), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateOutputDesc(4, te_desc4), GRAPH_FAILED);
  EXPECT_EQ(op_desc->UpdateOutputDesc("w", te_desc4), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->UpdateOutputDesc("weight", te_desc4), GRAPH_FAILED);

  GeTensorDesc get_te1 = op_desc->GetOutputDesc(1);
  GeTensorDesc get_te2 = op_desc->GetOutputDesc(4);
  GeTensorDesc get_te4 = op_desc->GetOutputDesc("w");
  GeTensorDesc get_te3 = op_desc->GetOutputDesc("weight");

  auto vistor_in = op_desc->GetAllOutputsDesc();
  EXPECT_EQ(vistor_in.size(), 2);
}

TEST_F(UtestGeOpdesc, ge_test_opdesc_attrs) {
  string name = "Conv2d";
  string type = "Data";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  auto defautl_attr_size = op_desc->GetAllAttrs().size();

  static const string PAD = "pad";
  static const string BIAS = "bias";

  op_desc->SetAttr(PAD, GeAttrValue::CreateFrom<GeAttrValue::INT>(6));
  op_desc->SetAttr(BIAS, GeAttrValue::CreateFrom<GeAttrValue::INT>(0));

  GeAttrValue at;
  EXPECT_EQ(op_desc->GetAttr(PAD, at), GRAPH_SUCCESS);
  int get_attr = -1;
  at.GetValue<GeAttrValue::INT>(get_attr);
  EXPECT_EQ(get_attr, 6);
  EXPECT_EQ(op_desc->GetAttr("xxx", at), GRAPH_FAILED);
  EXPECT_EQ(op_desc->GetAttr(BIAS, at), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->GetAttr("bia", at), GRAPH_FAILED);
  EXPECT_TRUE(op_desc->HasAttr(BIAS));
  EXPECT_FALSE(op_desc->HasAttr("xxx"));

  EXPECT_EQ(2, op_desc->GetAllAttrs().size() - defautl_attr_size);
  EXPECT_EQ(op_desc->DelAttr("xxx"), GRAPH_FAILED);
  EXPECT_EQ(op_desc->DelAttr(PAD), GRAPH_SUCCESS);
  EXPECT_EQ(1, op_desc->GetAllAttrs().size() - defautl_attr_size);
}

graphStatus InferFunctionStub(Operator &op) { return GRAPH_FAILED; }

TEST_F(UtestGeOpdesc, ge_test_opdesc_call_infer_func_failed) {
  GeTensorDesc ge_tensor_desc(GeShape({1, 2, 3, 4}), ge::FORMAT_NCHW, DT_FLOAT16);
  auto addn_op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  addn_op_desc->AddInputDesc(ge_tensor_desc);
  addn_op_desc->AddOutputDesc(ge_tensor_desc);
  addn_op_desc->AddInferFunc(InferFunctionStub);
  auto graph = std::make_shared<ComputeGraph>("test");
  auto addn_node = std::make_shared<Node>(addn_op_desc, graph);
  addn_node->Init();
  Operator op = OpDescUtils::CreateOperatorFromNode(addn_node);

  graphStatus ret = addn_op_desc->CallInferFunc(op);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

graphStatus InferFunctionSuccessStub(Operator &op) { return GRAPH_SUCCESS; }

TEST_F(UtestGeOpdesc, ge_test_opdesc_call_infer_func_success) {
  auto addn_op_desc = std::make_shared<OpDesc>("AddN", "AddN");
  addn_op_desc->AddInferFunc(InferFunctionSuccessStub);
  auto graph = std::make_shared<ComputeGraph>("test");
  auto addn_node = std::make_shared<Node>(addn_op_desc, graph);
  addn_node->Init();
  Operator op = OpDescUtils::CreateOperatorFromNode(addn_node);

  graphStatus ret = addn_op_desc->CallInferFunc(op);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGeOpdesc, ge_test_opdesc_infer_shape_and_type) {
  auto addn_op_desc = std::make_shared<OpDesc>("name", "type");
  graphStatus ret = addn_op_desc->InferShapeAndType();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGeOpdesc, default_infer_format_success) {
  auto addn_op_desc = std::make_shared<OpDesc>("name", "type");
  std::function<graphStatus(Operator &)> func = nullptr;
  addn_op_desc->AddInferFormatFunc(func);
  auto fun1 = addn_op_desc->GetInferFormatFunc();
  graphStatus ret = addn_op_desc->DefaultInferFormat();
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGeOpdesc, call_infer_format_func_success) {
  auto addn_op_desc = std::make_shared<OpDesc>("name", "type");
  Operator op;
  graphStatus ret = addn_op_desc->CallInferFormatFunc(op);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGeOpdesc, add_dynamic_output_desc) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddDynamicOutputDesc("x", 1, false), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddDynamicOutputDesc("x1", 1, false), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddDynamicOutputDesc("x", 1, false), GRAPH_FAILED);

  OpDescPtr desc_ptr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(desc_ptr2->AddDynamicOutputDesc("x", 1), GRAPH_SUCCESS);
}
