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
#include "graph/utils/op_desc_utils.h"

#include "debug/ge_op_types.h"
#include "graph/compute_graph.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#undef protected
#undef private

using namespace std;
using namespace ge;

class UtestGeOpdescUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeOpdescUtils, CreateOperatorFromDesc) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);
  GeAttrValue test_attr = GeAttrValue::CreateFrom<GeAttrValue::INT>(1);
  desc_ptr->SetAttr("test_attr", std::move(test_attr));

  Operator oprt = OpDescUtils::CreateOperatorFromOpDesc(desc_ptr);

  GeAttrValue::INT out;
  oprt.GetAttr("test_attr", out);
  EXPECT_EQ(out, 1);

  TensorDesc input_desc1 = oprt.GetInputDesc("x");
  EXPECT_TRUE(input_desc1.GetShape().GetDimNum() == 4);
  EXPECT_TRUE(input_desc1.GetShape().GetDim(0) == 1);
  EXPECT_TRUE(input_desc1.GetShape().GetDim(1) == 16);
  EXPECT_TRUE(input_desc1.GetShape().GetDim(2) == 16);
  EXPECT_TRUE(input_desc1.GetShape().GetDim(3) == 16);

  TensorDesc input_desc2 = oprt.GetInputDesc(1);
  EXPECT_TRUE(input_desc2.GetShape().GetDimNum() == 4);
  EXPECT_TRUE(input_desc2.GetShape().GetDim(0) == 1);
  EXPECT_TRUE(input_desc2.GetShape().GetDim(1) == 1);
  EXPECT_TRUE(input_desc2.GetShape().GetDim(2) == 1);
  EXPECT_TRUE(input_desc2.GetShape().GetDim(3) == 1);

  OpDescPtr out_ptr = OpDescUtils::GetOpDescFromOperator(oprt);
  EXPECT_TRUE(out_ptr == desc_ptr);

  string name1 = out_ptr->GetName();
  string name2 = oprt.GetName();
  EXPECT_TRUE(name1 == name2);
}

TEST_F(UtestGeOpdescUtils, clear_input_desc) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr desc_ptr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(desc_ptr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);
  NodePtr n2 = graph_ptr->AddNode(desc_ptr);
  EXPECT_TRUE(OpDescUtils::ClearInputDesc(n1));
  EXPECT_TRUE(OpDescUtils::ClearInputDesc(desc_ptr2, 0));
}

TEST_F(UtestGeOpdescUtils, mutable_weights) {
  OpDescPtr desc_ptr = std::make_shared<OpDesc>("name1", CONSTANT);
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  OpDescPtr desc_ptr2 = std::make_shared<OpDesc>("name2", "type2");
  EXPECT_EQ(desc_ptr2->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr2->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  ComputeGraphPtr graph_ptr = std::make_shared<ComputeGraph>("name");
  NodePtr n1 = graph_ptr->AddNode(desc_ptr);
  NodePtr n2 = graph_ptr->AddNode(desc_ptr);

  float f[1] = {1.0};
  GeTensorDesc tensor_desc(GeShape({1}));
  GeTensorPtr tensor = std::make_shared<GeTensor>(tensor_desc, (const uint8_t *)f, 1 * sizeof(float));

  OpDescPtr null_opdesc = nullptr;

  EXPECT_EQ(GRAPH_PARAM_INVALID, OpDescUtils::SetWeights(desc_ptr, nullptr));
  EXPECT_EQ(GRAPH_SUCCESS, OpDescUtils::SetWeights(desc_ptr, tensor));
  EXPECT_EQ(GRAPH_SUCCESS, OpDescUtils::SetWeights(*desc_ptr2.get(), tensor));
  EXPECT_EQ(GRAPH_FAILED, OpDescUtils::SetWeights(*desc_ptr2.get(), nullptr));

  EXPECT_NE(nullptr, OpDescUtils::MutableWeights(desc_ptr));
  EXPECT_NE(nullptr, OpDescUtils::MutableWeights(*desc_ptr.get()));

  EXPECT_EQ(nullptr, OpDescUtils::MutableWeights(null_opdesc));

  EXPECT_EQ(nullptr, OpDescUtils::CreateOperatorFromOpDesc(desc_ptr));

  auto tensor_vec = OpDescUtils::GetWeights(n1);
  EXPECT_NE(0, tensor_vec.size());
}
