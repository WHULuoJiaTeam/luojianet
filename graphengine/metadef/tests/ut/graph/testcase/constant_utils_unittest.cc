/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "graph/ge_tensor.h"
#include "graph/tensor.h"
#include "graph/utils/constant_utils.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
class UtestConstantUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestConstantUtils, TestIsConstant) {
  // check node is constant
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  const auto &const1 = builder.AddNode("const1", "Constant", 0, 1);
  ASSERT_EQ(ConstantUtils::IsConstant(const1), true);

  // check operator is constant
  const auto &const2 = builder.AddNode("const1", "Const", 0, 1);
  auto const_op2 = OpDescUtils::CreateOperatorFromNode(const2);
  ASSERT_EQ(ConstantUtils::IsConstant(const_op2), true);

  // check normal op is not constant
  const auto &cast = builder.AddNode("cast", "Cast", 1, 1);
  auto cast_op = OpDescUtils::CreateOperatorFromNode(cast);
  ASSERT_EQ(ConstantUtils::IsConstant(cast), false);
}

TEST_F(UtestConstantUtils, TestNodeIsPotentialConstant) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  // check potential const
  const auto &shape_node = builder.AddNode("shape", "Shape", 1, 1);
  AttrUtils::SetBool(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  AttrUtils::SetListInt(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0});
  AttrUtils::SetListTensor(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT, {tensor});
  ASSERT_EQ(ConstantUtils::IsConstant(shape_node), true);
  ASSERT_EQ(ConstantUtils::IsPotentialConst(shape_node->GetOpDesc()), true);

  // check const is not potential const
  const auto &const1 = builder.AddNode("const1", "Constant", 0, 1);
  ASSERT_EQ(ConstantUtils::IsPotentialConst(const1->GetOpDesc()), false);

  // check normal node is not potential const
  const auto &cast = builder.AddNode("cast", "Cast", 1, 1);
  ASSERT_EQ(ConstantUtils::IsPotentialConst(cast->GetOpDesc()), false);
}

TEST_F(UtestConstantUtils, TestGetWeightFromOpDesc) {
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);

  // build two const
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  const auto &shape_node = builder.AddNode("shape_node", "Shape", 1, 1);
  AttrUtils::SetBool(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  AttrUtils::SetListInt(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0});
  AttrUtils::SetListTensor(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT, {tensor});
  const auto &const_node = builder.AddNode("const_node", "Const", 0, 1);
  OpDescUtils::SetWeights(const_node, {tensor});

  // get weight from real const
  ConstGeTensorPtr weight;
  ASSERT_TRUE(ConstantUtils::GetWeight(const_node->GetOpDesc(), 0, weight));
  ASSERT_EQ(weight->GetTensorDesc().GetDataType(), DT_UINT8);
  ASSERT_EQ(weight->GetTensorDesc().GetShape().GetDims(), shape);

  // get weight from potential const
  ConstGeTensorPtr potential_weight;
  ASSERT_TRUE(ConstantUtils::GetWeight(shape_node->GetOpDesc(), 0, potential_weight));
  ASSERT_EQ(potential_weight->GetTensorDesc().GetDataType(), DT_UINT8);
  ASSERT_EQ(potential_weight->GetTensorDesc().GetShape().GetDims(), shape);

  // check invalid potential const get weight
  // build potential op
  const auto &shape_node_2 = builder.AddNode("shape_node_2", "Shape", 1, 1);
  AttrUtils::SetBool(shape_node_2->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  ASSERT_FALSE(ConstantUtils::GetWeight(shape_node_2->GetOpDesc(), 0, potential_weight));
  AttrUtils::SetListInt(shape_node_2->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0});
  ASSERT_FALSE(ConstantUtils::GetWeight(shape_node_2->GetOpDesc(), 0, potential_weight));
  AttrUtils::SetListInt(shape_node_2->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0, 1});
  AttrUtils::SetListTensor(shape_node_2->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT, {tensor});
  ASSERT_FALSE(ConstantUtils::GetWeight(shape_node_2->GetOpDesc(), 0, potential_weight));
}

TEST_F(UtestConstantUtils, TestGetWeightFromOperator) {
  // new a tensor
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  TensorDesc tensor_desc(Shape(shape), FORMAT_ND, DT_UINT8);
  Tensor tensor(tensor_desc, value);

  // build potential op
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  const auto &shape_node = builder.AddNode("shape_node", "Shape", 1, 1);
  auto shape_op = OpDescUtils::CreateOperatorFromNode(shape_node);
  shape_op.SetAttr(ATTR_NAME_POTENTIAL_CONST, true);
  shape_op.SetAttr(ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0});
  vector<Tensor> weights = {tensor};
  shape_op.SetAttr(ATTR_NAME_POTENTIAL_WEIGHT, weights);

  // get weight from potential const
  ASSERT_TRUE(ConstantUtils::IsConstant(shape_op));
  Tensor potential_weight;
  ASSERT_TRUE(ConstantUtils::GetWeight(shape_op, 0, potential_weight));
  ASSERT_EQ(potential_weight.GetTensorDesc().GetDataType(), DT_UINT8);
  ASSERT_EQ(potential_weight.GetTensorDesc().GetShape().GetDims(), shape);

  // check invalid potential const get weight
  // build potential op
  const auto &shape_node_2 = builder.AddNode("shape_node_2", "Shape", 1, 1);
  auto shape_op_2 = OpDescUtils::CreateOperatorFromNode(shape_node_2);
  shape_op_2.SetAttr(ATTR_NAME_POTENTIAL_CONST, true);
  ASSERT_FALSE(ConstantUtils::GetWeight(shape_op_2, 0, potential_weight));
  shape_op_2.SetAttr(ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0});
  ASSERT_FALSE(ConstantUtils::GetWeight(shape_op_2, 0, potential_weight));
  shape_op_2.SetAttr(ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0,1});
  shape_op_2.SetAttr(ATTR_NAME_POTENTIAL_WEIGHT, weights);
  ASSERT_FALSE(ConstantUtils::GetWeight(shape_op_2, 0, potential_weight));
}

TEST_F(UtestConstantUtils, TestSetWeight) {
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);

  // build two const
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  const auto &shape_node = builder.AddNode("shape_node", "Shape", 1, 1);
  AttrUtils::SetBool(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  AttrUtils::SetListInt(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0});
  AttrUtils::SetListTensor(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT, {tensor});
  const auto &const_node = builder.AddNode("const_node", "Constant", 0, 1);

  // set weight to real const
  auto ret = ConstantUtils::SetWeight(const_node->GetOpDesc(), 0, tensor);
  ASSERT_EQ(ret, true);

  // set weight to potential const
  ConstGeTensorPtr potential_weight;
  ret = ConstantUtils::SetWeight(shape_node->GetOpDesc(), 0, tensor);
  ASSERT_EQ(ret, true);
  // check weight index is out of range
  ret = ConstantUtils::SetWeight(shape_node->GetOpDesc(), 1, tensor);
  ASSERT_EQ(ret, false);
}
TEST_F(UtestConstantUtils, TestMarkUnmarkPotentialConst) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  const auto &shape_node = builder.AddNode("shape_node", "Shape", 1, 1);
  vector<int> indices = {0};
  // new a tensor
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);
  vector<GeTensorPtr> weights = {tensor};
  // test normal case mark potential const
  ASSERT_TRUE(ConstantUtils::MarkPotentialConst(shape_node->GetOpDesc(), indices, weights));
  bool is_potential_const = false;
  ASSERT_TRUE(AttrUtils::GetBool(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, is_potential_const));
  ASSERT_TRUE(is_potential_const);
  // test normal case unmark potential const
  ASSERT_TRUE(ConstantUtils::UnMarkPotentialConst(shape_node->GetOpDesc()));
  ASSERT_FALSE(AttrUtils::GetBool(shape_node->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, is_potential_const));

  // test mark potential const : indices not match weights
  const auto &shape_node2 = builder.AddNode("shape_node", "Shape", 1, 1);
  ASSERT_FALSE(ConstantUtils::MarkPotentialConst(shape_node->GetOpDesc(), {0,1}, weights));
}
}  // namespace ge
