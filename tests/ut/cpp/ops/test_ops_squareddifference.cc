/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <vector>
#include <memory>
#include "common/common_test.h"
#include "ops/squared_difference.h"
#include "ir/dtype/type.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"
namespace luojianet_ms {
namespace ops {
class TestSquaredDifference : public UT::Common {
 public:
  TestSquaredDifference() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestSquaredDifference, test_ops_squareddifference) {
  auto squareddifference = std::make_shared<SquaredDifference>();
  squareddifference->Init();
  auto tensor_x = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 3});
  auto tensor_y = TensorConstructUtils::CreateOnesTensor(kNumberTypeFloat32, std::vector<int64_t>{1, 3});
  MS_EXCEPTION_IF_NULL(tensor_x);
  MS_EXCEPTION_IF_NULL(tensor_y);
  auto squareddifference_abstract = squareddifference->Infer({tensor_x->ToAbstract(), tensor_y->ToAbstract()});
  MS_EXCEPTION_IF_NULL(squareddifference_abstract);
  EXPECT_EQ(squareddifference_abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = squareddifference_abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto squareddifference_shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(squareddifference_shape);
  auto shape_vec = squareddifference_shape->shape();
  auto type = squareddifference_abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto elem_type = tensor_type->element();
  EXPECT_EQ(elem_type->type_id(), kNumberTypeFloat32);
  EXPECT_EQ(shape_vec.size(), 2);
  EXPECT_EQ(shape_vec[0], 1);
  EXPECT_EQ(shape_vec[1], 3);
}

}  // namespace ops
}  // namespace luojianet_ms
