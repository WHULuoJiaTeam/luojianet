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
#include "ops/non_max_suppression.h"
#include "ir/dtype/type.h"
#include "ir/value.h"
#include "abstract/dshape.h"
#include "utils/tensor_construct_utils.h"

namespace luojianet_ms {
namespace ops {

class TestNonMaxSuppression : public UT::Common {
 public:
  TestNonMaxSuppression() {}
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestNonMaxSuppression, test_ops_non_max_suppression1) {
  auto non_max_suppression = std::make_shared<NonMaxSuppression>();
  non_max_suppression->Init();
  EXPECT_EQ(non_max_suppression->get_center_point_box(), 0);
  auto input0 = TensorConstructUtils::CreateOnesTensor(kNumberTypeInt32, std::vector<int64_t>{1});
  MS_EXCEPTION_IF_NULL(input0);
  auto abstract = non_max_suppression->Infer({input0->ToAbstract()});
  MS_EXCEPTION_IF_NULL(abstract);
  EXPECT_EQ(abstract->isa<abstract::AbstractTensor>(), true);
  auto shape_ptr = abstract->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  EXPECT_EQ(shape_ptr->isa<abstract::Shape>(), true);
  auto shape = shape_ptr->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape);
  auto shape_vec = shape->shape();
  EXPECT_EQ(shape_vec.size(), 0);
  auto type = abstract->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  EXPECT_EQ(type->isa<TensorType>(), true);
  auto tensor_type = type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  auto data_type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(data_type);
  EXPECT_EQ(data_type->type_id(), kNumberTypeInt32);
}

}  // namespace ops
}  // namespace luojianet_ms
