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
#include <memory>
#include <vector>
#include <string>
#include "ir/primitive.h"
#include "utils/utils.h"
#include "abstract/abstract_value.h"
#include "abstract/primitive_infer_map.h"
#include "backend/optimizer/common/const_input_to_attr_registry.h"
#include "backend/optimizer/common/helper.h"
#include "common/common_test.h"
namespace luojianet_ms {
namespace opt {
class TestAttr : public ops::PrimitiveC {
 public:
  TestAttr() : PrimitiveC("") {}
};
class TestDynamicInput : public ops::PrimitiveC {
 public:
  TestDynamicInput() : PrimitiveC("") {}
};
constexpr auto kAttrConvertTestName = "attr_convert_test";
constexpr auto kDynamicInputTestName = "dynamic_input_test";
inline const PrimitivePtr kPrimAttrConvertTest = std::make_shared<Primitive>(kAttrConvertTestName);
inline const PrimitivePtr kPrimDynamicInputTest = std::make_shared<Primitive>("dynamic_input_test");
AbstractBasePtr InferImplAttrTest(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list) {
  EXPECT_EQ(args_spec_list.size(), 3);
  EXPECT_NE(args_spec_list[1], nullptr);
  EXPECT_EQ(args_spec_list[1]->isa<abstract::AbstractTuple>(), true);
  return args_spec_list[0];
}
REGISTER_PRIMITIVE_EVAL_IMPL(TestAttr, kPrimAttrConvertTest, InferImplAttrTest, nullptr, true);
AbstractBasePtr InferImplDynamicInputTest(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list) {
  EXPECT_EQ(args_spec_list.size(), 3);
  EXPECT_NE(args_spec_list[1], nullptr);
  EXPECT_EQ(args_spec_list[1]->isa<abstract::AbstractTuple>(), true);
  auto item = args_spec_list[1]->cast<abstract::AbstractTuplePtr>();
  return args_spec_list[0];
}
REGISTER_PRIMITIVE_EVAL_IMPL(TestDynamicInput, kPrimDynamicInputTest, InferImplDynamicInputTest, nullptr, true);
class TestAttrAndDynamicBackendInfer : public UT::Common {
 public:
  TestAttrAndDynamicBackendInfer() {}
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(TestAttrAndDynamicBackendInfer, test_attr_and_dynamic_input_infer) {
  // Register Attr for ut
  ConstInputToAttrInfoRegistry &reg = ConstInputToAttrInfoRegistry::Instance();
  reg.Register(kAttrConvertTestName, {1});
  // call primitive
  PrimitivePtr prim_attr_test = std::make_shared<Primitive>(kAttrConvertTestName);
  PrimitivePtr prim_dynamic_input_test = std::make_shared<Primitive>(kDynamicInputTestName);
  // set primtive attr
  auto input_names = std::vector<std::string>{"a", "b", "c"};
  auto attr_name = "b";
  auto attr = MakeValue(std::vector<int>{1, 2, 3});
  prim_attr_test->AddAttr(kAttrInputNames, MakeValue(input_names));
  prim_attr_test->AddAttr(attr_name, attr);
  // set dynameic input list for primtive
  std::vector<int64_t> dynamic_input_list = {-1, 2, -1};
  prim_dynamic_input_test->AddAttr(kAttrDynInputSizes, MakeValue(dynamic_input_list));
  // call Abstract list
  auto abs_a = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{2, 2, 2, 2});
  auto abs_c = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{2, 2, 2, 2});
  auto attr_infer_result = CppInferShape(prim_attr_test, {abs_a, abs_c});
  auto abs_dynamic_a = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{2, 2, 2, 2});
  auto abs_dynamic_b = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{2, 2, 2, 2});
  auto abs_dynamic_c = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{2, 2, 2, 2});
  auto abs_dynamic_d = std::make_shared<abstract::AbstractTensor>(kFloat32, std::vector<int64_t>{2, 2, 2, 2});
  auto dynamic_infer_result =
    CppInferShape(prim_dynamic_input_test, {abs_dynamic_a, abs_dynamic_b, abs_dynamic_c, abs_dynamic_d});
}
}  // namespace opt
}  // namespace luojianet_ms
