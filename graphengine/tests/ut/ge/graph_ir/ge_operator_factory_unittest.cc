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

#include "../graph/ops_stub.h"
#include "operator_factory.h"

#define protected public
#define private public
#include "operator_factory_impl.h"
#undef private
#undef protected

using namespace ge;
class UtestGeOperatorFactory : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST(UtestGeOperatorFactory, create_operator) {
  Operator acosh = OperatorFactory::CreateOperator("acosh", "Acosh");
  EXPECT_EQ("Acosh", acosh.GetOpType());
  EXPECT_EQ("acosh", acosh.GetName());
  EXPECT_EQ(false, acosh.IsEmpty());
}

TEST(UtestGeOperatorFactory, create_operator_nullptr) {
  Operator abc = OperatorFactory::CreateOperator("abc", "ABC");
  EXPECT_EQ(true, abc.IsEmpty());
}

TEST(UtestGeOperatorFactory, get_infer_shape_func) {
  OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  InferShapeFunc infer_shape_func = OperatorFactoryImpl::GetInferShapeFunc("ABC");
  EXPECT_EQ(nullptr, infer_shape_func);
}

TEST(UtestGeOperatorFactory, get_verify_func) {
  OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  VerifyFunc verify_func = OperatorFactoryImpl::GetVerifyFunc("ABC");
  EXPECT_EQ(nullptr, verify_func);
}

TEST(UtestGeOperatorFactory, get_ops_type_list) {
  std::vector<std::string> all_ops;
  graphStatus status = OperatorFactory::GetOpsTypeList(all_ops);
  EXPECT_NE(0, all_ops.size());
  EXPECT_EQ(GRAPH_SUCCESS, status);
}

TEST(UtestGeOperatorFactory, is_exist_op) {
  graphStatus status = OperatorFactory::IsExistOp("Acosh");
  EXPECT_EQ(true, status);
  status = OperatorFactory::IsExistOp("ABC");
  EXPECT_EQ(false, status);
}

TEST(UtestGeOperatorFactory, register_func) {
  OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  graphStatus status = OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  EXPECT_EQ(GRAPH_FAILED, status);
  status = OperatorFactoryImpl::RegisterInferShapeFunc("ABC", nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, status);

  OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  status = OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  EXPECT_EQ(GRAPH_FAILED, status);
  status = OperatorFactoryImpl::RegisterVerifyFunc("ABC", nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, status);
}
/*
TEST(UtestGeOperatorFactory, get_ops_type_list_fail) {
  auto operator_creators_temp = OperatorFactoryImpl::operator_creators_;
  OperatorFactoryImpl::operator_creators_ = nullptr;
  std::vector<std::string> all_ops;
  graphStatus status = OperatorFactoryImpl::GetOpsTypeList(all_ops);
  EXPECT_EQ(GRAPH_FAILED, status);
  OperatorFactoryImpl::operator_creators_ = operator_creators_temp;
}
*/
