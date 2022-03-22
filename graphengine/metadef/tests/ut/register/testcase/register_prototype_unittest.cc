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

#include "register/prototype_pass_registry.h"
namespace ge {
class UtestProtoTypeRegister : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class RegisterPass : public ProtoTypeBasePass {
 public:
  Status Run(google::protobuf::Message *message) { return SUCCESS; }
};

class RegisterFail : public ProtoTypeBasePass {
 public:
  Status Run(google::protobuf::Message *message) { return FAILED; }
};

REGISTER_PROTOTYPE_PASS("RegisterPass", RegisterPass, domi::CAFFE);
REGISTER_PROTOTYPE_PASS("RegisterPass", RegisterPass, domi::CAFFE);

TEST_F(UtestProtoTypeRegister, register_test) {
  auto pass_vec = ProtoTypePassRegistry::GetInstance().GetCreateFnByType(domi::CAFFE);
  EXPECT_EQ(pass_vec.size(), 1);
}

TEST_F(UtestProtoTypeRegister, register_test_fail) {
  REGISTER_PROTOTYPE_PASS(nullptr, RegisterPass, domi::CAFFE);
  REGISTER_PROTOTYPE_PASS("RegisterFail", RegisterFail, domi::CAFFE);

  ProtoTypePassRegistry::GetInstance().RegisterProtoTypePass(nullptr, nullptr, domi::CAFFE);
  auto pass_vec = ProtoTypePassRegistry::GetInstance().GetCreateFnByType(domi::CAFFE);
  EXPECT_NE(pass_vec.size(), 1);
}
}  // namespace ge