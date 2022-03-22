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

#define protected public
#define private public
#include "host_kernels/sub_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

class UtestFoldingKernelSubKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

template <typename T>
void AssembleInput(vector<ConstGeTensorPtr> &input, const GeTensorDesc &sub_desc, const T &sub1, const T &sub2) {
  input.clear();
  T sub_0_value[1] = {sub1};
  ConstGeTensorPtr sub_0 = std::make_shared<ge::GeTensor>(sub_desc, (uint8_t *)sub_0_value, 1 * sizeof(T));

  T sub_1_value[1] = {sub2};
  ConstGeTensorPtr sub_1 = std::make_shared<ge::GeTensor>(sub_desc, (uint8_t *)sub_1_value, 1 * sizeof(T));

  input.push_back(sub_0);
  input.push_back(sub_1);
}

/// test func：SubKernel::Compute
/// case：optimize op of SUB
/// result： success
TEST_F(UtestFoldingKernelSubKernel, ComSuccessFloat) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_FLOAT);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (float)5, (float)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}
TEST_F(UtestFoldingKernelSubKernel, ComSuccessInt16) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_INT16);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (int16_t)5, (int16_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestFoldingKernelSubKernel, ComSuccessDouble) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_DOUBLE);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (double)5, (double)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestFoldingKernelSubKernel, ComSuccessInt8) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_INT8);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (int8_t)5, (int8_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestFoldingKernelSubKernel, ComSuccessUint8) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_UINT8);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (uint8_t)5, (uint8_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestFoldingKernelSubKernel, ComSuccessUint16) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_UINT16);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (uint16_t)5, (uint16_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestFoldingKernelSubKernel, ComSuccessInt32) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_INT32);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (int32_t)5, (int32_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}
TEST_F(UtestFoldingKernelSubKernel, ComSuccessInt64) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_INT64);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (int64_t)5, (int64_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestFoldingKernelSubKernel, ComSuccessUint32) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_UINT32);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (uint32_t)5, (uint32_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestFoldingKernelSubKernel, ComSuccessUint64) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_UINT64);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (uint64_t)5, (uint64_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestFoldingKernelSubKernel, StringComFailNotChange) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({1});
  GeTensorDesc sub_desc(sub_shape, ge::FORMAT_NCHW, DT_STRING);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);

  // float
  AssembleInput(input, sub_desc, (int64_t)5, (int64_t)3);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
}

/// test func：SubKernel::Compute
/// case：optimize op of SUB
/// result： failed
TEST_F(UtestFoldingKernelSubKernel, ComFailed) {
  OpDescPtr test_op = std::make_shared<OpDesc>("test", "Test");
  vector<ConstGeTensorPtr> input;
  vector<GeTensorPtr> v_output;
  ge::GeShape sub_shape({0});
  GeTensorDesc sub_desc(sub_shape);
  float sub_0_value[1] = {3.0};
  ConstGeTensorPtr sub_0 = std::make_shared<ge::GeTensor>(sub_desc, (uint8_t *)sub_0_value, 1 * sizeof(float));

  ge::GeShape sub_shape2({1});
  GeTensorDesc sub_desc2(sub_shape2);
  float sub_1_value[1] = {5.0};
  ConstGeTensorPtr sub_1 = std::make_shared<ge::GeTensor>(sub_desc2, (uint8_t *)sub_1_value, 1 * sizeof(float));

  input.push_back(sub_0);
  input.push_back(sub_1);

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SUB);
  test_op->AddOutputDesc(sub_desc);
  Status status = kernel->Compute(test_op, input, v_output);

  EXPECT_EQ(SUCCESS, status);
}
