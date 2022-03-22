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
#include "host_kernels/rsqrt_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/attr_value_util.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

class UtestFoldingKernelRsqrtKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// optimize op of sqrt success
TEST_F(UtestFoldingKernelRsqrtKernel, RsqrtOptimizerSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("RSQRT", RSQRT);

  vector<int64_t> dims_vec_0 = {3, 2};
  vector<float> data_vec_0 = {4.0, 16.0, 25.0, 100.0, 400.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RSQRT);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  float *outdata = (float *)outputs[0]->GetData().data();

  EXPECT_EQ(SUCCESS, status);
  EXPECT_FLOAT_EQ(outdata[0], 0.5);
  EXPECT_FLOAT_EQ(outdata[1], 0.25);
  EXPECT_FLOAT_EQ(outdata[2], 0.2);
  EXPECT_FLOAT_EQ(outdata[3], 0.1);
  EXPECT_FLOAT_EQ(outdata[4], 0.05);
}

// optimize op of sqrt fail(include 0)
TEST_F(UtestFoldingKernelRsqrtKernel, RsqrtOptimizerHasZero) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("RSQRT", RSQRT);

  vector<int64_t> dims_vec_0 = {2};
  vector<float> data_vec_0 = {4.0, 0.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RSQRT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}
