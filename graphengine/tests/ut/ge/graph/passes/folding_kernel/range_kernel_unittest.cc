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

#include "framework/common/ge_inner_error_codes.h"

#define protected public
#define private public
#include "host_kernels/range_kernel.h"

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

class UtestGraphPassesFoldingKernelRangeKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelRangeKernel, Int32Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Range", RANGE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  int32_t start = 1, limit = 20, delta = 2;

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RANGE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 1);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDim(0), 10);

  int32_t *val = (int32_t *)outputs[0]->GetData().data();
  for (int64_t i = 0; i < outputs[0]->GetTensorDesc().GetShape().GetDim(0); ++i) {
    EXPECT_EQ(val[i], start + delta * i);
  }
}

TEST_F(UtestGraphPassesFoldingKernelRangeKernel, FloatSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Range", RANGE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_FLOAT);

  float start = 0.0, limit = 10.0, delta = 1.0;

  vector<int64_t> dims_vec_0;
  vector<float> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1;
  vector<float> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<int64_t> dims_vec_2;
  vector<float> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RANGE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 1);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDim(0), 10);

  float *val = (float *)outputs[0]->GetData().data();
  for (int64_t i = 0; i < outputs[0]->GetTensorDesc().GetShape().GetDim(0); ++i) {
    EXPECT_EQ(val[i], start + delta * (float)i);
  }
}

TEST_F(UtestGraphPassesFoldingKernelRangeKernel, FloatSuccess1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Range", RANGE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_FLOAT);

  float start = 0.0, limit = 2.0, delta = 1.2345678;

  vector<int64_t> dims_vec_0;
  vector<float> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1;
  vector<float> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<int64_t> dims_vec_2;
  vector<float> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RANGE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 1);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDim(0), 2);

  float *val = (float *)outputs[0]->GetData().data();
  for (int64_t i = 0; i < outputs[0]->GetTensorDesc().GetShape().GetDim(0); ++i) {
    EXPECT_EQ(val[i], start + delta * (float)i);
  }
}

TEST_F(UtestGraphPassesFoldingKernelRangeKernel, DataTypeNotSupport) {
  OpDescPtr op_desc_ptr = nullptr;

  int8_t start = 1, limit = 1, delta = 1;

  vector<int64_t> dims_vec_0;
  vector<int8_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT8);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int8_t));

  vector<int64_t> dims_vec_1;
  vector<int8_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT8);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int8_t));

  vector<int64_t> dims_vec_2;
  vector<int8_t> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT8);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int8_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RANGE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(PARAM_INVALID, status);

  op_desc_ptr = std::make_shared<OpDesc>("Range", RANGE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT8);
  status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelRangeKernel, DimSizeIsZero) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Range", RANGE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  int32_t start = 1, limit = 1, delta = 1;

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RANGE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 0);
}

TEST_F(UtestGraphPassesFoldingKernelRangeKernel, ErrorTest) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Range", RANGE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  int32_t start = 1, limit = 1;

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<uint32_t> data_vec_2 = {1};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_UINT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(uint32_t));

  vector<int64_t> dims_vec_2_zero;
  vector<int32_t> data_vec_2_zero = {};
  GeTensorDesc tensor_desc_2_zero(GeShape(dims_vec_2_zero), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2_zero = std::make_shared<GeTensor>(tensor_desc_2_zero, (uint8_t *)data_vec_2_zero.data(),
                                                              data_vec_2_zero.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RANGE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);

  vector<ConstGeTensorPtr> input_has_zero = {tensor_0, tensor_1, tensor_2_zero};
  status = kernel->Compute(op_desc_ptr, input_has_zero, outputs);
  EXPECT_EQ(NOT_CHANGED, status);

  vector<ConstGeTensorPtr> input_has_diff = {tensor_0, tensor_1, tensor_2};
  status = kernel->Compute(op_desc_ptr, input_has_diff, outputs);
  EXPECT_EQ(NOT_CHANGED, status);

  vector<int64_t> dims_vec_3 = {2, 3};
  vector<int32_t> data_vec_3 = {1, -3, 3, -3, 3, -2};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input_not_scalar = {tensor_0, tensor_1, tensor_3};
  status = kernel->Compute(op_desc_ptr, input_not_scalar, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelRangeKernel, NegativeSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Range", RANGE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  int32_t start = -20, limit = 0, delta = 3;

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RANGE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 1);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDim(0), 7);
}

TEST_F(UtestGraphPassesFoldingKernelRangeKernel, RangeError) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Range", RANGE);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  int32_t start = 10, limit = 1, delta = 1;

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {start};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {limit};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {delta};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RANGE);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(PARAM_INVALID, status);
}
