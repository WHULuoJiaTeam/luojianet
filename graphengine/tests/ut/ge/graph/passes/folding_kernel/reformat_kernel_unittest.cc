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

#include "host_kernels/reformat_kernel.h"

#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "host_kernels/kernel_utils.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelReformatKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelReformatKernel, ComputeSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelReformatKernel, EmptyInput) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(ge::PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelReformatKernel, InputNullptr) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(nullptr, input, outputs);

  EXPECT_EQ(ge::PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelReformatKernel, InvalidInputsize) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(ge::PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelReformatKernel, MismatchShape) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelReformatKernel, MismatchDtype) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 1, 1, 1}), FORMAT_ND, DT_FLOAT16);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelReformatKernel, MismatchDataSize) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("ReFormat", "ReFormat");

  GeTensorDesc dims_tensor_desc(GeShape({1, 2, 3, 4}), FORMAT_ND, DT_FLOAT);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);
  GeTensorDesc dims_tensor_desc_in(GeShape({1, 2, 3, 4}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc(dims_tensor_desc_in);

  vector<int64_t> dims_vec_0 = {1, 1, 1, 1};
  vector<int32_t> data_vec_0 = {1, 1, 1, 1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), 1 * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(REFORMAT);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}
