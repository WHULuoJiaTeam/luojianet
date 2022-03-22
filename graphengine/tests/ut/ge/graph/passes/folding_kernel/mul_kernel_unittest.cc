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
#include "host_kernels/mul_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
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

class UtestGraphPassesFoldingKernelMulKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelMulKernel, Int32Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Mul", "Mul");

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {5};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(MUL);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  int32_t *out_data = (int32_t *)outputs[0]->GetData().data();

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(out_data[0], 15);
}

TEST_F(UtestGraphPassesFoldingKernelMulKernel, DoubleNotchanged) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Mul", "Mul");

  vector<int64_t> dims_vec_0;
  vector<double> data_vec_0 = {3.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_DOUBLE);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(double));

  vector<int64_t> dims_vec_1;
  vector<double> data_vec_1 = {5.0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_DOUBLE);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(double));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(MUL);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  const double *output_data = reinterpret_cast<const double *>(outputs[0]->GetData().data());
  double diff = output_data[0] - (15.0);
  bool is_same = fabs(diff) < 0.00001 ? true : false;

  EXPECT_EQ(is_same, true);
  EXPECT_EQ(domi::SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelMulKernel, MulOverflow) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Mul", "Mul");

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {99999};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {21476};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(MUL);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelMulKernel, Int32OneDSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Mul", "Mul");

  vector<int64_t> dims_vec_0 = {2};
  vector<int32_t> data_vec_0 = {2, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  vector<int64_t> dims_vec_1 = {2};
  vector<int32_t> data_vec_1 = {5, 6};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(MUL);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelMulKernel, Uint32OneDSuccess) {
  OpDescPtr op_desc_ptr = nullptr;

  vector<int64_t> dims_vec_0 = {2};
  vector<uint32_t> data_vec_0 = {2, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UINT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint32_t));
  vector<int64_t> dims_vec_1 = {2};
  vector<uint32_t> data_vec_1 = {5, 6};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UINT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(uint32_t));
  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(MUL);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::PARAM_INVALID, status);

  op_desc_ptr = std::make_shared<OpDesc>("Mul", "Mul");
  status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelMulKernel, Uint32OneDInputEmpty) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Mul", "Mul");
  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(MUL);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelMulKernel, MulOptimizerErrtypeFail) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Mul", "Mul");
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UNDEFINED);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {4};
  vector<int32_t> data_vec_1 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MUL);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, NOT_CHANGED);

  vector<int64_t> dims_vec_2 = {4};
  vector<int32_t> data_vec_2 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input_other = {tensor_0, tensor_2};
  status = kernel->Compute(op_desc_ptr, input_other, outputs);
  EXPECT_EQ(status, NOT_CHANGED);

  vector<int64_t> dims_vec_3 = {4};
  vector<int32_t> data_vec_3 = {};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input_other3 = {tensor_0, tensor_3};
  status = kernel->Compute(op_desc_ptr, input_other3, outputs);
  EXPECT_EQ(status, NOT_CHANGED);
}
