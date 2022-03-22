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
#include "host_kernels/add_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/passes/constant_folding_pass.h"
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

class UtestFoldingKernelAddKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestFoldingKernelAddKernel, AddOptimizeInitSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, static_cast<int64_t>(DT_INT32));

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {4};
  vector<int32_t> data_vec_1 = {1, 2, 3, 4};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelAddKernel, AddOptimizerInt32Scalar) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {1};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestFoldingKernelAddKernel, AddOptimizerFloatSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_FLOAT);

  vector<int64_t> dims_vec_0 = {4};
  vector<float> data_vec_0 = {1.0, 2.0, 3.0, 4.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1;
  vector<float> data_vec_1 = {1.0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}

// optimize op of slice success
TEST_F(UtestFoldingKernelAddKernel, OptimizeOpOfSliceSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Add", ADD);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
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
  vector<GeTensorPtr> v_output;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(ADD);
  Status status = kernel->Compute(op_desc_ptr, input, v_output);

  EXPECT_EQ(NOT_CHANGED, status);
}
