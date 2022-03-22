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
#include "host_kernels/maximum_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelMaximumKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerIntSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0 = {2, 3};
  vector<int32_t> data_vec_0 = {1, 2, 3, 5, 2, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<int32_t> data_vec_1 = {1, 2, 3, 4, 5, 6};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);

  GeTensorPtr out = outputs[0];
  vector<int32_t> data_y = {1, 2, 3, 5, 5, 6};
  EXPECT_EQ(out->GetData().size(), 24);
  size_t one_size = sizeof(int32_t);
  size_t out_nums = out->GetData().size() / one_size;
  for (size_t i = 0; i < out_nums; i++) {
    int32_t *one_val = (int32_t *)(out->GetData().data() + i * one_size);
    EXPECT_EQ(data_y[i], *one_val);
  }
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerIntSuccess2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0 = {1, 3};
  vector<int32_t> data_vec_0 = {1, 2, 3};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {2, 3};
  vector<int32_t> data_vec_1 = {1, 2, 3, 4, 5, 6};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);

  GeTensorPtr out = outputs[0];
  vector<int32_t> data_y = {1, 2, 3, 4, 5, 6};
  EXPECT_EQ(out->GetData().size(), 24);
  size_t one_size = sizeof(int32_t);
  size_t out_nums = out->GetData().size() / one_size;
  for (size_t i = 0; i < out_nums; i++) {
    int32_t *one_val = (int32_t *)(out->GetData().data() + i * one_size);
    EXPECT_EQ(data_y[i], *one_val);
  }
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumScalarSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int32_t> data_vec_1 = {2};
  GeTensorDesc tensor_desc_1(GeShape(), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);

  GeTensorPtr out = outputs[0];
  EXPECT_EQ(out->GetData().size(), 4);
  vector<int32_t> data_y = {2};
  size_t one_size = sizeof(int32_t);
  size_t out_nums = out->GetData().size() / one_size;
  for (size_t i = 0; i < out_nums; i++) {
    int32_t *one_val = (int32_t *)(out->GetData().data() + i * one_size);
    EXPECT_EQ(data_y[i], *one_val);
  }
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerInt8Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT8);

  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<int8_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT8);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int8_t));

  vector<int64_t> dims_vec_1 = {2, 2, 1, 3, 1};
  vector<int8_t> data_vec_1 = {1, 2, 3, 4, 5, 20, 7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT8);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int8_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 96);  // 2*2*4*3*2
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerInt16Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT16);

  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<int16_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT16);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int16_t));

  vector<int64_t> dims_vec_1 = {2, 2, 1, 3, 1};
  vector<int16_t> data_vec_1 = {1, 2, 3, 4, 5, 20, 7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT16);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int16_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 192);  // 2*2*4*3*2*2
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerInt64Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT64);

  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<int64_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT64);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));

  vector<int64_t> dims_vec_1 = {2, 2, 1, 3, 1};
  vector<int64_t> data_vec_1 = {1, 2, 3, 4, 5, 20, 7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT64);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int64_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(outputs[0]->GetData().size(), 768);  // 2*2*4*3*2*8
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerUint8Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UINT8);

  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<uint8_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UINT8);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint8_t));

  vector<int64_t> dims_vec_1 = {2, 2, 1, 3, 1};
  vector<uint8_t> data_vec_1 = {1, 2, 3, 4, 5, 20, 7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UINT8);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(uint8_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 96);  // 2*2*4*3*2
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerUint16Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UINT16);

  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<uint16_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UINT16);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint16_t));

  vector<int64_t> dims_vec_1 = {2, 2, 1, 3, 1};
  vector<uint16_t> data_vec_1 = {1, 2, 3, 4, 5, 20, 7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UINT16);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(uint16_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 192);  // 2*2*4*3*2*2
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerUint32Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UINT32);

  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<uint32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UINT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint32_t));

  vector<int64_t> dims_vec_1 = {2, 2, 1, 3, 1};
  vector<uint32_t> data_vec_1 = {1, 2, 3, 4, 5, 20, 7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UINT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(uint32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 384);  // 2*2*4*3*2*4
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerUint64Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UINT64);

  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<uint64_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UINT64);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint64_t));

  vector<int64_t> dims_vec_1 = {2, 2, 1, 3, 1};
  vector<uint64_t> data_vec_1 = {1, 2, 3, 4, 5, 20, 7, 8, 9, 10, 11, 12};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UINT64);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(uint64_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 768);  // 2*2*4*3*2*8
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerFloat16Success) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_FLOAT16);

  vector<int64_t> dims_vec_0 = {4};
  vector<float> data_vec_0 = {1.0, 2.0, 3.0, 4.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT16);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1;
  vector<float> data_vec_1 = {1.0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT16);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 8);  // 4*2
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerFloatSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
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
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 16);  // 4*4
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerDoubleSuccess) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_DOUBLE);

  vector<int64_t> dims_vec_0 = {4};
  vector<double> data_vec_0 = {1.0, 2.0, 3.0, 4.0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_DOUBLE);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(double));

  vector<int64_t> dims_vec_1;
  vector<double> data_vec_1 = {1.0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_DOUBLE);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(double));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, SUCCESS);
  EXPECT_EQ(outputs[0]->GetData().size(), 32);  // 4*8
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerErrtypeFail) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
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

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, NOT_CHANGED);
}

TEST_F(UtestGraphPassesFoldingKernelMaximumKernel, MaximumOptimizerDifferentType) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("Maximum", MAXIMUM);
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1 = {4};
  vector<float> data_vec_1 = {1.0, 2.0, 3.0, 4.0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(MAXIMUM);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(status, NOT_CHANGED);
}
