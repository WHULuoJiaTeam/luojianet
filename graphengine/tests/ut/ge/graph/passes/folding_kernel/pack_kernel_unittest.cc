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
#include "host_kernels/pack_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/dimension_compute_pass.h"
#include "host_kernels/kernel_utils.h"
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

class UtestGraphPassesFoldingKernelPackKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

// optimize op of pack success
TEST_F(UtestGraphPassesFoldingKernelPackKernel, Test1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true, true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_3;
  vector<int32_t> data_vec_3 = {0};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelPackKernel, Test2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true, true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, (int64_t)4);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_3;
  vector<int32_t> data_vec_3 = {0};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelPackKernel, Test3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true, true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, (int64_t)4);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_3;
  vector<int32_t> data_vec_3 = {0};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelPackKernel, Test4) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true, true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, (int64_t)4);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_AXIS, (int64_t)0);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelPackKernel, Test5) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true, true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, (int64_t)4);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_AXIS, (int64_t)1);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_3;
  vector<int32_t> data_vec_3 = {0};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelPackKernel, PackOptimizerSuccessInt) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true, true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, (int64_t)4);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_AXIS, (int64_t)0);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_3;
  vector<int32_t> data_vec_3 = {0};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelPackKernel, PackOptimizerSuccessInt1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);

  AttrUtils::SetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, (int64_t)2);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_AXIS, (int64_t)0);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  GeTensorDesc dims_tensor_desc(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddInputDesc(0, dims_tensor_desc);

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
}

TEST_F(UtestGraphPassesFoldingKernelPackKernel, PackOptimizerSuccessFloat) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true, true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, (int64_t)4);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_AXIS, (int64_t)0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_FLOAT);

  vector<int64_t> dims_vec_0;
  vector<float> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));

  vector<int64_t> dims_vec_1;
  vector<float> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(float));

  vector<int64_t> dims_vec_2;
  vector<float> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(float));

  vector<int64_t> dims_vec_3;
  vector<float> data_vec_3 = {0};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(float));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelPackKernel, PackOptimizerFailedErrtype) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("pack", "Pack");
  vector<bool> is_input_const_vec = {true, true, true, true};
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, PACK_ATTR_NAME_NUM, (int64_t)4);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_AXIS, (int64_t)0);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_UNDEFINED);

  vector<int64_t> dims_vec_0;
  vector<int32_t> data_vec_0 = {0};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_1;
  vector<int32_t> data_vec_1 = {0};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_2;
  vector<int32_t> data_vec_2 = {0};
  GeTensorDesc tensor_desc_2(GeShape(dims_vec_2), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));

  vector<int64_t> dims_vec_3;
  vector<int32_t> data_vec_3 = {0};
  GeTensorDesc tensor_desc_3(GeShape(dims_vec_3), FORMAT_NCHW, DT_UNDEFINED);
  ConstGeTensorPtr tensor_3 =
      std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2, tensor_3};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(PACK);
  Status status = kernel->Compute(op_desc_ptr, input, outputs);
}
