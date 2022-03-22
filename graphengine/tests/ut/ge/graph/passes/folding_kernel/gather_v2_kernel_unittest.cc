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
#include <cmath>

#define protected public
#define private public
#include "host_kernels/gather_v2_kernel.h"

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

class UtestGraphPassesFoldingKernelGatherV2Kernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT32Axis0VersionA) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int32_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int32_t *data_buf = (int32_t *)tensor_out->GetData().data();
  vector<int32_t> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT32Axis0VersionB) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int32_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {2, 2};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int32_t *data_buf = (int32_t *)tensor_out->GetData().data();
  vector<int32_t> expect_out = {3, 3};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT64Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT64);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT64);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT64);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int64_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int64_t> data_vec_1 = {2, 2};
  vector<int64_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT64);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT64);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT64);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int64_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int64_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int64_t *data_buf = (int64_t *)tensor_out->GetData().data();
  vector<int64_t> expect_out = {3, 3};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}

TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT32Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {2, 3, 3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);
  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};  // 2*3*3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2, 3, 3});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int32_t *data_buf = (int32_t *)tensor_out->GetData().data();
  vector<int32_t> expect_out = {11, 12, 13, 14, 15, 16, 17, 18, 19, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT32Axis0And1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {2, 3, 3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);
  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};  // 2*3*3
  vector<int32_t> data_vec_1 = {1, 0};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2, 3, 3});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int32_t *data_buf = (int32_t *)tensor_out->GetData().data();
  vector<int32_t> expect_out = {11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}

TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT32Axis1) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {2, 3, 3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};  // 2*3*3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {1};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2, 2, 3});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int32_t *data_buf = (int32_t *)tensor_out->GetData().data();
  vector<int32_t> expect_out = {4, 5, 6, 4, 5, 6, 14, 15, 16, 14, 15, 16};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT32Axis2) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {2, 3, 3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};  // 2*3*3
  vector<int32_t> data_vec_1 = {0, 0};
  vector<int32_t> axis_vec = {2};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2, 3, 2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int32_t *data_buf = (int32_t *)tensor_out->GetData().data();
  vector<int32_t> expect_out = {1, 1, 4, 4, 7, 7, 11, 11, 14, 14, 17, 17};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT32Axis3) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {2, 2, 3, 3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};  // 2*2*3*3
  vector<int32_t> data_vec_1 = {0, 1};
  vector<int32_t> axis_vec = {3};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2, 2, 3, 2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int32_t *data_buf = (int32_t *)tensor_out->GetData().data();
  vector<int32_t> expect_out = {1, 2, 4, 5, 7, 8, 11, 12, 14, 15, 17, 18, 1, 2, 4, 5, 7, 8, 11, 12, 14, 15, 17, 18};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}

TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT8Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT8);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int8_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT8);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int8_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int8_t *data_buf = (int8_t *)tensor_out->GetData().data();
  vector<int8_t> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, INT16Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT16);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int16_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT16);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int16_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  int16_t *data_buf = (int16_t *)tensor_out->GetData().data();
  vector<int16_t> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, UINT8Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_UINT8);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<uint8_t> data_vec_0 = {1, 2, 3};
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_UINT8);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint8_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  uint8_t *data_buf = (uint8_t *)tensor_out->GetData().data();
  vector<uint8_t> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, UINT16Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_UINT16);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<uint16_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_UINT16);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint16_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  uint16_t *data_buf = (uint16_t *)tensor_out->GetData().data();
  vector<uint16_t> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, UINT32Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_UINT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<uint32_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_UINT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  uint32_t *data_buf = (uint32_t *)tensor_out->GetData().data();
  vector<uint32_t> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, UINT64Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_UINT64);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<uint64_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_UINT64);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(uint64_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  uint64_t *data_buf = (uint64_t *)tensor_out->GetData().data();
  vector<uint64_t> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    EXPECT_EQ(*(data_buf + i), expect_out[i]);
  }
}

TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, DoubleAxis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_DOUBLE);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<double> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_DOUBLE);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(double));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  double *data_buf = (double *)tensor_out->GetData().data();
  vector<double> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    double diff = *(data_buf + i) - expect_out[i];
    bool is_same = fabs(diff) < 0.0001 ? true : false;
    EXPECT_EQ(is_same, true);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, Float16Axis0) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_FLOAT16);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<fp16_t> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_FLOAT16);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * 2);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  /// check result
  /// 1. check shape
  EXPECT_EQ(outputs.size(), 1);
  vector<int64_t> expect_output_shape({2});
  auto real_output_shape = outputs[0]->GetTensorDesc().GetShape().GetDims();
  bool is_same_shape = (real_output_shape == expect_output_shape);
  EXPECT_EQ(is_same_shape, true);
  // 2. check result
  GeTensorPtr tensor_out = outputs[0];
  fp16_t *data_buf = (fp16_t *)tensor_out->GetData().data();
  vector<fp16_t> expect_out = {2, 2};
  for (size_t i = 0; i < expect_out.size(); i++) {
    double diff = (double)*(data_buf + i) - (double)expect_out[i];
    bool is_same = fabs(diff) < 0.0001 ? true : false;
    EXPECT_EQ(is_same, true);
  }
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, AbnormalTestDatatypeNotSupport) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_FLOAT);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<float> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_FLOAT);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(float));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_NE(ge::SUCCESS, status);
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, AbnormalTestIndicesOverOne) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2, 2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<float> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1, 1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_NE(ge::SUCCESS, status);
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, AbnormalTestAxisOverThree) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {2, 2, 3, 3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19};  // 2*2*3*3
  vector<int32_t> data_vec_1 = {0, 1};
  vector<int32_t> axis_vec = {4};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
  ge::Status status = kernel->Compute(op_desc_ptr, input, outputs);
  EXPECT_NE(ge::SUCCESS, status);
}
TEST_F(UtestGraphPassesFoldingKernelGatherV2Kernel, AbnormalTest) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("GatherV2", "GatherV2");

  vector<int64_t> x_shape = {3};
  vector<int64_t> indices_shape = {2};
  GeTensorDesc tensor_desc_x(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_indices(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_axis(GeShape(), FORMAT_NHWC, DT_INT32);

  op_desc_ptr->AddInputDesc(0, tensor_desc_x);
  op_desc_ptr->AddInputDesc(1, tensor_desc_indices);
  op_desc_ptr->AddInputDesc(2, tensor_desc_axis);

  vector<float> data_vec_0 = {1, 2, 3};  // 3
  vector<int32_t> data_vec_1 = {1, 1, 1, 1};
  vector<int32_t> axis_vec = {0};
  GeTensorDesc tensor_desc_0(GeShape(x_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_1(GeShape(indices_shape), FORMAT_NHWC, DT_INT32);
  GeTensorDesc tensor_desc_2(GeShape(), FORMAT_NHWC, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int32_t));
  ConstGeTensorPtr tensor_2 =
      std::make_shared<GeTensor>(tensor_desc_2, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1, tensor_2};
  vector<GeTensorPtr> outputs;

  {
    shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(GATHERV2);
    ge::Status status = kernel->Compute(nullptr, input, outputs);
    EXPECT_NE(ge::SUCCESS, status);
    vector<ConstGeTensorPtr> input_1 = {tensor_0, tensor_1, tensor_2, tensor_2};
    status = kernel->Compute(op_desc_ptr, input_1, outputs);
    EXPECT_NE(ge::SUCCESS, status);
    vector<ConstGeTensorPtr> input_2 = {nullptr, nullptr, nullptr};
    status = kernel->Compute(op_desc_ptr, input_2, outputs);
    EXPECT_NE(ge::SUCCESS, status);
    ConstGeTensorPtr tensor_11 = std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), 0);
    vector<ConstGeTensorPtr> input_11 = {tensor_0, tensor_11, tensor_2};
    status = kernel->Compute(op_desc_ptr, input_11, outputs);
    EXPECT_NE(ge::SUCCESS, status);

    GeTensorDesc tensor_desc_3(GeShape({1, 2, 3, 4}), FORMAT_NHWC, DT_INT32);
    ConstGeTensorPtr tensor_3 =
        std::make_shared<GeTensor>(tensor_desc_3, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));
    vector<ConstGeTensorPtr> input_3 = {tensor_0, tensor_1, tensor_2, tensor_2};
    status = kernel->Compute(op_desc_ptr, input_3, outputs);
    EXPECT_NE(ge::SUCCESS, status);
    GeTensorDesc tensor_desc_indices_1(GeShape(indices_shape), FORMAT_NHWC, DT_UINT32);
    ConstGeTensorPtr tensor_22 = std::make_shared<GeTensor>(tensor_desc_indices_1, (uint8_t *)data_vec_1.data(),
                                                            data_vec_1.size() * sizeof(int32_t));
    vector<ConstGeTensorPtr> input_4 = {tensor_0, tensor_22, tensor_2};
    status = kernel->Compute(op_desc_ptr, input_4, outputs);
    EXPECT_NE(ge::SUCCESS, status);
    GeTensorDesc tensor_desc_axis_1(GeShape(indices_shape), FORMAT_NHWC, DT_UINT32);
    ConstGeTensorPtr tensor_33 =
        std::make_shared<GeTensor>(tensor_desc_axis_1, (uint8_t *)axis_vec.data(), axis_vec.size() * sizeof(int32_t));
    vector<ConstGeTensorPtr> input_5 = {tensor_0, tensor_33, tensor_2};
    status = kernel->Compute(op_desc_ptr, input_5, outputs);
    EXPECT_NE(ge::SUCCESS, status);
    vector<int32_t> data_vec_2 = {5, 1, 1, 1};
    ConstGeTensorPtr tensor_44 =
        std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_2.data(), data_vec_2.size() * sizeof(int32_t));
    vector<ConstGeTensorPtr> input_6 = {tensor_0, tensor_44, tensor_2};
    status = kernel->Compute(op_desc_ptr, input_6, outputs);
    EXPECT_NE(ge::SUCCESS, status);
    vector<int64_t> data_vec_3 = {5, 1, 1, 1};
    GeTensorDesc tensor_desc_55(GeShape(indices_shape), FORMAT_NHWC, DT_INT64);
    ConstGeTensorPtr tensor_55 =
        std::make_shared<GeTensor>(tensor_desc_55, (uint8_t *)data_vec_3.data(), data_vec_3.size() * sizeof(int64_t));
    vector<ConstGeTensorPtr> input_7 = {tensor_0, tensor_55, tensor_2};
    status = kernel->Compute(op_desc_ptr, input_7, outputs);
    EXPECT_NE(ge::SUCCESS, status);
  }
}
