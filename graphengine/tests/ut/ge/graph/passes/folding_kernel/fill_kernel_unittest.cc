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
#include "host_kernels/fill_kernel.h"

#include "common/debug/log.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;
using ge::SUCCESS;

class UtestGraphPassesFoldingKernelFillKernel : public testing::Test {
 protected:
  void SetUp() {
    graph = std::make_shared<ge::ComputeGraph>("default");
    op_desc_ptr = std::make_shared<OpDesc>("Fill", FILL);
    node = std::make_shared<Node>(op_desc_ptr, graph);
    kernel = KernelFactory::Instance().Create(FILL);
  }

  void TearDown() {}

  template <typename T, typename DimType>
  void TestShape2And3(DataType type, DataType dim_type = DT_INT32) {
    ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
    vector<int64_t> dims_vec = {2};
    vector<DimType> dims_value_vec = {2, 3};
    GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, dim_type);
    GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                        dims_value_vec.size() * sizeof(DimType));
    OpDescUtils::SetWeights(op_dims, dim_tensor);

    ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
    vector<T> data_vec = {1};
    GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, type);
    GeTensorPtr value_tensor =
        std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(T));
    OpDescUtils::SetWeights(op_value, value_tensor);

    op_desc_ptr->AddInputDesc(dims_tensor_desc);
    op_desc_ptr->AddInputDesc(value_tensor_desc);
    op_desc_ptr->AddOutputDesc(dims_tensor_desc);

    std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
    std::vector<GeTensorPtr> outputs;
    Status status = kernel->Compute(op_desc_ptr, input, outputs);

    EXPECT_EQ(SUCCESS, status);
    EXPECT_EQ(outputs[0]->GetTensorDesc().GetDataType(), type);
    EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 2);
    for (int i = 0; i < 2 * 3; i++) {
      uint8_t *ptr = (uint8_t *)outputs[0]->GetData().data();
      EXPECT_TRUE(memcmp(ptr + i * sizeof(T), (void *)&data_vec[0], sizeof(T)) == 0);
    }
  }

  ge::ComputeGraphPtr graph;
  OpDescPtr op_desc_ptr;
  NodePtr node;
  shared_ptr<Kernel> kernel;
};

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillFloatShape2And3) { TestShape2And3<float, int32_t>(DT_FLOAT); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, fill_int8_shape_2_3) { TestShape2And3<int8_t, int32_t>(DT_INT8); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillInt16Shape2And3) { TestShape2And3<int16_t, int32_t>(DT_INT16); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillInt32Shape2And3) { TestShape2And3<int32_t, int32_t>(DT_INT32); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillInt64Shape2And3) { TestShape2And3<int64_t, int32_t>(DT_INT64); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillUint8Shape2And3) { TestShape2And3<uint8_t, int32_t>(DT_UINT8); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillUint16Shape2And3) { TestShape2And3<uint16_t, int32_t>(DT_UINT16); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillUint32Shape2And3) { TestShape2And3<uint32_t, int32_t>(DT_UINT32); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillUint64Shape2And3) { TestShape2And3<uint64_t, int32_t>(DT_UINT64); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillDoubleShape2And3) { TestShape2And3<double, int32_t>(DT_DOUBLE); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillFloat16Shape2And3) { TestShape2And3<fp16_t, int32_t>(DT_FLOAT16); }

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillBoolShape2And3) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int32_t> dims_value_vec = {2, 3};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  op_desc_ptr->AddInputDesc(dims_tensor_desc);
  op_desc_ptr->AddInputDesc(value_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(SUCCESS, status);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetDataType(), DT_BOOL);
  EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 2);
  for (int i = 0; i < 2 * 3; i++) {
    uint8_t *ptr = (uint8_t *)outputs[0]->GetData().data();
    EXPECT_TRUE(memcmp(ptr + i * sizeof(bool), (void *)&data_vec[0], sizeof(bool)) == 0);
  }
}

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillDimTypeInt64FloatShape2And3) {
  TestShape2And3<float, int64_t>(DT_FLOAT, DT_INT64);
}

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillInputNumNotEqual2) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int32_t> dims_value_vec = {2, 3};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor, dim_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillInputValueNotScalar) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int32_t> dims_value_vec = {2, 3};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<int64_t> data_dims_vec = {2};
  vector<uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(data_dims_vec), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillInputDimNotInt32Int64) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int16_t> dims_value_vec = {2, 3};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT16);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int16_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillDimsHaveNegativeNumber) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int32_t> dims_value_vec = {-2, 3};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  op_desc_ptr->AddInputDesc(dims_tensor_desc);
  op_desc_ptr->AddInputDesc(value_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(PARAM_INVALID, status);
}

/*
TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillDataTypeNotSupport) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int32_t> dims_value_vec = {2, 3};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_DUAL);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  op_desc_ptr->AddInputDesc(dims_tensor_desc);
  op_desc_ptr->AddInputDesc(value_tensor_desc);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(PARAM_INVALID, status);
}
*/

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillDimsTypeNotSupport) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int8_t> dims_value_vec = {2, 3};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT8);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int8_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_DUAL);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  op_desc_ptr->AddInputDesc(dims_tensor_desc);
  op_desc_ptr->AddInputDesc(value_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillDimsOverflow) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int64_t> dims_value_vec = {9223372036854775807, 2};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  op_desc_ptr->AddInputDesc(dims_tensor_desc);
  op_desc_ptr->AddInputDesc(value_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelFillKernel, FillDimsMulDataTypeOverflow) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector<int64_t> dims_vec = {2};
  vector<int64_t> dims_value_vec = {9223372036854775807, 1};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *)dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector<int32_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_INT32);
  GeTensorPtr value_tensor =
      std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *)data_vec.data(), data_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(op_value, value_tensor);

  op_desc_ptr->AddInputDesc(dims_tensor_desc);
  op_desc_ptr->AddInputDesc(value_tensor_desc);
  op_desc_ptr->AddOutputDesc(dims_tensor_desc);

  std::vector<ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelFillKernel, OutputdescUnknown) {
  ge::OpDescPtr op_dims = std::make_shared<ge::OpDesc>();
  vector <int64_t> dims_vec = {2};
  vector <int32_t> dims_value_vec = {2, 3};
  GeTensorDesc dims_tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dims_tensor_desc, (uint8_t *) dims_value_vec.data(),
                                                      dims_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(op_dims, dim_tensor);

  ge::OpDescPtr op_value = std::make_shared<ge::OpDesc>();
  vector <uint8_t> data_vec = {1};
  GeTensorDesc value_tensor_desc(GeShape(), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr value_tensor =
          std::make_shared<GeTensor>(value_tensor_desc, (uint8_t *) data_vec.data(), data_vec.size() * sizeof(bool));
  OpDescUtils::SetWeights(op_value, value_tensor);

  op_desc_ptr->AddInputDesc(dims_tensor_desc);
  op_desc_ptr->AddInputDesc(value_tensor_desc);

  vector <int64_t> out_vec = {-1, -1};
  GeTensorDesc out_tensor_desc(GeShape(out_vec), FORMAT_NCHW, DT_INT32);
  op_desc_ptr->AddOutputDesc(out_tensor_desc);

  std::vector <ge::ConstGeTensorPtr> input = {dim_tensor, value_tensor};
  std::vector <GeTensorPtr> outputs;
  Status status = kernel->Compute(op_desc_ptr, input, outputs);

  EXPECT_EQ(NOT_CHANGED, status);
}