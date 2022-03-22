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
#include "host_kernels/reshape_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

class UtestGraphPassesFoldingKernelReshapeKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type>
  void TestReshape(vector<int64_t> &data_vec, vector<inner_dim_type> &dim_value_vec, vector<int64_t> &result) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

    ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
    int64_t dims_size = 1;
    for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
    vector<inner_data_type> data_value_vec(dims_size, 1);
    GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, data_type);
    GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                         data_value_vec.size() * sizeof(inner_data_type));
    OpDescUtils::SetWeights(data_op_desc, data_tensor);
    data_op_desc->AddOutputDesc(data_tensor_desc);
    NodePtr data_node = graph->AddNode(data_op_desc);
    data_node->Init();

    // add dim node
    ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
    vector<int64_t> dim_vec;
    dim_vec.push_back(dim_value_vec.size());
    GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), FORMAT_NCHW, dim_type);
    GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                        dim_value_vec.size() * sizeof(inner_dim_type));
    OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
    dim_op_desc->AddOutputDesc(dim_tensor_desc);
    NodePtr dim_node = graph->AddNode(dim_op_desc);
    dim_node->Init();

    // add expanddims node
    OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
    expanddim_op_desc->AddInputDesc(data_tensor_desc);
    expanddim_op_desc->AddInputDesc(dim_tensor_desc);
    NodePtr op_node = graph->AddNode(expanddim_op_desc);
    op_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

    shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
    Status status = kernel->Compute(op_node);
    EXPECT_EQ(ge::SUCCESS, status);
  }

  template <typename inner_data_type, DataType data_type, typename inner_dim_type, DataType dim_type, Format format>
  void TestInvalidReshape(vector<int64_t> &data_vec, vector<inner_dim_type> &dim_value_vec, vector<int64_t> &result) {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

    ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
    int64_t dims_size = 1;
    for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
    vector<inner_data_type> data_value_vec(dims_size, 1);
    GeTensorDesc data_tensor_desc(GeShape(data_vec), format, data_type);
    GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                         data_value_vec.size() * sizeof(inner_data_type));
    OpDescUtils::SetWeights(data_op_desc, data_tensor);
    data_op_desc->AddOutputDesc(data_tensor_desc);
    NodePtr data_node = graph->AddNode(data_op_desc);
    data_node->Init();

    // add dim node
    ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
    vector<int64_t> dim_vec;
    dim_vec.push_back(dim_value_vec.size());
    GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), format, dim_type);
    GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                        dim_value_vec.size() * sizeof(inner_dim_type));
    OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
    dim_op_desc->AddOutputDesc(dim_tensor_desc);
    NodePtr dim_node = graph->AddNode(dim_op_desc);
    dim_node->Init();

    // add expanddims node
    OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
    expanddim_op_desc->AddInputDesc(data_tensor_desc);
    expanddim_op_desc->AddInputDesc(dim_tensor_desc);
    NodePtr op_node = graph->AddNode(expanddim_op_desc);
    op_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

    shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
    Status status = kernel->Compute(op_node);
    EXPECT_NE(ge::SUCCESS, status);

    vector<ConstGeTensorPtr> input = {data_tensor};
    vector<GeTensorPtr> outputs;
    status = kernel->Compute(op_node->GetOpDesc(), input, outputs);
    EXPECT_EQ(NOT_CHANGED, status);
  }
};

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Int8Int32) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {3, 2};
  vector<int64_t> result = {3, 2};
  TestReshape<int8_t, DT_INT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Int16Int32) {
  vector<int64_t> data_vec = {3, 3};
  vector<int32_t> dim_value_vec = {9};
  vector<int64_t> result = {9};
  TestReshape<int16_t, DT_INT16, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Int32Int32) {
  vector<int64_t> data_vec = {3, 3, 3, 5, 6};
  vector<int32_t> dim_value_vec = {9, 90};
  vector<int64_t> reuslt = {9, 90};
  TestReshape<int32_t, DT_INT32, int32_t, DT_INT32>(data_vec, dim_value_vec, reuslt);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Int64Int32) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {12, 6, 3 * 4 * 56 * 7};
  vector<int64_t> result = {12, 6, 3 * 4 * 56 * 7};
  TestReshape<int64_t, DT_INT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Uint8Int32) {
  vector<int64_t> data_vec = {2, 3};
  vector<int32_t> dim_value_vec = {-1};
  vector<int64_t> result = {6};
  TestReshape<uint8_t, DT_UINT8, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Uint16Int32) {
  vector<int64_t> data_vec = {3};
  vector<int32_t> dim_value_vec = {-1};
  vector<int64_t> result = {3};
  TestReshape<uint16_t, DT_UINT16, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Uint32Int32) {
  vector<int64_t> data_vec = {3, 3, 3, 5, 6};
  vector<int32_t> dim_value_vec = {3, -1};
  vector<int64_t> result = {3, 3 * 3 * 5 * 6};
  TestReshape<uint32_t, DT_UINT32, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Uint64Int32) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {6, 12, 3, 4, 7, -1};
  vector<int64_t> result = {6, 12, 3, 4, 7, 56};
  TestReshape<uint64_t, DT_UINT64, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, Fp16Int32) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {-1};
  vector<int64_t> result = {6 * 12 * 3 * 4 * 56 * 7 * 1};
  TestReshape<fp16_t, DT_FLOAT16, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, FloatInt32) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {-1};
  vector<int64_t> result = {11};
  TestReshape<float, DT_FLOAT, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, DoubleInt32) {
  vector<int64_t> data_vec = {7, 7, 7, 12, 2};
  vector<int32_t> dim_value_vec = {7, 12, 2, 7, 7};
  vector<int64_t> result = {7, 12, 2, 7, 7};
  TestReshape<double, DT_DOUBLE, int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, DoubleInt64) {
  vector<int64_t> data_vec = {3, 4, 2, 2, 8};
  vector<int64_t> dim_value_vec = {12, -1};
  vector<int64_t> result = {12, 32};
  TestReshape<double, DT_DOUBLE, int64_t, DT_INT64>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, BoolInt64) {
  vector<int64_t> data_vec = {3, 4, 2, 2, 8};
  vector<int64_t> dim_value_vec = {12, -1};
  vector<int64_t> result = {12, 32};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
  vector<int64_t> dim_vec;
  dim_vec.push_back(dim_value_vec.size());
  GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                      dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  expanddim_op_desc->AddInputDesc(dim_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  Status status = kernel->Compute(op_node);
  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, InvalidFormat) {
  vector<int64_t> data_vec = {2, 3};
  vector<int64_t> dim_value_vec = {-1};
  vector<int64_t> result = {0};

  TestInvalidReshape<int32_t, DT_INT32, int64_t, DT_INT64, FORMAT_FRACTAL_Z>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, NodeIsNull) {
  NodePtr op_node = nullptr;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  Status status = kernel->Compute(op_node);
  EXPECT_NE(domi::PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, InvalidInputNodeSize) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int64_t> dim_value_vec = {7};
  vector<int64_t> result = {1, 1, 1, 11, 1, 1, 1, 1};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                      dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  expanddim_op_desc->AddInputDesc(dim_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  Status status = kernel->Compute(op_node);
  EXPECT_NE(ge::SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, FoldingInt64Success) {
  vector<int64_t> data_vec = {3, 4, 2, 2, 8};
  vector<int64_t> dim_value_vec = {12, -1};
  vector<int64_t> result = {12, 32};

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = std::make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<uint8_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_BOOL);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(uint8_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = std::make_shared<ge::OpDesc>("dim", CONSTANTOP);
  vector<int64_t> dim_vec;
  dim_vec.push_back(dim_value_vec.size());
  GeTensorDesc dim_tensor_desc(ge::GeShape(dim_vec), FORMAT_NCHW, DT_INT64);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(),
                                                      dim_value_vec.size() * sizeof(int64_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  expanddim_op_desc->AddInputDesc(dim_tensor_desc);
  expanddim_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  vector<ConstGeTensorPtr> input = {data_tensor, dim_tensor};
  vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(op_node->GetOpDesc(), input, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelReshapeKernel, OpdescIsNullFailed) {
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(RESHAPE);
  ge::OpDescPtr null_op_desc = nullptr;
  vector<ConstGeTensorPtr> input = {};
  vector<GeTensorPtr> outputs;
  Status status = kernel->Compute(null_op_desc, input, outputs);
  EXPECT_EQ(PARAM_INVALID, status);
}
