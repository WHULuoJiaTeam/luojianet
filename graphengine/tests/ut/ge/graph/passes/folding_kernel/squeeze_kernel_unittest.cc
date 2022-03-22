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
#include "host_kernels/squeeze_kernel.h"

#include "../graph_builder_utils.h"
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

class UtestGraphPassesFoldingKernelSqueenzeKernel : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  template <typename inner_data_type, DataType data_type>
  void TestSqueeze(vector<int64_t> &data_vec, vector<int32_t> &dim_value_vec, vector<int64_t> &result) {
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

    // add squeeze node
    OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Squeeze", SQUEEZE);
    if (!dim_value_vec.empty()) {
      AttrUtils::SetListInt(expanddim_op_desc, SQUEEZE_ATTR_AXIS, dim_value_vec);
    }
    expanddim_op_desc->AddInputDesc(data_tensor_desc);
    NodePtr op_node = graph->AddNode(expanddim_op_desc);
    op_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));

    shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SQUEEZE);
    Status status = kernel->Compute(op_node);
    EXPECT_EQ(ge::SUCCESS, status);
  }

  template <typename inner_data_type, DataType data_type, Format format>
  void TestInvalidSqueeze(vector<int64_t> &data_vec, vector<int32_t> &dim_value_vec, vector<int64_t> &result) {
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

    // add squeeze node
    OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Squeeze", SQUEEZE);
    if (!dim_value_vec.empty()) {
      AttrUtils::SetListInt(expanddim_op_desc, SQUEEZE_ATTR_AXIS, dim_value_vec);
    }
    expanddim_op_desc->AddInputDesc(data_tensor_desc);
    NodePtr op_node = graph->AddNode(expanddim_op_desc);
    op_node->Init();

    // add edge
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));

    shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SQUEEZE);
    Status status = kernel->Compute(op_node);
    EXPECT_NE(ge::SUCCESS, status);
  }
};
namespace {

///     netoutput1
///        |
///      Squeeze
///        |
///      const1
ComputeGraphPtr BuildGraph() {
  auto builder = ut::GraphBuilder("test");
  auto const1 = builder.AddNode("const1", CONSTANT, 0, 1);
  auto squeeze = builder.AddNode("squeeze1", SQUEEZE, 1, 1);
  auto netoutput1 = builder.AddNode("netoutput1", NETOUTPUT, 1, 0);

  builder.AddDataEdge(const1, 0, squeeze, 0);
  builder.AddDataEdge(squeeze, 0, netoutput1, 0);

  return builder.GetGraph();
}
}  // namespace

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int8Case0) {
  vector<int64_t> data_vec = {1, 1, 1, 2, 3};
  vector<int32_t> dim_value_vec = {0};
  vector<int64_t> result = {1, 1, 2, 3};
  TestSqueeze<int8_t, DT_INT8>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int8Case1) {
  vector<int64_t> data_vec = {1, 1, 1, 2, 3};
  vector<int32_t> dim_value_vec = {0, 1};
  vector<int64_t> result = {1, 2, 3};
  TestSqueeze<int8_t, DT_INT8>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int8Case2) {
  vector<int64_t> data_vec = {1, 1, 1, 2, 3};
  vector<int32_t> dim_value_vec = {0, 1, 2};
  vector<int64_t> result = {2, 3};
  TestSqueeze<int8_t, DT_INT8>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int8NegativeCase1) {
  vector<int64_t> data_vec = {1, 1, 1, 2, 3};
  vector<int32_t> dim_value_vec = {-5};
  vector<int64_t> result = {1, 1, 2, 3};

  TestSqueeze<int8_t, DT_INT8>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int8NegativeCase2) {
  vector<int64_t> data_vec = {1, 1, 1, 2, 3};
  vector<int32_t> dim_value_vec = {-5, -4};
  vector<int64_t> result = {1, 2, 3};

  TestSqueeze<int8_t, DT_INT8>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int8NegativeCase3) {
  vector<int64_t> data_vec = {1, 1, 1, 2, 3};
  vector<int32_t> dim_value_vec = {-5, -4, -3};
  vector<int64_t> result = {2, 3};

  TestSqueeze<int8_t, DT_INT8>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int16) {
  vector<int64_t> data_vec = {1, 1, 2};
  vector<int32_t> dim_value_vec = {-3};
  vector<int64_t> result = {1, 2};
  TestSqueeze<int16_t, DT_INT16>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int32) {
  vector<int64_t> data_vec = {3, 3, 3, 1, 6};
  vector<int32_t> dim_value_vec = {3};
  vector<int64_t> result = {3, 3, 3, 6};
  TestSqueeze<int32_t, DT_INT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Int64) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {1};
  vector<int64_t> result = {6, 12, 3, 4, 56, 7};
  TestSqueeze<int64_t, DT_INT64>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Uint8) {
  vector<int64_t> data_vec = {2, 1};
  vector<int32_t> dim_value_vec = {1};
  vector<int64_t> result = {2};
  TestSqueeze<uint8_t, DT_UINT8>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Uint16) {
  vector<int64_t> data_vec = {1, 3};
  vector<int32_t> dim_value_vec = {0};
  vector<int64_t> result = {3};
  TestSqueeze<uint16_t, DT_UINT16>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Uint32) {
  vector<int64_t> data_vec = {3, 3, 3, 5, 1};
  vector<int32_t> dim_value_vec = {4};
  vector<int64_t> result = {3, 3, 3, 5};
  TestSqueeze<uint32_t, DT_UINT32>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Uint64) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {1};
  vector<int64_t> result = {6, 12, 3, 4, 56, 7};
  TestSqueeze<uint64_t, DT_UINT64>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Fp16) {
  vector<int64_t> data_vec = {6, 1, 12, 3, 4, 56, 7};
  vector<int32_t> dim_value_vec = {1};
  vector<int64_t> result = {6, 12, 3, 4, 56, 7};
  TestSqueeze<fp16_t, DT_FLOAT16>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Float) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {0, 1, 2, 4, 5, 6};
  vector<int64_t> result = {11};
  TestSqueeze<float, DT_FLOAT>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, Double) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {0, 1, 2, 4, 5, 6};
  vector<int64_t> result = {11};
  TestSqueeze<double, DT_DOUBLE>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, NodeIsNull) {
  NodePtr op_node = nullptr;
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SQUEEZE);
  Status status = kernel->Compute(op_node);
  EXPECT_NE(domi::PARAM_INVALID, status);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, BoolInt64) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {0, 1, 2, 4, 5, 6};
  vector<int64_t> result = {11};

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

  // add expanddims node
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Squeeze", SQUEEZE);
  if (!dim_value_vec.empty()) {
    AttrUtils::SetListInt(expanddim_op_desc, SQUEEZE_ATTR_AXIS, dim_value_vec);
  }
  expanddim_op_desc->AddInputDesc(data_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddim_op_desc);
  op_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SQUEEZE);
  Status status = kernel->Compute(op_node);
  EXPECT_EQ(ge::SUCCESS, status);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, DoubleNotAttr) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec;
  vector<int64_t> result = {11};
  TestSqueeze<double, DT_DOUBLE>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, DoubleContainSameDims) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {0, 1, 0};
  vector<int64_t> result = {1, 11, 1, 1, 1};
  TestSqueeze<double, DT_DOUBLE>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, DoubleContainSameDimsInvalidFormat) {
  vector<int64_t> data_vec = {1, 1, 1, 11, 1, 1, 1};
  vector<int32_t> dim_value_vec = {0, 1, 0};
  vector<int64_t> result = {1, 11, 1, 1, 1};
  TestInvalidSqueeze<double, DT_DOUBLE, FORMAT_NC1HWC0>(data_vec, dim_value_vec, result);
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, ConstFoldingSuccess) {
  auto graph = BuildGraph();
  std::vector<GeTensorPtr> v_output;
  std::vector<ConstGeTensorPtr> inputs;
  ConstGeTensorPtr data_tensor = std::make_shared<const GeTensor>();

  inputs.push_back(data_tensor);
  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SQUEEZE);
  Status status = kernel->Compute(graph->FindNode("squeeze1")->GetOpDesc(), inputs, v_output);
  EXPECT_EQ(ge::SUCCESS, status);
  EXPECT_EQ(1, v_output.size());
}

TEST_F(UtestGraphPassesFoldingKernelSqueenzeKernel, ConstFoldingUnsuccess) {
  auto graph = BuildGraph();
  std::vector<GeTensorPtr> v_output;
  std::vector<ConstGeTensorPtr> inputs;

  shared_ptr<Kernel> kernel = KernelFactory::Instance().Create(SQUEEZE);
  Status status = kernel->Compute(graph->FindNode("squeeze1")->GetOpDesc(), inputs, v_output);
  EXPECT_EQ(NOT_CHANGED, status);
  status = kernel->Compute(nullptr, inputs, v_output);
  EXPECT_EQ(PARAM_INVALID, status);

  std::vector<ConstGeTensorPtr> inputs_invalid;
  inputs_invalid.push_back(nullptr);
  status = kernel->Compute(graph->FindNode("squeeze1")->GetOpDesc(), inputs_invalid, v_output);
  EXPECT_EQ(PARAM_INVALID, status);
}
