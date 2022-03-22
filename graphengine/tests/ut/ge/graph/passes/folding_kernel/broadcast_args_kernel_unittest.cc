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

#include <assert.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#define protected public
#define private public
#include "host_kernels/broadcast_args_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/op/attr_value_util.h"
#include "common/types.h"
#include "folding_kernel_unittest_utils.h"
#include "framework/common/ge_inner_error_codes.h"
#include "ge/ge_api.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator.h"
#include "graph/passes/constant_folding_pass.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace ge::test;

#define TEST_OPERATOR(op_, input_shapes, output_shapes)                                                 \
  {                                                                                                     \
    auto op = op_;                                                                                      \
    for (auto input_pair : input_shapes) {                                                              \
      SetInputShape(op, input_pair.first, input_pair.second);                                           \
      SetInputDataType(op, input_pair.first);                                                           \
    }                                                                                                   \
    op.InferShapeAndType();                                                                             \
    for (auto output_pair : output_shapes) CheckOutputShape(op, output_pair.first, output_pair.second); \
  }
#define LOOP_VEC(v) for (size_t i = 0; i < v.size(); i++)

class UtestBroadCastArgsKernel : public testing::Test {
 protected:
  void SetUp() { init(); }

  void TearDown() { destory(); }

 private:
  void init() {
    pass_ = new ConstantFoldingPass();
    graph_ = std::make_shared<ge::ComputeGraph>("default");
    op_desc_ptr_ = std::make_shared<OpDesc>("BroadcastArgs", BROADCASTARGS);
    node_ = std::make_shared<Node>(op_desc_ptr_, graph_);
  }
  void destory() {
    delete pass_;
    pass_ = NULL;
  }

 protected:
  void SetInputShape(Operator op, string name, vector<int64_t> shape) {
    TensorDesc tensor_desc = op.GetInputDesc(name);
    tensor_desc.SetShape(ge::Shape(shape));
    op.UpdateInputDesc(name, tensor_desc);
  }

  void SetInputDataType(Operator op, string name) {
    TensorDesc tensor_desc = op.GetInputDesc(name);
    tensor_desc.SetDataType(DT_INT32);
    op.UpdateInputDesc(name, tensor_desc);
  }

  void CheckOutputShape(Operator op, string name, vector<int64_t> shape) {
    ge::Shape s = op.GetOutputDesc(name).GetShape();
    EXPECT_EQ(s.GetDims().size(), shape.size());
    LOOP_VEC(shape) EXPECT_EQ(s.GetDim(i), shape[i]);
  }

  void InitNodeSuccessSame(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("broadcast_args", BROADCASTARGS);
    op_desc->AddOutputDesc(ge::GeTensorDesc());

    vector<bool> is_input_const = {true, true};
    op_desc->SetIsInputConst(is_input_const);

    vector<int64_t> dims_vec_0 = {1, 2, 3, 4};
    vector<int64_t> data_vec_0 = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    ge::GeTensorDesc tensor_desc_0(ge::GeShape(dims_vec_0), ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeTensorPtr tensor_0 = std::make_shared<ge::GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(),
                                                              data_vec_0.size() * sizeof(int32_t));

    vector<int64_t> dims_vec_1 = {1, 2, 3, 4};
    vector<int64_t> data_vec_1 = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    ge::GeTensorDesc tensor_desc_1(ge::GeShape(dims_vec_1), ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeTensorPtr tensor_1 = std::make_shared<ge::GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(),
                                                              data_vec_1.size() * sizeof(int32_t));

    vector<ge::GeTensorPtr> weights = {tensor_0, tensor_1};

    ge::NodePtr node = graph->AddNode(op_desc);

    ge::OpDescUtils::SetWeights(node, weights);

    ge::OpDescPtr op_desc1 = std::make_shared<ge::OpDesc>();
    op_desc1->AddInputDesc(ge::GeTensorDesc());
    vector<bool> is_input_const1 = {false};
    op_desc1->SetIsInputConst(is_input_const1);
    ge::NodePtr node1 = graph->AddNode(op_desc1);

    ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  }

  void InitNodeSuccessNotSame(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("broadcast_args", BROADCASTARGS);
    op_desc->AddOutputDesc(ge::GeTensorDesc());

    vector<bool> is_input_const = {true, true};
    op_desc->SetIsInputConst(is_input_const);

    vector<int64_t> dims_vec_0 = {1, 2, 3, 4};
    vector<int64_t> data_vec_0 = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    ge::GeTensorDesc tensor_desc_0(ge::GeShape(dims_vec_0), ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeTensorPtr tensor_0 = std::make_shared<ge::GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(),
                                                              data_vec_0.size() * sizeof(int32_t));

    vector<int64_t> dims_vec_1 = {4};
    vector<int64_t> data_vec_1 = {0, 9, 5, 6};
    ge::GeTensorDesc tensor_desc_1(ge::GeShape(dims_vec_1), ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeTensorPtr tensor_1 = std::make_shared<ge::GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(),
                                                              data_vec_1.size() * sizeof(int32_t));

    vector<ge::GeTensorPtr> weights = {tensor_0, tensor_1};

    ge::NodePtr node = graph->AddNode(op_desc);

    ge::OpDescUtils::SetWeights(node, weights);

    ge::OpDescPtr op_desc1 = std::make_shared<ge::OpDesc>();
    op_desc1->AddInputDesc(ge::GeTensorDesc());
    vector<bool> is_input_const1 = {false};
    op_desc1->SetIsInputConst(is_input_const1);
    ge::NodePtr node1 = graph->AddNode(op_desc1);

    ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  }

  void InitNodeFailed(ge::ComputeGraphPtr graph) {
    ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("broadcast_gradient_args", BROADCASTGRADIENTARGS);
    op_desc->AddOutputDesc(ge::GeTensorDesc());

    vector<bool> is_input_const = {true, true};
    op_desc->SetIsInputConst(is_input_const);

    vector<int64_t> dims_vec_0 = {1, 2, 3, 4};
    vector<int64_t> data_vec_0 = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
    ge::GeTensorDesc tensor_desc_0(ge::GeShape(dims_vec_0), ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeTensorPtr tensor_0 = std::make_shared<ge::GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(),
                                                              data_vec_0.size() * sizeof(int32_t));

    vector<int64_t> dims_vec_1 = {1, 2};
    vector<int64_t> data_vec_1 = {0, 9};
    ge::GeTensorDesc tensor_desc_1(ge::GeShape(dims_vec_1), ge::FORMAT_NCHW, ge::DT_INT32);
    ge::GeTensorPtr tensor_1 = std::make_shared<ge::GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(),
                                                              data_vec_1.size() * sizeof(int32_t));

    vector<ge::GeTensorPtr> weights = {tensor_0, tensor_1};

    ge::NodePtr node = graph->AddNode(op_desc);

    ge::OpDescUtils::SetWeights(node, weights);

    ge::OpDescPtr op_desc1 = std::make_shared<ge::OpDesc>();
    op_desc1->AddInputDesc(ge::GeTensorDesc());
    vector<bool> is_input_const1 = {false};
    op_desc1->SetIsInputConst(is_input_const1);
    ge::NodePtr node1 = graph->AddNode(op_desc1);

    ge::GraphUtils::AddEdge(node->GetOutDataAnchor(0), node1->GetInDataAnchor(0));
  }
  ConstantFoldingPass *pass_;

  ge::ComputeGraphPtr graph_;
  OpDescPtr op_desc_ptr_;
  NodePtr node_;
};

TEST_F(UtestBroadCastArgsKernel, BroadCastArgsSuccessSame) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  InitNodeSuccessSame(graph);
  for (auto node : graph->GetAllNodes()) {
    if (node->GetOpDesc()->GetType() == BROADCASTARGS) {
      Status ret = pass_->Run(node);
      EXPECT_EQ(ge::PARAM_INVALID, ret);
    }
  }
}

TEST_F(UtestBroadCastArgsKernel, BroadCastArgsSuccessNotSame) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  InitNodeSuccessSame(graph);
  for (auto node : graph->GetAllNodes()) {
    if (node->GetOpDesc()->GetType() == BROADCASTARGS) {
      Status ret = pass_->Run(node);
      EXPECT_EQ(ge::PARAM_INVALID, ret);
    }
  }
}

TEST_F(UtestBroadCastArgsKernel, BroadCastArgsFailed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  InitNodeFailed(graph);
  for (auto n : graph->GetAllNodes()) {
    if (n->GetOpDesc()->GetType() == BROADCASTARGS) {
      Status ret = pass_->Run(n);
      EXPECT_EQ(ge::PARAM_INVALID, ret);
    }
  }
}

TEST_F(UtestBroadCastArgsKernel, SizeCheckFail) {
  vector<int64_t> dims_vec_0 = {8, 2};
  vector<int64_t> data_vec_0 = {2, 1, 4, 1, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));
  op_desc_ptr_->AddInputDesc(tensor_desc_0);

  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT64);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out);

  vector<ConstGeTensorPtr> input = {tensor_0};

  std::vector<GeTensorPtr> v_output;
  auto kernel_ptr = KernelFactory::Instance().Create(BROADCASTARGS);
  if (kernel_ptr != nullptr) {
    Status status = kernel_ptr->Compute(op_desc_ptr_, input, v_output);
    EXPECT_EQ(NOT_CHANGED, status);
  }
}

TEST_F(UtestBroadCastArgsKernel, UnknowShapeFail) {
  vector<int64_t> dims_vec_0 = {-1};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr_->AddInputDesc(tensor_desc_0);

  vector<int64_t> dims_vec_1 = {-1};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr_->AddInputDesc(tensor_desc_1);

  GeTensorDesc tensor_desc_out_1(GeShape(), FORMAT_NCHW, DT_INT64);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out_1);

  vector<int64_t> data_vec_0 = {2, 1, 4, 1, 2};
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));

  vector<int64_t> data_vec_1 = {2, 2, 1, 3, 1};
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int64_t));

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};

  std::vector<GeTensorPtr> v_output;

  auto kernel_ptr = KernelFactory::Instance().Create(BROADCASTARGS);
  if (kernel_ptr != nullptr) {
    Status status = kernel_ptr->Compute(op_desc_ptr_, input, v_output);
    EXPECT_EQ(ge::PARAM_INVALID, status);
  }
}

TEST_F(UtestBroadCastArgsKernel, CheckOutputNormal) {
  string op_type = BROADCASTARGS;
  vector<vector<int64_t>> i_shape_dims({
      {2},
      {5},
  });
  vector<vector<int64_t>> i_data({
      {1, 1},
      {2, 2, 1, 3, 1},
  });

  vector<vector<int64_t>> o_shape_dims({
      {5},
  });
  vector<vector<int64_t>> o_data({
      {2, 2, 1, 3, 1},
  });

  bool result = ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, i_shape_dims, i_data, DT_INT64, o_shape_dims,
                                                                o_data, DT_INT64);
  EXPECT_EQ(result, true);
}

TEST_F(UtestBroadCastArgsKernel, CheckOutputNormalInt32) {
  string op_type = BROADCASTARGS;
  vector<vector<int64_t>> i_shape_dims({
      {2},
      {5},
  });
  vector<vector<int32_t>> i_data({
      {1, 1},
      {2, 2, 1, 3, 1},
  });

  vector<vector<int64_t>> o_shape_dims({
      {5},
  });
  vector<vector<int32_t>> o_data({
      {2, 2, 1, 3, 1},
  });

  bool result = ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, i_shape_dims, i_data, DT_INT32, o_shape_dims,
                                                                o_data, DT_INT32);
  EXPECT_EQ(result, true);
}

TEST_F(UtestBroadCastArgsKernel, CheckOutputInputsSame) {
  string op_type = BROADCASTARGS;
  vector<vector<int64_t>> i_shape_dims({
      {5},
      {5},
  });
  vector<vector<int64_t>> i_data({
      {2, 2, 1, 3, 1},
      {2, 2, 1, 3, 1},
  });

  vector<vector<int64_t>> o_shape_dims({
      {5},
  });
  vector<vector<int64_t>> o_data({
      {2, 2, 1, 3, 1},
  });

  bool result = ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, i_shape_dims, i_data, DT_INT64, o_shape_dims,
                                                                o_data, DT_INT64);
  EXPECT_EQ(result, true);
}

TEST_F(UtestBroadCastArgsKernel, CheckOutputInputsOneScalar) {
  string op_type = BROADCASTARGS;
  vector<vector<int64_t>> i_shape_dims({
      {1},
      {3},
  });
  vector<vector<int64_t>> i_data({
      {5},
      {2, 3, 5},
  });

  vector<vector<int64_t>> o_shape_dims({
      {3},
  });
  vector<vector<int64_t>> o_data({
      {2, 3, 5},
  });

  bool result = ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, i_shape_dims, i_data, DT_INT64, o_shape_dims,
                                                                o_data, DT_INT64);
  EXPECT_EQ(result, true);
}
TEST_F(UtestBroadCastArgsKernel, CheckOutputInputsBothScalar) {
  string op_type = BROADCASTARGS;
  vector<vector<int64_t>> i_shape_dims({
      {1},
      {1},
  });
  vector<vector<int64_t>> i_data({
      {4},
      {1},
  });

  vector<vector<int64_t>> o_shape_dims({
      {1},
  });
  vector<vector<int64_t>> o_data({
      {4},
  });

  bool result = ge::test::ConstFoldingKernelCheckShapeAndOutput(op_type, i_shape_dims, i_data, DT_INT64, o_shape_dims,
                                                                o_data, DT_INT64);
  EXPECT_EQ(result, true);
}

TEST_F(UtestBroadCastArgsKernel, GetShapeDataFromConstTensorFail) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("BroadcastArgs", BROADCASTARGS);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0 = {};
  vector<int64_t> data_vec_0 = {};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));
  op_desc_ptr->AddInputDesc(tensor_desc_0);

  vector<int64_t> dims_vec_1 = {5};
  vector<int64_t> data_vec_1 = {0, 1, 4, 1, 2};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int64_t));
  op_desc_ptr->AddInputDesc(tensor_desc_1);
  op_desc_ptr->AddOutputDesc(GeTensorDesc());

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  std::shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(BROADCASTARGS);

  auto kernel_ptr = KernelFactory::Instance().Create(BROADCASTARGS);
  if (kernel_ptr != nullptr) {
    Status status = kernel_ptr->Compute(op_desc_ptr, input, outputs);
    EXPECT_EQ(ge::PARAM_INVALID, status);
  }
}

TEST_F(UtestBroadCastArgsKernel, GenerateBcastInfoFailed) {
  OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("BroadcastArgs", BROADCASTARGS);
  vector<bool> is_input_const_vec = {
      true,
      true,
  };
  op_desc_ptr->SetIsInputConst(is_input_const_vec);
  AttrUtils::SetInt(op_desc_ptr, ATTR_NAME_T, (int64_t)DT_INT32);

  vector<int64_t> dims_vec_0 = {5};
  vector<int64_t> data_vec_0 = {-1, 0, 4, 1, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_0 =
      std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)data_vec_0.data(), data_vec_0.size() * sizeof(int64_t));
  op_desc_ptr->AddInputDesc(tensor_desc_0);

  vector<int64_t> dims_vec_1 = {5};
  vector<int64_t> data_vec_1 = {1, 1, 4, 1, 2};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_INT32);
  ConstGeTensorPtr tensor_1 =
      std::make_shared<GeTensor>(tensor_desc_1, (uint8_t *)data_vec_1.data(), data_vec_1.size() * sizeof(int64_t));
  op_desc_ptr->AddInputDesc(tensor_desc_1);
  op_desc_ptr->AddOutputDesc(GeTensorDesc());

  vector<ConstGeTensorPtr> input = {tensor_0, tensor_1};
  vector<GeTensorPtr> outputs;

  std::shared_ptr<ge::Kernel> kernel = ge::KernelFactory::Instance().Create(BROADCASTARGS);
  if (kernel != nullptr) {
    Status status = kernel->Compute(op_desc_ptr, input, outputs);
    EXPECT_EQ(ge::PARAM_INVALID, status);
  }
}
