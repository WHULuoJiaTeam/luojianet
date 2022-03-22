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
#include "host_kernels/shape_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/fp16_t.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/passes/dimension_compute_pass.h"
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

class UtestGraphPassesFoldingKernelShapeKernel : public testing::Test {
 protected:
  void SetUp() { init(); }

  void TearDown() { destory(); }

 private:
  void init() {
    pass_ = new ge::DimensionComputePass();
    graph_ = std::make_shared<ge::ComputeGraph>("default");
    op_desc_ptr_ = std::make_shared<OpDesc>("Shape", SHAPE);
    node_ = std::make_shared<Node>(op_desc_ptr_, graph_);
    kernel_ = KernelFactory::Instance().Create(SHAPE);
  }

  void destory() {
    delete pass_;
    pass_ = NULL;
  }

 protected:
  ge::DimensionComputePass *pass_;
  ge::ComputeGraphPtr graph_;
  OpDescPtr op_desc_ptr_;
  NodePtr node_;
  shared_ptr<Kernel> kernel_;

  NodePtr init_node(ComputeGraphPtr graph) {
    // middle
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", SHAPE);
    OpDescPtr in_op_def = std::make_shared<OpDesc>("op_def_in", "test");
    OpDescPtr out_op_def = std::make_shared<OpDesc>("op_def_in", "test");
    // input tensor
    vector<int64_t> dims = {11, 16, 10, 12};
    ge::GeShape shape_desc(dims);
    GeTensorDesc tensor_desc(shape_desc);
    (void)TensorUtils::SetRealDimCnt(tensor_desc, dims.size());
    op_def->AddInputDesc(tensor_desc);

    GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
    op_def->AddOutputDesc(tensor_desc_out);
    // first
    in_op_def->AddOutputDesc(tensor_desc);

    // add attr of out_node
    vector<bool> is_input_const(3, false);
    is_input_const[0] = true;
    out_op_def->SetIsInputConst(is_input_const);
    out_op_def->AddInputDesc(tensor_desc);
    out_op_def->AddInputDesc(tensor_desc);

    // Add node
    NodePtr in_node = graph->AddNode(in_op_def);
    NodePtr node = graph->AddNode(op_def);
    NodePtr out_node = graph->AddNode(out_op_def);

    // Add edge
    GraphUtils::AddEdge(in_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
    GraphUtils::AddEdge(node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));

    return node;
  }
};

TEST_F(UtestGraphPassesFoldingKernelShapeKernel, ShapeOptimizerSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = init_node(graph);
  NodePtr out_node = node->GetOutDataNodes().at(0);

  ge::Status ret = pass_->Run(node);
  EXPECT_EQ(ge::SUCCESS, ret);

  vector<ConstGeTensorPtr> out_weights = OpDescUtils::GetWeights(out_node);
  if (out_weights.size() > 1) {
    int32_t dim = *(int32_t *)out_weights[1]->GetData().data();
    EXPECT_EQ(11, dim);
  }
}
TEST_F(UtestGraphPassesFoldingKernelShapeKernel, ShapeDataInt32) {
  vector<int64_t> dims_vec_0 = {8, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  (void)TensorUtils::SetRealDimCnt(tensor_desc_0, dims_vec_0.size());
  op_desc_ptr_->AddInputDesc(tensor_desc_0);

  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out);

  std::vector<GeTensorPtr> outputs;
  Status status = kernel_->Compute(node_, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  if (status == ge::SUCCESS) {
    EXPECT_EQ(outputs[0]->GetTensorDesc().GetDataType(), DT_INT32);
    EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 1);
    EXPECT_EQ(outputs[0]->GetData().size(), sizeof(int32_t) * dims_vec_0.size());
    EXPECT_EQ(*(int32_t *)(outputs[0]->GetData().data()), 8);
  }
}

TEST_F(UtestGraphPassesFoldingKernelShapeKernel, ShapeDataInt64) {
  vector<int64_t> dims_vec_0 = {8, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetRealDimCnt(tensor_desc_0, dims_vec_0.size());
  op_desc_ptr_->AddInputDesc(tensor_desc_0);

  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT64);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out);

  std::vector<GeTensorPtr> outputs;
  Status status = kernel_->Compute(node_, outputs);
  EXPECT_EQ(ge::SUCCESS, status);
  if (status == ge::SUCCESS) {
    EXPECT_EQ(outputs[0]->GetTensorDesc().GetDataType(), DT_INT64);
    EXPECT_EQ(outputs[0]->GetTensorDesc().GetShape().GetDimNum(), 1);
    EXPECT_EQ(outputs[0]->GetData().size(), sizeof(int64_t) * dims_vec_0.size());
    EXPECT_EQ(*(int64_t *)(outputs[0]->GetData().data()), 8);
  }
}

TEST_F(UtestGraphPassesFoldingKernelShapeKernel, ShapeInputSizeFail) {
  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT64);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out);

  std::vector<GeTensorPtr> outputs;
  Status status = kernel_->Compute(node_, outputs);
  EXPECT_EQ(NOT_CHANGED, status);
}
