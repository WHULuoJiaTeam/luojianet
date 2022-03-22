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
#include <unordered_map>

#define protected public
#define private public
#include "graph/passes/stop_gradient_pass.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "external/graph/operator_reg.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/operator.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;

// for ir
REG_OP(StopGradient)
    .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                          DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                           DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .OP_END_FACTORY_REG(StopGradient)

        IMPLEMT_INFERFUNC(StopGradient, StopGradientInfer) {
  TensorDesc input_desc = op.GetInputDesc("x");
  (void)op.UpdateOutputDesc("y", input_desc);
  return GRAPH_SUCCESS;
}

INFER_FUNC_REG(StopGradient, StopGradientInfer);

#define TEST_OPERATOR(op_, input_shapes, output_shapes)                                                 \
  {                                                                                                     \
    auto op = op_;                                                                                      \
    for (auto input_pair : input_shapes) SetInputShape(op, input_pair.first, input_pair.second);        \
    op.InferShapeAndType();                                                                             \
    for (auto output_pair : output_shapes) CheckOutputShape(op, output_pair.first, output_pair.second); \
  }
#define LOOP_VEC(v) for (unsigned i = 0; i < v.size(); i++)

class UtestGraphPassesStopGradientPass : public testing::Test {
 protected:
  void SetUp() { init(); }

  void TearDown() { destory(); }

 private:
  void init() {
    pass_ = new ::ge::StopGradientPass();

    graph_ = std::make_shared<ge::ComputeGraph>("default");
    op_desc_ptr_ = std::make_shared<OpDesc>("stop_gradient", STOPGRADIENT);
    node_ = std::make_shared<Node>(op_desc_ptr_, graph_);
    kernel_ = KernelFactory::Instance().Create(STOPGRADIENT);
  }

  void destory() {
    delete pass_;
    pass_ = NULL;
  }

 protected:
  ge::StopGradientPass *pass_;
  ge::ComputeGraphPtr graph_;
  OpDescPtr op_desc_ptr_;
  NodePtr node_;
  shared_ptr<Kernel> kernel_;

  void SetInputShape(Operator op, string name, vector<int64_t> shape) {
    TensorDesc td = op.GetInputDesc(name);
    td.SetShape(ge::Shape(shape));
    op.UpdateInputDesc(name, td);
  }

  void CheckOutputShape(Operator op, string name, vector<int64_t> shape) {
    ge::Shape s = op.GetOutputDesc(name).GetShape();
    EXPECT_EQ(s.GetDims().size(), shape.size());
    LOOP_VEC(shape) EXPECT_EQ(s.GetDim(i), shape[i]);
  }

  NodePtr init_node(ComputeGraphPtr graph, string &type) {
    // middle
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", type);
    OpDescPtr in_op_def_0 = std::make_shared<OpDesc>("op_def_in", "test");
    OpDescPtr out_op_def = std::make_shared<OpDesc>("op_def_in", "test");

    // in_op_def_0
    vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
    vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
    (void)TensorUtils::SetRealDimCnt(tensor_desc_0, dims_vec_0.size());
    ge::ConstGeTensorPtr constTensor_0 =
        std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)&data_vec_0[0], data_vec_0.size() * sizeof(int32_t));
    ge::AttrUtils::SetTensor(in_op_def_0, ge::ATTR_NAME_WEIGHTS, constTensor_0);
    vector<int64_t> dims = {2, 2, 4, 3, 2};
    ge::GeShape shape_desc(dims);
    GeTensorDesc tensor_desc(shape_desc);
    in_op_def_0->AddOutputDesc(tensor_desc);
    in_op_def_0->SetType("Constant");

    // op_def
    GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT32);
    op_def->AddInputDesc(tensor_desc_0);
    op_def->AddOutputDesc(tensor_desc_out);
    vector<bool> is_input_const_vec = {
        true,
    };
    op_def->SetIsInputConst(is_input_const_vec);
    AttrUtils::SetInt(op_def, ge::ATTR_NAME_T, (int64_t)DT_INT32);

    // add attr of out_node
    vector<bool> is_input_const(1);
    is_input_const[0] = true;
    out_op_def->SetIsInputConst(is_input_const);
    out_op_def->AddInputDesc(tensor_desc_0);

    // Add node
    NodePtr in_node_0 = graph->AddNode(in_op_def_0);
    NodePtr node = graph->AddNode(op_def);
    NodePtr out_node = graph->AddNode(out_op_def);

    // Add edge
    GraphUtils::AddEdge(in_node_0->GetOutDataAnchor(0), node->GetInDataAnchor(0));
    GraphUtils::AddEdge(node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));

    return node;
  }
};

TEST_F(UtestGraphPassesStopGradientPass, success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  string type = STOPGRADIENT;
  NodePtr node = init_node(graph, type);
  ge::Status ret = pass_->Run(node);
  EXPECT_EQ(ge::SUCCESS, ret);
}

TEST_F(UtestGraphPassesStopGradientPass, not_changed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  string type = SIZE;
  NodePtr node = init_node(graph, type);
  ge::Status ret = pass_->Run(node);
  EXPECT_EQ(ge::SUCCESS, ret);
}

TEST_F(UtestGraphPassesStopGradientPass, get_origenal_type_fail) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  string type = STOPGRADIENT;
  NodePtr node = init_node(graph, type);
  string type2 = "FrameworkOp";
  node->GetOpDesc()->SetType(type2);
  ge::Status ret = pass_->Run(node);
}
TEST_F(UtestGraphPassesStopGradientPass, size_check_fail) {
  vector<int64_t> dims_vec_0 = {8, 2};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr_->AddInputDesc(tensor_desc_0);

  vector<int64_t> dims_vec_1 = {3, 4, 5};
  GeTensorDesc tensor_desc_1(GeShape(dims_vec_1), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr_->AddInputDesc(tensor_desc_1);

  GeTensorDesc tensor_desc_out(GeShape(), FORMAT_NCHW, DT_INT64);
  op_desc_ptr_->AddOutputDesc(tensor_desc_out);

  ge::Status ret = pass_->Run(node_);
  EXPECT_EQ(ge::FAILED, ret);
}

TEST_F(UtestGraphPassesStopGradientPass, ir_infer_shape) {
  auto i = std::unordered_map<string, vector<int64_t>>({
      {"x", {2, 1, 5, 3}},
  });
  auto o = std::unordered_map<string, vector<int64_t>>({
      {"y", {2, 1, 5, 3}},
  });

  auto test_op = op::StopGradient("test_op");

  TEST_OPERATOR(test_op, i, o);
}
