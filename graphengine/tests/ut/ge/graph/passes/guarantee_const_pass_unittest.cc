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
#include <vector>

#define protected public
#define private public
#include "graph/passes/guarantee_const_pass.h"

#include "../ops_stub.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "inc/pass_manager.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace std;

// To check whether the shape of output is correct or not
#define TEST_OPERATOR(op_, input_shapes, output_shapes)                                                 \
  {                                                                                                     \
    auto op = op_;                                                                                      \
    for (auto input_pair : input_shapes) SetInputShape(op, input_pair.first, input_pair.second);        \
    op.InferShapeAndType();                                                                             \
    for (auto output_pair : output_shapes) CheckOutputShape(op, output_pair.first, output_pair.second); \
  }

#define LOOP_VEC(v) for (unsigned i = 0; i < v.size(); i++)

class UtestGraphPassesGuaranteeConstPass : public testing::Test {
 protected:
  void SetUp() { init(); }

  void TearDown() { destory(); }

 private:
  void init() { guarantee_const_op_remove_pass_ = new ::ge::GuaranteeConstPass(); }

  void destory() {
    delete guarantee_const_op_remove_pass_;
    guarantee_const_op_remove_pass_ = NULL;
  }

 protected:
  ge::GuaranteeConstPass *guarantee_const_op_remove_pass_;

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

  /// Init the node which will be passed in graph, isMultiInput represents whether using more than
  /// one data anchor or not.
  NodePtr init_node(ComputeGraphPtr graph, vector<int64_t> dims_vec, vector<int32_t> data_vec, bool isMultiInput,
                    string type) {
    // middle
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", type);
    OpDescPtr in_op_def = std::make_shared<OpDesc>("op_def_in", "test");
    OpDescPtr out_op_def = std::make_shared<OpDesc>("op_def_out", "test");
    OpDescPtr another_in_op_def = std::make_shared<OpDesc>("another_op_def_in", "test");

    // whether using another input data anchor or not
    if (isMultiInput) {
      vector<bool> is_input_const_vec = {true, true};
      op_def->SetIsInputConst(is_input_const_vec);
      AttrUtils::SetInt(op_def, ge::ATTR_NAME_T, (int64_t)DT_INT32);
    }

    // input tensor;
    GeTensorDesc tensor_desc(GeShape(dims_vec), FORMAT_NCHW, DT_INT32);
    ge::ConstGeTensorPtr const_tensor =
        std::make_shared<GeTensor>(tensor_desc, (uint8_t *)&data_vec[0], data_vec.size() * sizeof(int32_t));
    ge::AttrUtils::SetTensor(in_op_def, ge::ATTR_NAME_WEIGHTS, const_tensor);
    op_def->AddInputDesc(tensor_desc);

    // whether using another input data anchor or not
    if (isMultiInput) {
      vector<int64_t> dims_vec_another = {6};
      vector<int32_t> data_vec_another = {1, 2, 3, 4, 5, 6};
      GeTensorDesc another_tensor_desc(GeShape(dims_vec_another), FORMAT_NCHW, DT_INT32);
      ge::ConstGeTensorPtr const_tensor_another = std::make_shared<GeTensor>(
          another_tensor_desc, (uint8_t *)&data_vec_another[0], data_vec_another.size() * sizeof(int32_t));
      ge::AttrUtils::SetTensor(another_in_op_def, ge::ATTR_NAME_WEIGHTS, const_tensor_another);
      op_def->AddInputDesc(another_tensor_desc);
      another_in_op_def->AddOutputDesc(another_tensor_desc);
      out_op_def->AddInputDesc(another_tensor_desc);
    }

    GeTensorDesc tensor_desc_out(GeShape(dims_vec), FORMAT_NCHW, DT_INT32);
    op_def->AddOutputDesc(tensor_desc_out);
    in_op_def->AddOutputDesc(tensor_desc);

    // add attr of out_node
    vector<bool> is_output_const(3, false);
    is_output_const[0] = true;
    out_op_def->SetIsInputConst(is_output_const);
    out_op_def->AddInputDesc(tensor_desc);

    // Add node
    NodePtr in_node = graph->AddNode(in_op_def);
    NodePtr node = graph->AddNode(op_def);
    NodePtr out_node = graph->AddNode(out_op_def);

    // Add edge
    GraphUtils::AddEdge(in_node->GetOutDataAnchor(0), node->GetInDataAnchor(0));
    GraphUtils::AddEdge(node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));

    // when need multi input nodes (which to verify the isolate node function)
    if (isMultiInput) {
      NodePtr another_in_node = graph->AddNode(another_in_op_def);
      GraphUtils::AddEdge(another_in_node->GetOutDataAnchor(0), node->GetInDataAnchor(1));
    }

    return node;
  }
};

TEST_F(UtestGraphPassesGuaranteeConstPass, not_changed) {
  // the original type of op is not guarantee_const
  string type = SIZE;
  // input tensor
  vector<int64_t> dims_vec = {6};
  vector<int32_t> data_vec = {1, 2, 3, 4, 5, 6};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = init_node(graph, dims_vec, data_vec, false, type);
  ge::Status ret = guarantee_const_op_remove_pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestGraphPassesGuaranteeConstPass, get_origenal_type_fail) {
  string type = GUARANTEECONST;
  // input tensor
  vector<int64_t> dims_vec = {6};
  vector<int32_t> data_vec = {1, 2, 3, 4, 5, 6};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = init_node(graph, dims_vec, data_vec, false, type);
  // change the type
  string type2 = "FrameworkOp";
  node->GetOpDesc()->SetType(type2);
  ge::Status ret = guarantee_const_op_remove_pass_->Run(node);
}

TEST_F(UtestGraphPassesGuaranteeConstPass, int32_success_6) {
  // input tensor
  string type = GUARANTEECONST;
  vector<int64_t> dims_vec = {6};
  vector<int32_t> data_vec = {1, 2, 3, 4, 5, 6};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = init_node(graph, dims_vec, data_vec, false, type);
  // when input tensor is [1, 2, 3, 4, 5, 6], return success
  ge::Status output = guarantee_const_op_remove_pass_->Run(node);
  EXPECT_EQ(ge::SUCCESS, output);
}

TEST_F(UtestGraphPassesGuaranteeConstPass, int32_success_2_3) {
  // input tensor
  string type = GUARANTEECONST;
  vector<int64_t> dims_vec = {2, 3};
  vector<int32_t> data_vec = {1, 2, 3, 4, 5, 6};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = init_node(graph, dims_vec, data_vec, false, type);
  // when input tensor is [[1, 2, 3], [4, 5, 6]], return success
  ge::Status output = guarantee_const_op_remove_pass_->Run(node);
  EXPECT_EQ(ge::SUCCESS, output);
}

TEST_F(UtestGraphPassesGuaranteeConstPass, isolate_node_failed) {
  // input tensor
  string type = GUARANTEECONST;
  vector<int64_t> dims_vec = {2, 3};
  vector<int32_t> data_vec = {1, 2, 3, 4, 5, 6};
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // add another input node
  NodePtr node = init_node(graph, dims_vec, data_vec, true, type);
  // when there are more than one input anchors, return failed
  ge::Status output = guarantee_const_op_remove_pass_->Run(node);
  EXPECT_EQ(ge::PARAM_INVALID, output);
}

// IR test, the shape and data type of input should be equal to the shape and data type of output
TEST_F(UtestGraphPassesGuaranteeConstPass, ir_infer_shape) {
  auto input = unordered_map<string, vector<int64_t>>({
      {"x", {3, 5, 3, 4}},
  });
  auto output = unordered_map<string, vector<int64_t>>({
      {"y", {3, 5, 3, 4}},
  });
  auto guaranteeConst = op::GuaranteeConst("guaranteeconst");

  TEST_OPERATOR(guaranteeConst, input, output);
}
