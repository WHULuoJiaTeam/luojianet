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
#include "host_kernels/slice_kernel.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/types.h"
#include "gen_node.h"
#include "graph/op_desc.h"
#include "graph/passes/constant_folding_pass.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#undef protected
#undef private

using namespace ge;
using namespace testing;

class UtestFoldingSliceKernel : public testing::Test {
 protected:
  void SetUp() { init(); }

  void TearDown() { destory(); }

 private:
  void init() { pass_ = new ::ge::ConstantFoldingPass(); }

  void destory() {
    delete pass_;
    pass_ = NULL;
  }

 protected:
  ::ge::ConstantFoldingPass *pass_;

  NodePtr initNode_int32(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_temp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);
    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_temp);
    ge::GeShape op_shape({4});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[4];
    value[0] = 1;
    value[1] = 2;
    value[2] = 3;
    value[3] = 4;
    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(int32_t));
    int32_t value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)&value1, sizeof(int32_t));
    int32_t value2 = 2;
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)&value2, sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);

    OpDescUtils::SetWeights(node_temp, op_weights);
    NodePtr node = graph->AddNode(node_temp);
    return node;
  }

  NodePtr initNode_float(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_FLOAT);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    ge::GeShape op_shape({4});
    GeTensorDesc op_desc(op_shape);
    float value[4];
    value[0] = 1.0;
    value[1] = 2.0;
    value[2] = 3.0;
    value[3] = 4.0;
    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(float));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value1, sizeof(int32_t));
    int32_t value2 = 2;
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value2, sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_errtype(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_UNDEFINED);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    ge::GeShape op_shape({4});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_UNDEFINED);
    int32_t value[4];
    value[0] = 1;
    value[1] = 2;
    value[2] = 3;
    value[3] = 4;
    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value1, sizeof(int32_t));
    int32_t value2 = 2;
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value2, sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_muldims(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    vector<int64_t> dims(4, 2);
    ge::GeShape op_shape(dims);
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[2][2][2][2] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 16 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1[4] = {1, 1, 1, 1};
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value1, 4 * sizeof(int32_t));
    int32_t value2[4] = {1, 1, 1, 1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 4 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_3muldims(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    vector<int64_t> dims(3, 2);
    ge::GeShape op_shape(dims);
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[2][2][2] = {1, 2, 3, 4, 5, 6, 7, 8};

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 8 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1[3] = {1, 1, 1};
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value1, 3 * sizeof(int32_t));
    int32_t value2[3] = {1, 1, 1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 3 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_2muldims(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    vector<int64_t> dims(2, 2);
    ge::GeShape op_shape(dims);
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[2][2] = {1, 2, 3, 4};

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 4 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1[2] = {1, 1};
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value1, 2 * sizeof(int32_t));
    int32_t value2[2] = {1, -1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 2 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_1muldims(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);

    vector<bool> is_input_const(4, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    ge::GeShape op_shape({2});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[2] = {
        1,
        2,
    };

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 2 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value1[1] = {1};
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value1, 1 * sizeof(int32_t));
    int32_t value2[1] = {1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 1 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    return node;
  }

  NodePtr initNode_size_not_equal_fail(ComputeGraphPtr graph) {
    OpDescPtr op_def = std::make_shared<OpDesc>("op_def", "Slice");
    auto node_tmp = GenNodeFromOpDesc(op_def);
    OpDescPtr child_opdef = std::make_shared<OpDesc>("child_opdef", "test");
    child_opdef->SetIsInputConst({false});

    vector<bool> is_input_const(3, true);
    op_def->SetIsInputConst(is_input_const);
    AttrUtils::SetInt(op_def, ATTR_NAME_T, DT_INT32);

    // Add weights
    vector<GeTensorPtr> op_weights = OpDescUtils::MutableWeights(node_tmp);
    ge::GeShape op_shape({3});
    GeTensorDesc op_desc(op_shape, FORMAT_NCHW, DT_INT32);
    int32_t value[3] = {1, 2, 3};

    GeTensorPtr weight0 = std::make_shared<ge::GeTensor>(op_desc, (uint8_t *)value, 3 * sizeof(int32_t));

    GeTensorDesc op_desc_1(op_shape, FORMAT_NCHW, DT_INT32);
    int value1 = 0;
    GeTensorPtr weight1 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)&value1, 1 * sizeof(int32_t));
    int32_t value2[2] = {0, 1};
    GeTensorPtr weight2 = std::make_shared<ge::GeTensor>(op_desc_1, (uint8_t *)value2, 2 * sizeof(int32_t));

    op_weights.push_back(weight0);
    op_weights.push_back(weight1);
    op_weights.push_back(weight2);
    OpDescUtils::SetWeights(node_tmp, op_weights);
    NodePtr node = graph->AddNode(node_tmp);
    NodePtr child_node = graph->AddNode(child_opdef);
    return node;
  }
};

/// test func：SliceKernel::Compute
/// case：optimize op of int
/// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerIntSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_int32(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

/// test func：SliceKernel::Compute
/// case：optimize op of float
/// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerFloatSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_float(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

/// test func：SliceKernel::Compute
/// case：optimize op of initNode_errtype
/// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerErrtypeSuccess) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_errtype(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

/// test func：SliceKernel::Compute
/// case：optimize op of initNode_muldims
/// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerIntMulDims) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_muldims(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

/// test func：SliceKernel::Compute
/// case：optimize op of initNode_3muldims
/// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerInt3MulDims) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_3muldims(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

/// test func：SliceKernel::Compute
/// case：optimize op of initNode_2muldims
/// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerInt2MulDims) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_2muldims(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}
/// test func：SliceKernel::Compute
/// case：optimize op of initNode_1muldims
/// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerInt1MulDims) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_1muldims(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}

/// test func：SliceKernel::Compute
/// case：optimize op of initNode_size_not_equal_fail
/// result： optimize op of slice success
TEST_F(UtestFoldingSliceKernel, SliceOptimizerSizeNotEqual) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = initNode_size_not_equal_fail(graph);
  Status ret = pass_->Run(node);
  EXPECT_EQ(SUCCESS, ret);
}
