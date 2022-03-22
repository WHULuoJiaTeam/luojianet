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
#include "graph/passes/dimension_adjust_pass.h"

#include "common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "common/ge_inner_error_codes.h"
#include "common/op/ge_op_utils.h"
#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "inc/kernel.h"
#include "inc/kernel_factory.h"
#undef protected
#undef private

using namespace std;
using namespace testing;

namespace ge {

class TestExpandDimKernel : public Kernel {
 public:
  Status Compute(const NodePtr &node_ptr) override {
    return SUCCESS;
  }
};
REGISTER_KERNEL(EXPANDDIMS, TestExpandDimKernel);
class TestExpandDimKernelNotChange : public Kernel {
 public:
  Status Compute(const NodePtr &node_ptr) override {
    return NOT_CHANGED;
  }
};

class UtestGraphPassesDimensionAdjustPass : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {
    KernelFactory::Instance().creator_map_.clear();
  }
};

TEST_F(UtestGraphPassesDimensionAdjustPass, succ) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");

  ge::OpDescPtr data_op_desc = make_shared<ge::OpDesc>("data", CONSTANTOP);
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {1, 2, 3};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(data_op_desc, data_tensor);
  data_op_desc->AddOutputDesc(data_tensor_desc);
  NodePtr data_node = graph->AddNode(data_op_desc);
  data_node->Init();

  // add dim node
  ge::OpDescPtr dim_op_desc = make_shared<ge::OpDesc>("dim", CONSTANTOP);
  vector<int32_t> dim_value_vec = {0};
  GeTensorDesc dim_tensor_desc(ge::GeShape(), FORMAT_NCHW, DT_INT32);
  GeTensorPtr dim_tensor =
      make_shared<GeTensor>(dim_tensor_desc, (uint8_t *)dim_value_vec.data(), dim_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(dim_op_desc, dim_tensor);
  dim_op_desc->AddOutputDesc(dim_tensor_desc);
  NodePtr dim_node = graph->AddNode(dim_op_desc);
  dim_node->Init();

  // add expanddims node
  OpDescPtr expanddims_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
  vector<int64_t> expanddims_vec = {1, 1, 2, 3};
  GeTensorDesc expanddims_tensor_desc(ge::GeShape(expanddims_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr expanddims_tensor = make_shared<GeTensor>(expanddims_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                        data_value_vec.size() * sizeof(int32_t));
  OpDescUtils::SetWeights(expanddims_op_desc, expanddims_tensor);
  expanddims_op_desc->AddInputDesc(data_tensor_desc);
  expanddims_op_desc->AddInputDesc(dim_tensor_desc);
  expanddims_op_desc->AddOutputDesc(expanddims_tensor_desc);
  NodePtr op_node = graph->AddNode(expanddims_op_desc);
  op_node->Init();

  // add output node
  OpDescPtr netoutput_op_desc = std::make_shared<OpDesc>("NetOutput", "NetOutput");
  netoutput_op_desc->AddInputDesc(expanddims_tensor_desc);
  NodePtr netoutput_node = graph->AddNode(netoutput_op_desc);
  netoutput_node->Init();

  // add edge
  GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(dim_node->GetOutDataAnchor(0), op_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(op_node->GetOutDataAnchor(0), netoutput_node->GetInDataAnchor(0));

  std::shared_ptr<DimensionAdjustPass> pass = make_shared<DimensionAdjustPass>();
  NamesToPass names_to_passes;
  EXPECT_EQ(4, graph->GetDirectNodesSize());
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(SUCCESS, ret);
  EXPECT_EQ(2, op_node->GetOwnerComputeGraph()->GetDirectNodesSize());
}

TEST_F(UtestGraphPassesDimensionAdjustPass, input_node_is_nullptr) {
  std::shared_ptr<DimensionAdjustPass> pass = make_shared<DimensionAdjustPass>();
  ge::NodePtr node = nullptr;
  ge::Status ret = pass->Run(node);
  EXPECT_EQ(PARAM_INVALID, ret);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, node_op_desc_is_nullptr) {
  NodePtr op_node = make_shared<Node>(nullptr, nullptr);

  std::shared_ptr<DimensionAdjustPass> pass = make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(PARAM_INVALID, ret);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, node_get_original_type_failed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", FRAMEWORKOP);
  NodePtr op_node = make_shared<Node>(expanddim_op_desc, graph);

  std::shared_ptr<DimensionAdjustPass> pass = make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, node_not_register_op) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", FRAMEWORKOP);
  AttrUtils::SetStr(expanddim_op_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "expanddims_fake");
  NodePtr op_node = make_shared<Node>(expanddim_op_desc, graph);

  std::shared_ptr<DimensionAdjustPass> pass = make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(SUCCESS, ret);
}

TEST_F(UtestGraphPassesDimensionAdjustPass, node_compute_failed) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr expanddim_op_desc = std::make_shared<OpDesc>("Expanddims", EXPANDDIMS);
  NodePtr op_node = make_shared<Node>(expanddim_op_desc, graph);

  std::shared_ptr<DimensionAdjustPass> pass = make_shared<DimensionAdjustPass>();
  ge::Status ret = pass->Run(op_node);
  EXPECT_EQ(SUCCESS, ret);
}

}  // namespace ge
