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
#include "graph/load/model_manager/model_utils.h"
#include "graph/manager/graph_var_manager.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"

using namespace std;

namespace ge {
class UtestModelUtils : public testing::Test {
 protected:
  void TearDown() {}
};

static NodePtr CreateNode(ComputeGraph &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetSize(tensor, 64);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  return graph.AddNode(op_desc);
}

// test ModelUtils::GetVarAddr
TEST_F(UtestModelUtils, get_var_addr_hbm) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = pf;
  runtime_param.var_size = 16;

  int64_t offset = 8;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_HBM;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  uint8_t *var_addr = nullptr;
  EXPECT_EQ(ModelUtils::GetVarAddr(runtime_param, op_desc, offset, 0, var_addr), SUCCESS);
  EXPECT_EQ(runtime_param.var_base + offset - runtime_param.logic_var_base, var_addr);
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, get_var_addr_rdma_hbm) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = pf;

  int64_t offset = 8;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_RDMA_HBM;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  uint8_t *var_addr = nullptr;
  EXPECT_EQ(ModelUtils::GetVarAddr(runtime_param, op_desc, offset, 0, var_addr), SUCCESS);
  EXPECT_EQ(reinterpret_cast<uint8_t *>(offset), var_addr);
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, get_var_addr_rdma_hbm_negative_offset) {
  uint8_t test = 2;
  uint8_t *pf = &test;
  RuntimeParam runtime_param;
  runtime_param.session_id = 0;
  runtime_param.logic_var_base = 0;
  runtime_param.var_base = pf;

  int64_t offset = -1;
  EXPECT_EQ(VarManager::Instance(runtime_param.session_id)->Init(0, 0, 0, 0), SUCCESS);
  EXPECT_NE(VarManager::Instance(runtime_param.session_id)->var_resource_, nullptr);
  VarManager::Instance(runtime_param.session_id)->var_resource_->var_offset_map_[offset] = RT_MEMORY_RDMA_HBM;
  std::shared_ptr<OpDesc> op_desc = std::make_shared<OpDesc>("test", "test");
  uint8_t *var_addr = nullptr;
  EXPECT_NE(ModelUtils::GetVarAddr(runtime_param, op_desc, offset, 0, var_addr), SUCCESS);
  VarManager::Instance(runtime_param.session_id)->Destory();
}

TEST_F(UtestModelUtils, test_GetInputDataAddrs_input_const) {
  RuntimeParam runtime_param;
  uint8_t weight_base_addr = 0;
  runtime_param.session_id = 0;
  runtime_param.weight_base = &weight_base_addr;
  runtime_param.weight_size = 64;

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 2, 1);
  auto op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  vector<bool> is_input_const = {true, true};
  op_desc->SetIsInputConst(is_input_const);
  {
    auto tensor_desc = op_desc->MutableInputDesc(0);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 64);
    tensor_desc->SetShape(GeShape({1, 1}));
    tensor_desc->SetOriginShape(GeShape({1, 1}));
    TensorUtils::SetDataOffset(*tensor_desc, 0);
  }
  {
    auto tensor_desc = op_desc->MutableInputDesc(1);
    EXPECT_NE(tensor_desc, nullptr);
    TensorUtils::SetSize(*tensor_desc, 32);
    tensor_desc->SetShape(GeShape({1, 0}));
    tensor_desc->SetOriginShape(GeShape({1, 0}));
    TensorUtils::SetDataOffset(*tensor_desc, 64);
  }
  vector<void *> input_data_addr = ModelUtils::GetInputDataAddrs(runtime_param, op_desc);
  EXPECT_EQ(input_data_addr.size(), 2);
  EXPECT_EQ(input_data_addr.at(0), static_cast<void *>(&weight_base_addr + 0));
  EXPECT_EQ(input_data_addr.at(1), static_cast<void *>(&weight_base_addr + 64));
}
}  // namespace ge
