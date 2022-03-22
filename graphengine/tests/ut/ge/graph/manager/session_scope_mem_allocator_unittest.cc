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
#include <memory>

#include "graph/anchor.h"
#include "graph/attr_value.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "omg/omg_inner_types.h"

#define protected public
#define private public
#include "graph/manager/graph_mem_manager.h"
#undef protected
#undef private

using namespace std;
using namespace testing;
using namespace ge;
using domi::GetContext;

class UtestSessionScopeMemAllocator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { GetContext().out_nodes_map.clear(); }
};

TEST_F(UtestSessionScopeMemAllocator, initialize_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  MemManager::Instance().Finalize();
}

TEST_F(UtestSessionScopeMemAllocator, malloc_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(1000, 0);
  EXPECT_NE(nullptr, ptr);
  MemManager::Instance().Finalize();
}

TEST_F(UtestSessionScopeMemAllocator, free_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(100, 0);
  EXPECT_NE(nullptr, ptr);
  ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(100, 0);
  EXPECT_NE(nullptr, ptr);

  EXPECT_EQ(SUCCESS, MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Free(0));
  EXPECT_NE(SUCCESS, MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Free(0));
  MemManager::Instance().Finalize();
}

TEST_F(UtestSessionScopeMemAllocator, free_success_session) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  mem_type.push_back(RT_MEMORY_P2P_DDR);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(100, 0);
  EXPECT_NE(nullptr, ptr);
  ptr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(100, 0);
  EXPECT_NE(nullptr, ptr);
  for (auto memory_type : MemManager::Instance().GetAllMemoryType()) {
    if (RT_MEMORY_P2P_DDR == memory_type) {
      EXPECT_NE(MemManager::Instance().SessionScopeMemInstance(memory_type).Free(0), SUCCESS);
    } else {
      EXPECT_EQ(MemManager::Instance().SessionScopeMemInstance(memory_type).Free(0), SUCCESS);
    }
  }
  MemManager::Instance().Finalize();
}
