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

#define protected public
#define private public
#include "graph/manager/graph_var_manager.h"
#include "graph/ge_context.h"
#undef protected
#undef private

namespace ge {
class UtestGraphVarManagerTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGraphVarManagerTest, test_get_total_memory_size) {
  size_t total_mem_size = 0;
  Status ret = VarManager::Instance(0)->GetTotalMemorySize(total_mem_size);
  EXPECT_EQ(total_mem_size, 1024UL * 1024UL * 1024UL);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, test_set_memory_malloc_size_no_related_option) {
  const map<string, string> options{};
  Status ret = VarManager::Instance(0)->SetMemoryMallocSize(options);
  EXPECT_EQ(VarManager::Instance(0)->graph_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (26.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(0)->var_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (5.0f / 32.0f)));
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, test_set_memory_malloc_size_with_user_specify_graph_mem_max_size) {
  const map<string, string> options{{"ge.graphMemoryMaxSize", "536870912"}};
  Status ret = VarManager::Instance(0)->SetMemoryMallocSize(options);
  EXPECT_EQ(VarManager::Instance(0)->graph_mem_max_size_, floor(1024UL * 1024UL * 1024UL / 2));
  EXPECT_EQ(VarManager::Instance(0)->var_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (5.0f / 32.0f)));
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphVarManagerTest, test_set_memory_malloc_size_with_user_specify_var_mem_max_size) {
  const map<string, string> options{{"ge.variableMemoryMaxSize", "536870912"}};
  Status ret = VarManager::Instance(0)->SetMemoryMallocSize(options);
  EXPECT_EQ(VarManager::Instance(0)->graph_mem_max_size_, floor(1024UL * 1024UL * 1024UL * (26.0f / 32.0f)));
  EXPECT_EQ(VarManager::Instance(0)->var_mem_max_size_, floor(1024UL * 1024UL * 1024UL / 2));
  EXPECT_EQ(ret, SUCCESS);
}
} // namespace ge
