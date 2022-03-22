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

class UtestGraphCachingAllocatorTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { GetContext().out_nodes_map.clear(); }
};

TEST_F(UtestGraphCachingAllocatorTest, initialize_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  MemManager::Instance().Finalize();
}

TEST_F(UtestGraphCachingAllocatorTest, malloc_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kMByteSize);
  EXPECT_NE(nullptr, ptr);
  MemManager::Instance().Finalize();
}

TEST_F(UtestGraphCachingAllocatorTest, extend_malloc_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kMByteSize);
  EXPECT_NE(nullptr, ptr);
  ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit32*kMByteSize);
  EXPECT_NE(nullptr, ptr);
  MemManager::Instance().Finalize();
}

TEST_F(UtestGraphCachingAllocatorTest, malloc_same_success) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit8*kMByteSize);
  EXPECT_NE(nullptr, ptr);
  uint8_t *ptr1 = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit8*kMByteSize);
  EXPECT_NE(nullptr, ptr1);
  uint8_t *ptr2 = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit8*kMByteSize);
  EXPECT_NE(nullptr, ptr2);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr), SUCCESS);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr1), SUCCESS);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr2), SUCCESS);
  ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kBinSizeUnit8*kMByteSize, ptr1);
  EXPECT_EQ(ptr, ptr1);
  MemManager::Instance().Finalize();
}

TEST_F(UtestGraphCachingAllocatorTest, malloc_statics) {
  std::vector<rtMemType_t> mem_type;
  mem_type.push_back(RT_MEMORY_HBM);
  EXPECT_EQ(MemManager::Instance().Initialize(mem_type), SUCCESS);
  uint8_t *ptr = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kMByteSize);
  EXPECT_NE(nullptr, ptr);
  uint8_t *ptr1 = MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Malloc(kKByteSize);
  EXPECT_NE(nullptr, ptr);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr), SUCCESS);
  EXPECT_EQ(MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(ptr1), SUCCESS);
  MemManager::Instance().CachingInstance(RT_MEMORY_HBM).FreeCachedBlocks();
  MemManager::Instance().Finalize();
}