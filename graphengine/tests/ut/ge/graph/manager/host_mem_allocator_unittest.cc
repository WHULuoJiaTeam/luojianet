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
#include "graph/manager/host_mem_allocator.h"
#undef protected
#undef private

namespace ge {
class UtestHostMemManagerTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestHostMemManagerTest, malloc_zero_size) {
  HostMemAllocator allocator(RT_MEMORY_HBM);
  EXPECT_EQ(allocator.allocated_blocks_.size(), 0);
  EXPECT_EQ(allocator.Malloc(nullptr, 0), nullptr);
  EXPECT_EQ(allocator.allocated_blocks_.size(), 1);
  EXPECT_EQ(allocator.Malloc(nullptr, 1), nullptr);
  EXPECT_EQ(allocator.allocated_blocks_.size(), 1);
}
} // namespace ge
