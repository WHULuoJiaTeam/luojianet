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
#include <vector>

#include "runtime/rt.h"

#define protected public
#define private public
#include "single_op/stream_resource.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;

class UtestStreamResource : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

  rtStream_t stream;
};

/*
TEST_F(UtestStreamResource, test_cache_op) {
  StreamResource res((uintptr_t)1);
  auto *op = new SingleOp();
  string stub_name = "stubFunc";
  const void *key = stub_name.c_str();
  ASSERT_EQ(res.GetOperator(key), nullptr);
  res.CacheOperator(key, op);
  ASSERT_NE(res.GetOperator(key), nullptr);
}
*/

TEST_F(UtestStreamResource, test_malloc_memory) {
  StreamResource res((uintptr_t)1);
  string purpose("test");
  ASSERT_NE(res.MallocMemory(purpose, 100), nullptr);
  ASSERT_NE(res.MallocMemory(purpose, 100), nullptr);
  ASSERT_NE(res.MallocMemory(purpose, 100), nullptr);
}

TEST_F(UtestStreamResource, test_build_op) {
  StreamResource res((uintptr_t)1);
  ModelData model_data;
  SingleOp *single_op = nullptr;
  DynamicSingleOp *dynamic_single_op = nullptr;
  res.op_map_[0].reset(single_op);
  res.dynamic_op_map_[1].reset(dynamic_single_op);

  ThreadPool *thread_pool = nullptr;
  EXPECT_EQ(res.GetThreadPool(&thread_pool), SUCCESS);

  EXPECT_EQ(res.GetOperator(0), nullptr);
  EXPECT_EQ(res.GetDynamicOperator(1), nullptr);
  EXPECT_EQ(res.BuildOperator(model_data, &single_op, 0), SUCCESS);
  EXPECT_EQ(res.BuildDynamicOperator(model_data, &dynamic_single_op, 1), SUCCESS);
}

/*
TEST_F(UtestStreamResource, test_do_malloc_memory) {
  size_t max_allocated = 0;
  vector<uint8_t *> allocated;
  string purpose("test");

  StreamResource res((uintptr_t)1);
  uint8_t *ret = res.DoMallocMemory(purpose, 100, max_allocated, allocated);
  ASSERT_EQ(allocated.size(), 1);
  ASSERT_NE(allocated.back(), nullptr);
  ASSERT_EQ(max_allocated, 100);

  res.DoMallocMemory(purpose, 50, max_allocated, allocated);
  res.DoMallocMemory(purpose, 99, max_allocated, allocated);
  res.DoMallocMemory(purpose, 100, max_allocated, allocated);
  ASSERT_EQ(allocated.size(), 1);
  ASSERT_EQ(max_allocated, 100);

  res.DoMallocMemory(purpose, 101, max_allocated, allocated);
  ASSERT_EQ(allocated.size(), 2);
  ASSERT_EQ(max_allocated, 101);

  for (auto res : allocated) {
      rtFree(res);
  }
}
*/
