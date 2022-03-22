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
#include <gmock/gmock.h>
#include <vector>

#define private public
#define protected public
#include "hybrid/node_executor/node_executor.h"
#undef protected
#undef private

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

namespace {
  bool finalized = false;
}

class NodeExecutorTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() { }
};

class FailureNodeExecutor : public NodeExecutor {
 public:
  Status Initialize() override {
    return INTERNAL_ERROR;
  }
};

class SuccessNodeExecutor : public NodeExecutor {
 public:
  Status Initialize() override {
    initialized = true;
    finalized = false;
    return SUCCESS;
  }

  Status Finalize() override {
    finalized = true;
  }

  bool initialized = false;
};

REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICORE, FailureNodeExecutor);
REGISTER_NODE_EXECUTOR_BUILDER(NodeExecutorManager::ExecutorType::AICPU_TF, SuccessNodeExecutor);

TEST_F(NodeExecutorTest, TestGetOrCreateExecutor) {
  auto &manager = NodeExecutorManager::GetInstance();
  const NodeExecutor *executor = nullptr;
  Status ret = SUCCESS;
  // no builder
  ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::RESERVED, &executor);
  ASSERT_EQ(ret, INTERNAL_ERROR);
  // initialize failure
  ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::AICORE, &executor);
  ASSERT_EQ(ret, INTERNAL_ERROR);
  ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::AICPU_TF, &executor);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_TRUE(executor != nullptr);
  ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::AICPU_TF, &executor);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_TRUE(executor != nullptr);
  ASSERT_TRUE(((SuccessNodeExecutor*)executor)->initialized);
}

TEST_F(NodeExecutorTest, TestInitAndFinalize) {
  auto &manager = NodeExecutorManager::GetInstance();
  manager.FinalizeExecutors();
  manager.FinalizeExecutors();
  manager.EnsureInitialized();
  manager.EnsureInitialized();
  const NodeExecutor *executor = nullptr;
  auto ret = manager.GetOrCreateExecutor(NodeExecutorManager::ExecutorType::AICPU_TF, &executor);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_TRUE(executor != nullptr);
  ASSERT_TRUE(((SuccessNodeExecutor*)executor)->initialized);
  manager.FinalizeExecutors();
  ASSERT_FALSE(manager.executors_.empty());
  manager.FinalizeExecutors();
  ASSERT_TRUE(manager.executors_.empty());
  ASSERT_TRUE(finalized);
}
} // namespace ge
