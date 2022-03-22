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
#include "init/gelib.h"
#include "hybrid/node_executor/aicore/aicore_task_compiler.h"
#undef private
#undef protected

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;

class UtestAiCoreTaskCompiler : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestAiCoreTaskCompiler, test_aicore_task_compiler_init) {
  ge::hybrid::AiCoreTaskCompiler aicore_task_compiler;
  NodePtr node = MakeShared<Node>();
  std::vector<domi::TaskDef> tasks{};
  EXPECT_EQ(aicore_task_compiler.Initialize(), ge::PARAM_INVALID);  // cause: ge lib is nullptr
  EXPECT_EQ(aicore_task_compiler.CompileOp(node, tasks), ge::PARAM_INVALID); // cause: aicore task compiler init failed.
}
} // namespace ge

