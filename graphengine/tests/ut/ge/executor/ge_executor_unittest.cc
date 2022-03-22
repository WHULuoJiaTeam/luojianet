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

#define private public
#define protected public
#include "executor/ge_executor.h"
#include "graph/utils/tensor_utils.h"

using namespace std;

namespace ge {
class UtestGeExecutor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeExecutor, test_single_op_exec) {
  GeExecutor exeutor;
  ModelData model_data;
  string model_name = "1234";

  EXPECT_EQ(exeutor.LoadSingleOp(model_name, model_data, nullptr, nullptr), ACL_ERROR_GE_INTERNAL_ERROR);
  EXPECT_EQ(exeutor.LoadDynamicSingleOp(model_name, model_data, nullptr, nullptr), PARAM_INVALID);
}

TEST_F(UtestGeExecutor, test_ge_initialize) {
  GeExecutor executor;
  EXPECT_EQ(executor.Initialize(), SUCCESS);
  EXPECT_EQ(executor.Initialize(), SUCCESS);
}
}  // namespace ge