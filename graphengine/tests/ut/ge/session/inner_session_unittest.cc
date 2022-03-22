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
#include "session/inner_session.h"

using namespace std;

namespace ge {
class UtestInnerSession : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestInnerSession, build_graph_success) {
  std::map <string, string> options;
  uint64_t session_id = 1;
  InnerSession inner_seesion(session_id, options);
  std::vector<ge::Tensor> inputs;
  ge::Tensor tensor;
  inputs.emplace_back(tensor);
  Status ret = inner_seesion.BuildGraph(1, inputs);
  EXPECT_NE(ret, ge::SUCCESS);
}

TEST_F(UtestInnerSession, initialize) {
  std::map<std::string, std::string> options = {};
  uint64_t session_id = 1;
  InnerSession inner_session(session_id, options);
  EXPECT_EQ(inner_session.Initialize(), SUCCESS);
  EXPECT_EQ(inner_session.Finalize(), SUCCESS);
}

TEST_F(UtestInnerSession, check_op_precision_mode) {
  std::map<std::string, std::string> options = {
    {ge::OP_PRECISION_MODE, "./op_precision_mode.ini"}
  };
  uint64_t session_id = 1;
  InnerSession inner_session(session_id, options);
  auto ret = inner_session.Initialize();
  EXPECT_NE(ret, ge::SUCCESS);
}
}  // namespace ge
