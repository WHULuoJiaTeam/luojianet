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
#include "session/session_manager.h"
#undef private
#undef protected


using namespace std;

namespace ge {
class Utest_SessionManager : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(Utest_SessionManager, build_graph_failed) {
  map<string, string> session_manager_option;
  map<string, string> session_option;
  SessionManager *session_manager = new SessionManager();
  uint64_t session_id = 0;
  uint32_t graph_id = 0;
  std::vector<ge::Tensor> inputs;

  Status ret = session_manager->BuildGraph(session_id, graph_id, inputs);
  EXPECT_EQ(ret, ge::GE_SESSION_MANAGER_NOT_INIT);

  session_manager->Initialize(session_manager_option);
  ret = session_manager->BuildGraph(session_id, graph_id, inputs);
  EXPECT_NE(ret, ge::SUCCESS);
  delete session_manager;
}

TEST_F(Utest_SessionManager, RungraphAsync_before_init) {
  SessionManager *session_manager = new SessionManager();
  SessionId session_id;
  uint32_t graph_id = 0;
  std::vector<ge::Tensor> inputs;
  RunAsyncCallback callback;
  Status ret = session_manager->RunGraphAsync(session_id, graph_id, inputs, callback);
  EXPECT_EQ(ret, ge::GE_SESSION_MANAGER_NOT_INIT);
  delete session_manager;
}

TEST_F(Utest_SessionManager, RungraphAsync_failed) {
  map<string, string> session_manager_option;
  SessionManager *session_manager = new SessionManager();
  session_manager->Initialize(session_manager_option);

  SessionId session_id;
  uint32_t graph_id = 0;
  std::vector<ge::Tensor> inputs;
  RunAsyncCallback callback;
  Status ret = session_manager->RunGraphAsync(session_id, graph_id, inputs, callback);
  EXPECT_EQ(ret, ge::GE_SESSION_NOT_EXIST);
  delete session_manager;
}

}  // namespace ge
