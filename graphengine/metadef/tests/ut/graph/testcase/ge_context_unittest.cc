/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <iostream>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/ge_context.h"
#include "graph/ge_global_options.h"
#include "graph/ge_local_context.h"
#include "graph/node.h"
#include "graph_builder_utils.h"

namespace ge {


class GeContextUt : public testing::Test {};

TEST_F(GeContextUt, All) {
  ge::GEContext cont = GetContext();
  cont.Init();
  EXPECT_EQ(cont.GetHostExecFlag(), false);
  EXPECT_EQ(GetMutableGlobalOptions().size(), 0);
  EXPECT_EQ(cont.SessionId(), 0);
  EXPECT_EQ(cont.ContextId(), 0);
  EXPECT_EQ(cont.WorkStreamId(), 0);
  EXPECT_EQ(cont.DeviceId(), 0);
  EXPECT_EQ(cont.TraceId(), 65536);

  cont.SetSessionId(1);
  cont.SetContextId(2);
  cont.SetWorkStreamId(3);
  cont.SetCtxDeviceId(4);
  EXPECT_EQ(cont.SessionId(), 1);
  EXPECT_EQ(cont.ContextId(), 2);
  EXPECT_EQ(cont.WorkStreamId(), 3);
  EXPECT_EQ(cont.DeviceId(), 4);
}

TEST_F(GeContextUt, Plus) {
  std::map<std::string, std::string> session_option{{"ge.exec.placement", "ge.exec.placement"}};
  GetThreadLocalContext().SetSessionOption(session_option);
  std::string exec_placement;
  GetThreadLocalContext().GetOption("ge.exec.placement", exec_placement);
  EXPECT_EQ(exec_placement, "ge.exec.placement");
  ge::GEContext cont = GetContext();
  EXPECT_EQ(cont.GetHostExecFlag(), false);
  std::map<std::string, std::string> session_option2{{"ge.exec.sessionId", "12345678987654321"}};
  GetThreadLocalContext().SetSessionOption(session_option2);
  cont.Init();
  std::map<std::string, std::string> session_option3{{"ge.exec.deviceId", "12345678987654321"}};
  GetThreadLocalContext().SetSessionOption(session_option3);
  cont.Init();
  std::map<std::string, std::string> session_option4{{"ge.exec.jobId", "12345"}};
  GetThreadLocalContext().SetSessionOption(session_option4);
  cont.Init();
  std::map<std::string, std::string> session_option5{{"ge.exec.jobId", "65536"}};
  GetThreadLocalContext().SetSessionOption(session_option5);
  cont.Init();
}

}  // namespace ge