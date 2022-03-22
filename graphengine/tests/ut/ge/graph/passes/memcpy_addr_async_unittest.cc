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
#include <cstdint>
#include <memory>
#include <string>

#define private public
#include "graph/passes/memcpy_addr_async_pass.h"
#include "common/ge_inner_error_codes.h"
#include "inc/pass_manager.h"
#undef private

namespace ge {
class UtestMemcpyAddrAsyncPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestMemcpyAddrAsyncPass, run) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>();
  op->SetType(STREAMSWITCH);
  op->SetName("stream_switch");
  op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node = graph->AddNode(op);
  graph->SetGraphUnknownFlag(true);
  MemcpyAddrAsyncPass pass;
  Status ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}
}  // namespace ge
