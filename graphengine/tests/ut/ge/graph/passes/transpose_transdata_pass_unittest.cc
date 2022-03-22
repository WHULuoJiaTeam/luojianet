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

#include <vector>
#include <gtest/gtest.h>

#define protected public
#define private public
#include "graph/passes/transpose_transdata_pass.h"
#include "graph_builder_utils.h"
#undef private
#undef protected

#include "graph/graph.h"
#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
class UtestGraphPassesTransposeTransdataPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

static ComputeGraphPtr BuildGraphTransposeD() {
  auto builder = ut::GraphBuilder("g1");
  auto transdata1 = builder.AddNode("transdata1", "TransData", 1, 1, FORMAT_NC1HWC0, DT_FLOAT, std::vector<int64_t>({1, 1, 224, 224, 16}));
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NHWC);
  transdata1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transpose1 = builder.AddNode("transpose1", "TransposeD", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NHWC);
  transpose1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 224, 224, 3})));

  auto transdata2 = builder.AddNode("transdata2", "TransData", 1, 1, FORMAT_NCHW, DT_FLOAT, std::vector<int64_t>({1, 3, 224, 224}));
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  transdata2->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224, 16})));

  builder.AddDataEdge(transdata1, 0, transpose1, 0);
  builder.AddDataEdge(transpose1, 0, transdata2, 0);

  return builder.GetGraph();
}

TEST_F(UtestGraphPassesTransposeTransdataPass, test_run) {
  auto compute_graph = BuildGraphTransposeD();
  compute_graph->SetSessionID(0);

  auto transpose = compute_graph->FindNode("transpose1");
  TransposeTransDataPass pass;
  EXPECT_EQ(pass.Run(transpose), SUCCESS);
}
}  // namespace ge
