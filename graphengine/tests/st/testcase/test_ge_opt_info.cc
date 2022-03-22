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
#include "external/ge/ge_api.h"
#include "easy_graph/builder/graph_dsl.h"
#include "graph/compute_graph.h"
#include "framework/common/types.h"
#include "graph/ge_local_context.h"
#include "ge_graph_dsl/graph_dsl.h"

namespace ge {
class STEST_opt_info : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(STEST_opt_info, get_opt_info_all) {
  std::map<std::string, std::string> options = {{ge::SOC_VERSION, "Ascend310"}};
  GetThreadLocalContext().SetGlobalOption(options);

  ///     data1   data2
  ///       \    /
  ///        add
  // build graph
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->NODE("add"));
  };

  auto graph = ToGeGraph(g1);

  // new session & add graph
  Session session(options);
  auto ret = session.AddGraph(1, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  auto itr = graph_options.find("opt_module.fe");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "all");
  itr = graph_options.find("opt_module.pass");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "all");
  itr = graph_options.find("opt_module.op_tune");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "all");
  itr = graph_options.find("opt_module.rl_tune");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "all");
  itr = graph_options.find("opt_module.aoe");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "all");
}

TEST_F(STEST_opt_info, get_opt_info_success) {
  std::map<std::string, std::string> options = {{ge::SOC_VERSION, "Ascend910"}};
  GetThreadLocalContext().SetGlobalOption(options);

  ///     data1   data2
  ///       \    /
  ///        add
  // build graph
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->NODE("add"));
  };

  auto graph = ToGeGraph(g1);

  // new session & add graph
  Session session(options);
  auto ret = session.AddGraph(1, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(1, inputs);
  EXPECT_EQ(ret, SUCCESS);

  std::map<std::string, std::string> graph_options = GetThreadLocalContext().GetAllGraphOptions();
  auto itr = graph_options.find("opt_module.fe");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "all");
  itr = graph_options.find("opt_module.pass");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "all");
  itr = graph_options.find("opt_module.op_tune");
  EXPECT_NE(itr, graph_options.end());
  EXPECT_EQ(itr->second, "all");
}
}  // namespace ge
