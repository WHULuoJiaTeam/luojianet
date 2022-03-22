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
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/assert/graph_assert.h"

using namespace std;
using namespace ge;
namespace {
/** data a = 2;
 *  for(int i =0; i<5; ++i){
 *    a=a * 2;
 * }
 *  return a;
 *                     ----------------------------------------------|
 *                    |  const(5)             exit  const(1)         |
 *                    |     \                  /       \             |
 *   data(i)--Enter--merge--less--loopcond--switch-----add-----nextiteration
 *                       \________________\___/
 *                                   ------\------------------------|
 *                                  |       \        const(2)       |
 *                                  |        \         \            |
 *                 data(a)--Enter--merge--switch------mul-----nextiteration
 *                                            \
 *                                           exit
 *                                             \
 *                                           netoutput
 *
 **/
Graph BuildV1ControlFlowGraph() {
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                  data_value_vec.size() * sizeof(int32_t));

  auto enter = OP_CFG(ENTER).Attr(ENTER_ATTR_FRAME_NAME, "1");
  auto const_op = OP_CFG(CONSTANT).Weight(data_tensor);

  DEF_GRAPH(g1) {
    CHAIN(NODE("data_i", DATA)
              ->NODE("enter_i", enter)
              ->EDGE(0, 0)
              ->NODE("merge_i", MERGE)
              ->NODE("less", LESS)
              ->NODE("loopcond", LOOPCOND));
    CHAIN(NODE("const_1", const_op)
              ->EDGE(0, 1)
              ->NODE("add", ADD)
              ->NODE("iteration_i", NEXTITERATION)
              ->EDGE(0, 1)
              ->NODE("merge_i"));
    CHAIN(NODE("const_5", const_op)->EDGE(0, 1)->NODE("less"));
    CHAIN(NODE("loopcond")
              ->EDGE(0, 1)
              ->NODE("switch_i", SWITCH)
              ->EDGE(0, 0)
              ->NODE("exit_i", EXIT)
              ->EDGE(0, 1)
              ->NODE("netoutput", NETOUTPUT));
    CHAIN(NODE("merge_i")->EDGE(0, 0)->NODE("switch_i")->EDGE(1, 0)->NODE("add"));
    CHAIN(NODE("data_a", DATA)
              ->NODE("enter_a", enter)
              ->NODE("merge_a", MERGE)
              ->NODE("switch_a", SWITCH)
              ->NODE("exit_a", EXIT)
              ->EDGE(0, 0)
              ->NODE("netoutput"));
    CHAIN(NODE("iteration_a", NEXTITERATION)->EDGE(0, 1)->NODE("merge_a"));
    CHAIN(NODE("loopcond")->EDGE(0, 1)->NODE("switch_a")->EDGE(1, 0)->NODE("mul", MUL));
    CHAIN(NODE("const_2", const_op)->EDGE(0, 1)->NODE("mul")->EDGE(0, 0)->NODE("iteration_a"));
  };
  return ToGeGraph(g1);
}
}  // namespace
class FrameworkTest : public testing::Test {
 protected:
  GeRunningEnvFaker ge_env;
  void SetUp() { ge_env.InstallDefault(); }
  void TearDown() {}
};

///     data   data
///       \    /
///        add
TEST_F(FrameworkTest, test_framework_add) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", DATA)->NODE("add", ADD));
    CHAIN(NODE("data2", DATA)->NODE("add"));
  };

  map<AscendString, AscendString> options;
  Session session(options);
  session.AddGraph(1, ToGeGraph(g1), options);
  std::vector<InputTensorInfo> inputs;
  auto ret = session.BuildGraph(1, inputs);

  EXPECT_EQ(ret, SUCCESS);
  CHECK_GRAPH(PreRunAfterBuild) {
    ASSERT_EQ(graph->GetName(), "g1_1");
    ASSERT_EQ(graph->GetAllNodesSize(), 4);
  };
}

/** data a = 2;
 *  for(int i =0; i<5; ++i){
 *    a=a * 2;
 * }
 *  return a;
 *                     ----------------------------------------------|
 *                    |  const(5)             exit  const(1)         |
 *                    |     \                  /       \             |
 *   data(i)--Enter--merge--less--loopcond--switch-----add-----nextiteration
 *                       \________________\___/
 *                                   ------\------------------------|
 *                                  |       \        const(2)       |
 *                                  |        \         \            |
 *                 data(a)--Enter--merge--switch------mul-----nextiteration
 *                                            \
 *                                           exit
 *                                             \
 *                                           netoutput
 *
 **/
TEST_F(FrameworkTest, test_framework_v1_control_flow) {
  // build graph
  Graph graph = BuildV1ControlFlowGraph();
  // new session & add graph
  map<AscendString, AscendString> options;
  Session session(options);
  auto ret = session.AddGraph(2, graph, options);
  EXPECT_EQ(ret, SUCCESS);
  // build input tensor
  std::vector<InputTensorInfo> inputs;
  // build_graph through session
  ret = session.BuildGraph(2, inputs);
  EXPECT_EQ(ret, SUCCESS);
  // check result
}
