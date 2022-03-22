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

#include <vector>
#include <string>
#include <map>

#define protected public
#define private public
#include "common/ge/ge_util.h"
#include "proto/ge_ir.pb.h"
#include "inc/external/ge/ge_api.h"
#include "session/session_manager.h"

using namespace std;

namespace ge {
class UtestGeApi : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestGeApi, run_graph_with_stream) {
  vector<Tensor> inputs;
  vector<Tensor> outputs;
  std::map<std::string, std::string> options;
  Session session(options);
  auto ret = session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
  SessionManager session_manager;
  session_manager.init_flag_ = true;
  ret = session_manager.RunGraphWithStreamAsync(10, 10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
  InnerSession inner_session(1, options);
  inner_session.init_flag_ = true;
  ret = inner_session.RunGraphWithStreamAsync(10, nullptr, inputs, outputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, build_graph_success) {
  vector<Tensor> inputs;
  std::map<std::string, std::string> options;
  Session session(options);
  auto ret = session.BuildGraph(1, inputs);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, ge_initialize_modify_mixlist) {
  std::map<std::string, std::string> options = {
    {ge::MODIFY_MIXLIST, "/mixlist.json"}
  };
  auto ret = GEInitialize(options);
  ASSERT_NE(ret, SUCCESS);
}

TEST_F(UtestGeApi, ge_not_initialized) {
  EXPECT_EQ(GEFinalize(), SUCCESS);

  std::map<std::string, std::string> options;
  std::map<AscendString, AscendString> ascend_options;
  Session session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), FAILED);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), FAILED);

  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph), FAILED);
  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph, ascend_options), FAILED);

  vector<Tensor> inputs;
  vector<InputTensorInfo> tensors;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), FAILED);
  EXPECT_EQ(session.BuildGraph(graph_id, tensors), FAILED);

  vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), FAILED);

  vector<string> var_inputs;
  EXPECT_EQ(session.GetVariables(var_inputs, outputs), FAILED);

  vector<AscendString> var_names;
  EXPECT_EQ(session.GetVariables(var_names, outputs), FAILED);

  std::string key;
  pCallBackFunc ge_callback;
  EXPECT_EQ(session.RegisterCallBackFunc(key, ge_callback), FAILED);

  session::pCallBackFunc session_callback;
  EXPECT_EQ(session.RegisterCallBackFunc(key.c_str(), session_callback), FAILED);

  EXPECT_FALSE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), FAILED);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, ge_session_ascend_string) {
  std::map<AscendString, AscendString> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  Session session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  EXPECT_EQ(session.AddGraph(graph_id, GraphUtils::CreateGraphFromComputeGraph(compute_graph)), SUCCESS);

  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);

  EXPECT_EQ(GEFinalize(), SUCCESS);
}

TEST_F(UtestGeApi, ge_session_test) {
  std::map<std::string, std::string> options;
  EXPECT_EQ(GEInitialize(options), SUCCESS);

  std::map<AscendString, AscendString> ascend_options;
  Session session(options);

  GraphId graph_id = 1;
  const auto compute_graph = MakeShared<ComputeGraph>("test_graph");
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  EXPECT_EQ(session.AddGraph(graph_id, graph), SUCCESS);
  EXPECT_EQ(session.AddGraph(graph_id, graph, ascend_options), SUCCESS);

  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph), FAILED);
  EXPECT_EQ(session.AddGraphWithCopy(graph_id, graph, ascend_options), FAILED);

  vector<Tensor> inputs;
  vector<InputTensorInfo> tensors;
  EXPECT_EQ(session.BuildGraph(graph_id, inputs), FAILED);
  EXPECT_EQ(session.BuildGraph(graph_id, tensors), FAILED);

  vector<Tensor> outputs;
  EXPECT_EQ(session.RunGraph(graph_id, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphWithStreamAsync(graph_id, nullptr, inputs, outputs), FAILED);
  EXPECT_EQ(session.RunGraphAsync(graph_id, inputs, nullptr), SUCCESS); // Push to queue.

  vector<string> var_inputs;
  EXPECT_EQ(session.GetVariables(var_inputs, outputs), FAILED);

  vector<AscendString> var_names;
  EXPECT_EQ(session.GetVariables(var_names, outputs), FAILED);

  std::string key;
  pCallBackFunc ge_callback;
  EXPECT_EQ(session.RegisterCallBackFunc(key, ge_callback), SUCCESS);

  session::pCallBackFunc session_callback;
  EXPECT_EQ(session.RegisterCallBackFunc(key.c_str(), session_callback), SUCCESS);

  EXPECT_TRUE(session.IsGraphNeedRebuild(graph_id));

  EXPECT_EQ(session.RemoveGraph(graph_id), SUCCESS);
  EXPECT_EQ(GEFinalize(), SUCCESS);
}

}  // namespace ge
