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

#include <bits/stdc++.h>
#include <dirent.h>
#include <gtest/gtest.h>
#include <fstream>
#include <map>
#include <string>

#include "graph/load/model_manager/davinci_model.h"

#define protected public
#define private public
#include "common/profiling/profiling_manager.h"
#include "graph/ge_local_context.h"
#include "inc/framework/common/profiling/ge_profiling.h"
#include "graph/manager/graph_manager.h"
#include "graph/ops_stub.h"
#include "inc/framework/omg/omg_inner_types.h"
#undef protected
#undef private

using namespace ge;
using namespace std;

class UtestGeProfilinganager : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

int32_t ReporterCallback(uint32_t moduleId, uint32_t type, void *data, uint32_t len) {
  return -1;
}

void CreateGraph(Graph &graph) {
  TensorDesc desc(ge::Shape({1, 3, 224, 224}));
  uint32_t size = desc.GetShape().GetShapeSize();
  desc.SetSize(size);
  auto data = op::Data("Data").set_attr_index(0);
  data.update_input_desc_data(desc);
  data.update_output_desc_out(desc);

  auto flatten = op::Flatten("Flatten").set_input_x(data, data.name_out_out());

  std::vector<Operator> inputs{data};
  std::vector<Operator> outputs{flatten};
  std::vector<Operator> targets{flatten};
  // Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs).SetTargets(targets);
}

TEST_F(UtestGeProfilinganager, init_success) {
  setenv("PROFILING_MODE", "true", true);
  Options options;
  options.device_id = 0;
  options.job_id = "0";
  options.profiling_mode = "1";
  options.profiling_options = R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})";


  struct MsprofGeOptions prof_conf = {{ 0 }};

  Status ret = ProfilingManager::Instance().InitFromOptions(options, prof_conf);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, ParseOptions) {
  setenv("PROFILING_MODE", "true", true);
  Options options;
  options.device_id = 0;
  options.job_id = "0";
  options.profiling_mode = "1";
  options.profiling_options = R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})";

  struct MsprofGeOptions prof_conf = {{ 0 }};
  Status ret = ProfilingManager::Instance().ParseOptions(options.profiling_options);
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(ProfilingManager::Instance().is_training_trace_, true);
  EXPECT_EQ(ProfilingManager::Instance().fp_point_, "Data_0");
  EXPECT_EQ(ProfilingManager::Instance().bp_point_, "addn");
}

TEST_F(UtestGeProfilinganager, plungin_init_) {
  ProfilingManager::Instance().prof_cb_.msprofReporterCallback = ReporterCallback;

  Status ret = ProfilingManager::Instance().PluginInit();
  EXPECT_EQ(ret, INTERNAL_ERROR);
  ProfilingManager::Instance().prof_cb_.msprofReporterCallback = nullptr;
}

TEST_F(UtestGeProfilinganager, report_data_) {
  std::string data = "ge is better than tensorflow.";
  std::string tag_name = "fmk";
  ProfilingManager::Instance().ReportData(0, data, tag_name);
}

TEST_F(UtestGeProfilinganager, get_fp_bp_point_) {
  map<std::string, string> options_map = {
    {OPTION_EXEC_PROFILING_OPTIONS,
    R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","fp_point":"Data_0","bp_point":"addn","ai_core_metrics":"ResourceConflictRatio"})"}};
  GEThreadLocalContext &context = GetThreadLocalContext();
  context.SetGraphOption(options_map);

  std::string fp_point;
  std::string bp_point;
  ProfilingManager::Instance().GetFpBpPoint(fp_point, bp_point);
  EXPECT_EQ(fp_point, "Data_0");
  EXPECT_EQ(bp_point, "addn");
}

TEST_F(UtestGeProfilinganager, get_fp_bp_point_empty) {
  // fp bp empty
  map<std::string, string> options_map = {
    { OPTION_EXEC_PROFILING_OPTIONS,
      R"({"result_path":"/data/profiling","training_trace":"on","task_trace":"on","aicpu_trace":"on","ai_core_metrics":"ResourceConflictRatio"})"}};
  GEThreadLocalContext &context = GetThreadLocalContext();
  context.SetGraphOption(options_map);
  std::string fp_point = "fp";
  std::string bp_point = "bp";
  ProfilingManager::Instance().bp_point_ = "";
  ProfilingManager::Instance().fp_point_ = "";
  ProfilingManager::Instance().GetFpBpPoint(fp_point, bp_point);
  EXPECT_EQ(fp_point, "");
  EXPECT_EQ(bp_point, "");
}

TEST_F(UtestGeProfilinganager, set_step_info_success) {
  uint64_t index_id = 0;
  auto stream = (rtStream_t)0x1;
  Status ret = ProfSetStepInfo(index_id, 0, stream);
  EXPECT_EQ(ret, ge::SUCCESS);
  ret = ProfSetStepInfo(index_id, 1, stream);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, set_step_info_failed) {
  uint64_t index_id = 0;
  auto stream = (rtStream_t)0x1;
  Status ret = ProfSetStepInfo(index_id, 1, stream);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(UtestGeProfilinganager, get_device_from_graph) {
  GraphId graph_id = 1;
  uint32_t device_id = 0;
  GraphManager graph_manager;
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(graph_id);
  graph_manager.AddGraphNode(graph_id, graph_node);
  graph_manager.SetAddGraphCondition(graph_id, 2);
  Graph graph("test_graph");
  CreateGraph(graph);
  std::map<std::string, std::string> options;
  OmgContext context;
  Status ret = graph_manager.AddGraph(graph_id, graph, options, context);
  EXPECT_EQ(ret, ge::SUCCESS);
  ret = ProfGetDeviceFormGraphId(graph_id, device_id);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, handle_subscribe_info) {
  ProfCommandHandleType prof_type = kProfCommandhandleModelSubscribe;
  ProfCommandHandleData prof_data;
  prof_data.profSwitch = 0;
  prof_data.modelId = 1;
  domi::GetContext().train_flag = true;
  auto prof_ptr = std::make_shared<ProfCommandHandleData>(prof_data);
  Status ret = ProfCommandHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, handle_unsubscribe_info) {
  ProfCommandHandleType prof_type = kProfCommandhandleModelUnsubscribe;
  ProfCommandHandleData prof_data;
  prof_data.profSwitch = 0;
  prof_data.modelId = 1;
  domi::GetContext().train_flag = true;
  auto &profiling_manager = ge::ProfilingManager::Instance();
  profiling_manager.SetSubscribeInfo(0, 1, true);
  auto prof_ptr = std::make_shared<ProfCommandHandleData>(prof_data);
  Status ret = ProfCommandHandle(prof_type, static_cast<void *>(prof_ptr.get()), sizeof(prof_data));
  profiling_manager.CleanSubscribeInfo();
}

TEST_F(UtestGeProfilinganager, set_subscribe_info) {
  auto &profiling_manager = ge::ProfilingManager::Instance();
  profiling_manager.SetSubscribeInfo(0, 1, true);
  const auto &subInfo = profiling_manager.GetSubscribeInfo();
  EXPECT_EQ(subInfo.prof_switch, 0);
  EXPECT_EQ(subInfo.graph_id, 1);
  EXPECT_EQ(subInfo.is_subscribe, true);
}

TEST_F(UtestGeProfilinganager, clean_subscribe_info) {
  auto &profiling_manager = ge::ProfilingManager::Instance();
  profiling_manager.CleanSubscribeInfo();
  const auto &subInfo = profiling_manager.GetSubscribeInfo();
  EXPECT_EQ(subInfo.prof_switch, 0);
  EXPECT_EQ(subInfo.graph_id, 0);
  EXPECT_EQ(subInfo.is_subscribe, false);
}

TEST_F(UtestGeProfilinganager, get_model_id_success) {
  auto &profiling_manager = ge::ProfilingManager::Instance();
  profiling_manager.SetGraphIdToModelMap(0, 1);
  uint32_t model_id = 0;
  Status ret = profiling_manager.GetModelIdFromGraph(0, model_id);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, get_model_id_failed) {
  auto &profiling_manager = ge::ProfilingManager::Instance();
  profiling_manager.SetGraphIdToModelMap(0, 1);
  uint32_t model_id = 0;
  Status ret = profiling_manager.GetModelIdFromGraph(10, model_id);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(UtestGeProfilinganager, get_device_id_success) {
  auto &profiling_manager = ge::ProfilingManager::Instance();
  profiling_manager.SetGraphIdToDeviceMap(0, 1);
  uint32_t device_id = 0;
  Status ret = profiling_manager.GetDeviceIdFromGraph(0, device_id);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UtestGeProfilinganager, get_device_id_failed) {
  auto &profiling_manager = ge::ProfilingManager::Instance();
  profiling_manager.SetGraphIdToDeviceMap(0, 1);
  uint32_t device_id = 0;
  Status ret = profiling_manager.GetDeviceIdFromGraph(10, device_id);
  EXPECT_EQ(ret, ge::FAILED);
}
