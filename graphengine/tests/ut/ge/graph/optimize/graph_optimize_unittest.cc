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
#include <memory>
#include <iostream>
#define protected public
#define private public
#include "graph/optimize/graph_optimize.h"
#include "init/gelib.h"
#include "ge/ge_api.h"
#undef private
#undef protected

using namespace std;
using namespace testing;
using namespace ge;

namespace {
const char *const kVectorCore = "VectorCore";
const char *const kAicoreEngine = "AIcoreEngine";
void CreateEngineConfigJson(string &dir_path, string &file_path) {
  GELOGI("Begin to create engine config json file.");
  string base_path = PluginManager::GetPath();
  GELOGI("Base path is %s.", base_path.c_str());
  dir_path = base_path.substr(0, base_path.rfind('/') + 1) + "plugin/nnengine/ge_config";
  string cmd = "mkdir -p " + dir_path;
  system(cmd.c_str());
  file_path = dir_path + "/engine_conf.json";
  GELOGI("Begin to write into the config file: %s.", file_path.c_str());
  ofstream ofs(file_path, ios::out);
  EXPECT_EQ(!ofs, false);
  ofs << "{\n"
         "  \"schedule_units\" : [ {\n"
         "    \"id\" : \"TS_1\",\n"
         "    \"name\" : \"1980_hwts\",\n"
         "    \"ex_attrs\" : \"\",\n"
         "    \"cal_engines\" : [\n"
         "      {\"id\" : \"DNN_VM_GE_LOCAL\", \"name\" : \"GE_LOCAL\", \"independent\" : false, \"attch\" : true, \"skip_assign_stream\" : true },\n"
         "      {\"id\" : \"AIcoreEngine\", \"name\" : \"AICORE\", \"independent\" : false, \"attch\" : false, \"skip_assign_stream\" : false}\n"
         "    ]\n"
         "  } ]\n"
         "}";
  ofs.close();
  GELOGI("Json config file %s has been written.", file_path.c_str());
}

void DeleteFile(const string &file_name) {
 auto ret = remove(file_name.c_str());
 if (ret == 0) {
   GELOGI("Delete file successfully, file:%s.", file_name.c_str());
 }
}
}
class UtestGraphOptimizeTest : public testing::Test {
 protected:
  void SetUp() {
    CreateEngineConfigJson(config_dir_, config_file_);
  }

  void TearDown() {
    DeleteFile(config_file_);
    DeleteFile(config_dir_);
  }

 private:
  string config_dir_;
  string config_file_;
};

class TestGraphOptimizerSuccess : public GraphOptimizer {
 public:
  ~TestGraphOptimizerSuccess() override { Finalize(); }
  Status Initialize(const map<string, string> &options) override { return SUCCESS; }
  Status Finalize() override { return SUCCESS; }
  Status OptimizeGraphPrepare(ComputeGraph& graph) override { return SUCCESS; }
  Status OptimizeGraphBeforeBuild(ComputeGraph& graph) override { return SUCCESS; }
  Status OptimizeOriginalGraph(ComputeGraph &graph) override { return SUCCESS; }
  Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &graph) override { return SUCCESS; }
  Status OptimizeFusedGraph(ComputeGraph &graph) override { return SUCCESS; }
  Status OptimizeWholeGraph(ComputeGraph &graph) override { return SUCCESS; }
  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    attrs.engineName = "AIcoreEngine";
    attrs.scope = OPTIMIZER_SCOPE::ENGINE;
    return SUCCESS;
  }
  Status OptimizeStreamGraph(ComputeGraph &graph, const RunContext &context) override { return SUCCESS; }
  Status OptimizeFusedGraphAfterGraphSlice(ComputeGraph &graph) override { return SUCCESS; }
  Status OptimizeAfterStage1(ComputeGraph &graph) override { return SUCCESS; }
};

class TestGraphOptimizerFail : public GraphOptimizer {
 public:
  ~TestGraphOptimizerFail() override { Finalize(); }
  Status Initialize(const map<string, string> &options) override { return SUCCESS; }
  Status Finalize() override { return SUCCESS; }
  Status OptimizeGraphPrepare(ComputeGraph& graph) override { return FAILED; }
  Status OptimizeGraphBeforeBuild(ComputeGraph& graph) override { return FAILED; }
  Status OptimizeOriginalGraph(ComputeGraph &graph) override { return FAILED; }
  Status OptimizeOriginalGraphJudgeInsert(ComputeGraph &graph) override { return FAILED; }
  Status OptimizeFusedGraph(ComputeGraph &graph) override { return FAILED; }
  Status OptimizeWholeGraph(ComputeGraph &graph) override { return FAILED; }
  Status GetAttributes(GraphOptimizerAttribute &attrs) const override {
    attrs.engineName = "AIcoreEngine";
    attrs.scope = OPTIMIZER_SCOPE::ENGINE;
    return SUCCESS;
  }
  Status OptimizeStreamGraph(ComputeGraph &graph, const RunContext &context) override { return FAILED; }
  Status OptimizeFusedGraphAfterGraphSlice(ComputeGraph &graph) override { return FAILED; }
  Status OptimizeAfterStage1(ComputeGraph &graph) override { return FAILED; }
};

TEST_F(UtestGraphOptimizeTest, test_OptimizeAfterStage1_succ) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerSuccess>();
  instance_ptr->opsManager_.graph_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphOptimize base_optimize;
  ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  base_optimize.core_type_ = kVectorCore;
  ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, test_OptimizeAfterStage1_fail) {
  ComputeGraphPtr compute_graph = nullptr;
  GraphOptimize base_optimize;

  // 1. Input graph is nullptr.
  Status ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, PARAM_INVALID);

  // 2. GELib is not initialized.
  compute_graph = MakeShared<ComputeGraph>("test_graph");
  ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, GE_CLI_GE_NOT_INITIALIZED);

  // 3. The optimizer registered with the engine returned a failure.
  map<string, string> options;
  ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerFail>();
  instance_ptr->opsManager_.graph_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));
  ret = base_optimize.OptimizeAfterStage1(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, test_optimizers_succ) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerSuccess>();
  instance_ptr->opsManager_.graph_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphOptimize base_optimize;

  ret = base_optimize.OptimizeOriginalGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeOriginalGraphJudgeInsert(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeOriginalGraphForQuantize(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeGraphBeforeBuildForRts(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = base_optimize.OptimizeWholeGraph(compute_graph);
  EXPECT_EQ(ret, SUCCESS);

  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestGraphOptimizeTest, test_optimizers_fail) {
  map<string, string> options;
  Status ret = ge::GELib::Initialize(options);
  EXPECT_EQ(ret, SUCCESS);

  shared_ptr<GELib> instance_ptr = ge::GELib::GetInstance();
  EXPECT_NE(instance_ptr, nullptr);
  GraphOptimizerPtr graph_opt = MakeShared<TestGraphOptimizerFail>();
  instance_ptr->opsManager_.graph_optimizers_by_priority_.push_back(make_pair("AIcoreEngine", graph_opt));

  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test_graph");
  GraphOptimize base_optimize;

  ret = base_optimize.OptimizeOriginalGraph(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = base_optimize.OptimizeOriginalGraphJudgeInsert(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = base_optimize.OptimizeOriginalGraphForQuantize(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = base_optimize.OptimizeGraphBeforeBuildForRts(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = base_optimize.OptimizeWholeGraph(compute_graph);
  EXPECT_EQ(ret, FAILED);

  ret = instance_ptr->Finalize();
  EXPECT_EQ(ret, SUCCESS);
}
