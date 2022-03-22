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

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "runtime/mem.h"
#include "common/util.h"
#include "omg/omg_inner_types.h"

#define private public
#define protected public
#include "executor/ge_executor.h"

#include "common/auth/file_saver.h"
#include "common/debug/log.h"
#include "common/properties_manager.h"
#include "common/types.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/davinci_model.h"
#include "hybrid/hybrid_davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/task_info/kernel_task_info.h"
#include "graph/load/model_manager/task_info/kernel_ex_task_info.h"
#include "graph/execute/graph_execute.h"
#include "ge/common/dump/dump_properties.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/utils/graph_utils.h"
#include "proto/ge_ir.pb.h"
#include "graph/manager/graph_var_manager.h"
#undef private
#undef protected

using namespace std;
namespace ge {
class UtestGeExecutor : public testing::Test {
 protected:
  static void InitModelDefault(ge::Model &model) {
    ge::AttrUtils::SetInt(&model, ATTR_MODEL_MEMORY_SIZE, 0);
    ge::AttrUtils::SetInt(&model, ATTR_MODEL_WEIGHT_SIZE, 0);
    ge::AttrUtils::SetInt(&model, ATTR_MODEL_STREAM_NUM, 0);
    ge::AttrUtils::SetInt(&model, ATTR_MODEL_EVENT_NUM, 0);
    ge::AttrUtils::SetStr(&model, ATTR_MODEL_TARGET_TYPE, "MINI");  // domi::MINI

    auto compute_graph = std::make_shared<ge::ComputeGraph>("graph");
    auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);
  }

  void SetUp() {
    unsetenv("FMK_SYSMODE");
    unsetenv("FMK_DUMP_PATH");
    unsetenv("FMK_USE_FUSION");
    unsetenv("DAVINCI_TIMESTAT_ENABLE");
  }
};

class DModelListener : public ge::ModelListener {
 public:
  DModelListener() {
  };
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t resultCode,
                       std::vector<ge::Tensor> &outputs) {
    GELOGI("In Call back. OnComputeDone");
    return SUCCESS;
  }
};

shared_ptr<ge::ModelListener> g_label_call_back(new DModelListener());

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  ge::AttrUtils::SetFloat(op_desc, ge::ATTR_NAME_ALPHA, 0);
  ge::AttrUtils::SetFloat(op_desc, ge::ATTR_NAME_BETA, 0);

  op_desc->SetWorkspace({});
  ;
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({});
  op_desc->SetOutputOffset({});

  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_WEIGHT_NAME, {});
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_MODE, 0);
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_PAD_MODE, 0);
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_DATA_MODE, 0);
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_CEIL_MODE, 0);
  ge::AttrUtils::SetInt(op_desc, ge::POOLING_ATTR_NAN_OPT, 0);
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_WINDOW, {});
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_PAD, {});
  ge::AttrUtils::SetListInt(op_desc, ge::POOLING_ATTR_STRIDE, {});
  ge::AttrUtils::SetListInt(op_desc, ge::ATTR_NAME_ACTIVE_STREAM_LIST, {1, 1});
  ge::AttrUtils::SetInt(op_desc, ge::ATTR_NAME_STREAM_SWITCH_COND, 0);
  return op_desc;
}

TEST_F(UtestGeExecutor, load_data_from_file) {
  GeExecutor ge_executor;
  ge_executor.isInit_ = true;

  string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);

  ModelData model_data;
  EXPECT_EQ(ge_executor.LoadDataFromFile(test_smap, model_data), SUCCESS);

  EXPECT_NE(model_data.model_data, nullptr);
  delete[] static_cast<char *>(model_data.model_data);
  model_data.model_data = nullptr;

  ge_executor.isInit_ = false;
}

/*
TEST_F(UtestGeExecutor, fail_UnloadModel_model_manager_stop_unload_error) {
  uint32_t model_id = 1;
  ge::GeExecutor ge_executor;
  ge_executor.isInit_ = true;
  ge::Status ret = ge_executor.UnloadModel(model_id);
  EXPECT_EQ(ge::PARAM_INVALID, ret);

  ge_executor.isInit_ = false;
  ret = ge_executor.UnloadModel(model_id);
  EXPECT_EQ(ge::GE_EXEC_NOT_INIT, ret);
}

TEST_F(UtestGeExecutor, fail_CommandHandle_model_manager_HandleCommand_error) {
  ge::Command cmd;
  ge::GeExecutor ge_executor;
  ge::Status ret = ge_executor.CommandHandle(cmd);
  EXPECT_EQ(ge::PARAM_INVALID, ret);
}
*/
TEST_F(UtestGeExecutor, InitFeatureMapAndP2PMem_failed) {
  DavinciModel model(0, g_label_call_back);
  model.is_feature_map_mem_has_inited_ = true;
  EXPECT_EQ(model.InitFeatureMapAndP2PMem(nullptr, 0), PARAM_INVALID);
}

TEST_F(UtestGeExecutor, kernel_InitDumpArgs) {
  DavinciModel model(0, g_label_call_back);
  model.om_name_ = "testom";
  model.name_ = "test";
  OpDescPtr op_desc = CreateOpDesc("test", "test");

  std::map<std::string, std::set<std::string>> model_dump_properties_map;
  std::set<std::string> s;
  model_dump_properties_map[DUMP_ALL_MODEL] = s;
  DumpProperties dp;
  dp.model_dump_properties_map_ = model_dump_properties_map;
  model.SetDumpProperties(dp);

  KernelTaskInfo kernel_task_info;
  kernel_task_info.davinci_model_ = &model;
  kernel_task_info.op_desc_ = op_desc;
  kernel_task_info.InitDumpArgs(0);
}

TEST_F(UtestGeExecutor, kernel_ex_InitDumpArgs) {
  DavinciModel model(0, g_label_call_back);
  model.om_name_ = "testom";
  model.name_ = "test";
  OpDescPtr op_desc = CreateOpDesc("test", "test");

  std::map<std::string, std::set<std::string>> model_dump_properties_map;
  std::set<std::string> s;
  model_dump_properties_map[DUMP_ALL_MODEL] = s;
  DumpProperties dp;
  dp.model_dump_properties_map_ = model_dump_properties_map;
  model.SetDumpProperties(dp);

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  kernel_ex_task_info.InitDumpArgs(nullptr, op_desc);
}

TEST_F(UtestGeExecutor, kernel_ex_InitDumpFlag) {
  DavinciModel model(0, g_label_call_back);
  model.om_name_ = "testom";
  model.name_ = "test";
  OpDescPtr op_desc = CreateOpDesc("test", "test");

  std::map<std::string, std::set<std::string>> model_dump_properties_map;
  std::set<std::string> s;
  model_dump_properties_map[DUMP_ALL_MODEL] = s;
  DumpProperties dp;
  dp.model_dump_properties_map_ = model_dump_properties_map;
  model.SetDumpProperties(dp);

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  kernel_ex_task_info.InitDumpFlag(op_desc);
}

TEST_F(UtestGeExecutor, execute_graph_with_stream) {
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  map<string, string> options;
  options[GRAPH_MEMORY_MAX_SIZE] = "1048576";
  VarManager::Instance(0)->SetMemoryMallocSize(options);

  DavinciModel model(0, nullptr);
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");

  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("square", "Square");
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(RT_MODEL_TASK_KERNEL);
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_op_index(op_desc->GetId());
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({5120});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 2

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(RT_MODEL_TASK_MEMCPY_ASYNC);
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(5120);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({5120});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  EXPECT_EQ(model.Assign(ge_model), SUCCESS);
  EXPECT_EQ(model.Init(), SUCCESS);

  EXPECT_EQ(model.input_addrs_list_.size(), 1);
  EXPECT_EQ(model.output_addrs_list_.size(), 1);
  EXPECT_EQ(model.task_list_.size(), 2);

  OutputData output_data;
  vector<Tensor> outputs;
  EXPECT_EQ(model.GenOutputTensorInfo(&output_data, outputs), SUCCESS);

  GraphExecutor graph_executer;
  graph_executer.init_flag_ = true;
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  std::vector<GeTensor> input_tensor;
  std::vector<GeTensor> output_tensor;
  std::vector<InputOutputDescInfo> output_desc;
  InputOutputDescInfo desc0;
  output_desc.push_back(desc0);
  graph_executer.ExecuteGraphWithStream(0, nullptr, ge_root_model, input_tensor, output_tensor);
}

TEST_F(UtestGeExecutor, get_op_attr) {
  shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(1, g_label_call_back);
  model->SetId(1);
  model->om_name_ = "testom";
  model->name_ = "test";

  shared_ptr<hybrid::HybridDavinciModel> hybrid_model = MakeShared<hybrid::HybridDavinciModel>();
  model->SetId(2);
  model->om_name_ = "testom_hybrid";
  model->name_ = "test_hybrid";

  std::shared_ptr<ModelManager> model_manager = ModelManager::GetInstance();
  model_manager->InsertModel(1, model);
  model_manager->InsertModel(2, hybrid_model);

  OpDescPtr op_desc = CreateOpDesc("test", "test");
  std::vector<std::string> value{"test"};
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, value);

  model->SaveSpecifyAttrValues(op_desc);

  GeExecutor ge_executor;
  GeExecutor::isInit_ = true;
  std::string attr_value;
  auto ret = ge_executor.GetOpAttr(1, "test", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, attr_value);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(attr_value, "[4]test");
  ret = ge_executor.GetOpAttr(2, "test", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, attr_value);
  EXPECT_EQ(ret, PARAM_INVALID);
  ret = ge_executor.GetOpAttr(3, "test", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, attr_value);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}
}