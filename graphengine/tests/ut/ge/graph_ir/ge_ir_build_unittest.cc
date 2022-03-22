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
#include <stdio.h>
#include <gtest/gtest.h>
#include "ir_build/option_utils.h"
#include "graph/testcase/ge_graph/graph_builder_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "ge/ge_ir_build.h"
#include "graph/ops_stub.h"
#include "ge/ir_build/attr_options/attr_options.h"
#define protected public
#define private public

#undef private
#undef protected

const string DATA = "Data";
const string AddNYes = "AddNYes";
const string NETOUTPUT = "NetOutput";

using namespace ge;
class UtestIrCommon : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class UtestIrBuild : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ge::OpDescPtr CreateOpDesc(const std::string &name, const std::string &type) {
  OpDescPtr op_desc = std::make_shared<ge::OpDesc>(name, type);
  ge::GeTensorDesc ge_tensor_desc;
  op_desc->AddInputDesc("input", ge_tensor_desc);
  op_desc->AddOutputDesc("output", ge_tensor_desc);

  return op_desc;
}

static ComputeGraphPtr BuildComputeGraph() {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto addn1 = builder.AddNode("addn1", AddNYes, 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(data2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0,netoutput, 0);

  return builder.GetGraph();
}

static ComputeGraphPtr BuildComputeGraph1() {
  auto builder = ut::GraphBuilder("test");
  auto data1 = builder.AddNode("input1", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {1, 2, 3});
  auto data2 = builder.AddNode("input2", DATA, 1, 1, FORMAT_NCHW, DT_FLOAT, {4, 10});
  auto addn1 = builder.AddNode("addn1", AddNYes, 2, 1);
  auto node1 = builder.AddNode("addd", "Mul", 2, 1);
  auto node2 = builder.AddNode("ffm", "FrameworkOp", 2, 1);
  auto netoutput = builder.AddNode("netoutput", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data1, 0, addn1, 0);
  builder.AddDataEdge(data2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0,netoutput, 0);

  return builder.GetGraph();
}

// data not set attr index;
// but becasue of op proto, register attr index. so all data index is zero;
static Graph BuildIrGraph() {
  auto data1 = op::Data("data1");
  auto data2 = op::Data("data2");
  auto data3 = op::Data("data3");
  std::vector<Operator> inputs {data1, data2, data3};
  std::vector<Operator> outputs;

  Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

// data set attr index, but is not valid
static Graph BuildIrGraph1() {
  auto data1 = op::Data("data1").set_attr_index(0);
  auto data2 = op::Data("data2").set_attr_index(1);
  auto data3 = op::Data("data3");
  auto data4 = op::Data("Test");
  std::vector<Operator> inputs {data1, data2, data3, data4};
  std::vector<Operator> outputs;

  Graph graph("test_graph");
  graph.AddNodeByOp(Operator("gg", "Mul"));
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

// data set attr index, but is not valid
static Graph BuildIrGraph2() {
  auto data1 = op::Data("data1").set_attr_index(0);
  auto data2 = op::Data("data2");
  auto data3 = op::Data("data3").set_attr_index(2);
  std::vector<Operator> inputs {data1, data2, data3};
  std::vector<Operator> outputs;

  Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

// data set attr index
static Graph BuildIrGraph3() {
  auto data1 = op::Data("data1").set_attr_index(0);
  auto data2 = op::Data("data2").set_attr_index(1);
  auto data3 = op::Data("data3").set_attr_index(2);
  std::vector<Operator> inputs {data1, data2, data3};
  std::vector<Operator> outputs;

  Graph graph("test_graph");
  graph.SetInputs(inputs).SetOutputs(outputs);
  return graph;
}

TEST(UtestIrCommon, update_data_op_shape) {
  ge::OpDescPtr op_desc = CreateOpDesc("Data", "Data");
  map<string, vector<int64_t>> shape_map;
  shape_map["Data"] = {{1,2}};

  Status ret = UpdateDataOpShape(op_desc, shape_map);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, update_data_op_shape_range) {
  ge::OpDescPtr op_desc = CreateOpDesc("Data", "Data");
  std::vector<std::vector<std::pair<int64_t, int64_t>>> index_shape_range_map;

  std::pair<int64_t, int64_t> range_pair(1, 2);
  vector<pair<int64_t, int64_t>> range_pair_tmp = { range_pair };

  index_shape_range_map.push_back(range_pair_tmp);

  AttrUtils::SetInt(op_desc, ATTR_NAME_INDEX, 0);
  Status ret = UpdateDataOpShapeRange(op_desc, index_shape_range_map);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, update_dynamic_shape_range_success) {
  ComputeGraphPtr graph = BuildComputeGraph();
  std::string input_shape_range = "input1:[1, 2~3, -1];input2:[3~5, 10]";

  Status ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, update_dynamic_shape_range_failed) {
  ComputeGraphPtr graph = BuildComputeGraph();
  // 1
  std::string input_shape_range = "input1;[1, 2~3, -1]";
  Status ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  // 2
  input_shape_range = "input1:[1, 2~3, -1)";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  //3
  input_shape_range = "input1:[1, 3~2, -1];input2:[3~5, 10]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::FAILED);

  //4
  input_shape_range = "input1:[1, 2~-3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  //5
  input_shape_range = "input:[1, 2~3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);

  //6
  input_shape_range = "addn1:[1, 2~3, -1]";
  ret = UpdateDynamicInputShapeRange(graph, input_shape_range);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_dynamic_image_size_fail) {
  map<string, vector<int64_t>> shape_map;
  shape_map["input1"] = {8, 3, -1, -1};
  string input_format = "NCHW";
  string dynamic_image_size = "@64,64;128,128;";

  bool ret = CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, false);
}

TEST(UtestIrCommon, check_input_format_failed) {
  std::string format = "invalid";
  Status ret = CheckInputFormat(format);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_dynamic_batch_size_input_shape_succ) {
  map<string, vector<int64_t>> shape_map;
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {-1, 2, 3}));
  std::string dynamic_batch_size = "11";

  bool ret = CheckDynamicBatchSizeInputShapeValid(shape_map, dynamic_batch_size);
  EXPECT_EQ(ret, true);
}

TEST(UtestIrCommon, check_dynamic_images_size_input_shape_succ) {
  map<string, vector<int64_t>> shape_map;
  shape_map.insert(std::pair<string, vector<int64_t>>("data", {4, -1, -1, 5}));
  std::string input_format = "NCHW";
  std::string dynamic_image_size = "4,5";

  Status ret = CheckDynamicImagesizeInputShapeValid(shape_map, input_format, dynamic_image_size);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, check_dynamic_input_param_succ) {
  string dynamic_batch_size = "1";
  string dynamic_image_size;
  string dynamic_dims;
  string input_shape = "data:-1,3,244,244";
  string input_shape_range;
  string input_format = "NCHW";
  bool is_dynamic_input = false;

  Status ret = CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
                                           input_shape, input_shape_range, input_format,is_dynamic_input);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST(UtestIrCommon, check_dynamic_input_param_failed) {
  string dynamic_batch_size = "1";
  string dynamic_image_size;
  string dynamic_dims;
  string input_shape = "data:1,3,244,244";
  string input_shape_range;
  string input_format = "NCHW";
  bool is_dynamic_input = false;

  Status ret = CheckDynamicInputParamValid(dynamic_batch_size, dynamic_image_size, dynamic_dims,
                                           input_shape, input_shape_range, input_format,is_dynamic_input);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_modify_mixlist_param) {
  std::string precision_mode = "allow_mix_precision";
  std::string modify_mixlist = "/mixlist.json";
  Status ret = CheckModifyMixlistParamValid(precision_mode, modify_mixlist);
  EXPECT_EQ(ret, ge::SUCCESS);

  precision_mode = "";
  ret = CheckModifyMixlistParamValid(precision_mode, modify_mixlist);
  EXPECT_EQ(ret, ge::PARAM_INVALID);
}

TEST(UtestIrCommon, check_compress_weight) {
  std::string enable_compress_weight = "true";
  std::string compress_weight_conf="./";
  Status ret = CheckCompressWeightParamValid(enable_compress_weight, compress_weight_conf);
  EXPECT_EQ(ret, PARAM_INVALID);

  enable_compress_weight = "yes";
  compress_weight_conf = "./";
  ret = CheckCompressWeightParamValid(enable_compress_weight, compress_weight_conf);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST(UtestIrCommon, check_param_failed) {
  std::string param_invalid = "invalid";

  Status ret = CheckOutputTypeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckBufferOptimizeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckKeepTypeParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckInsertOpConfParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckDisableReuseMemoryParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckEnableSingleStreamParamValid(param_invalid);
  EXPECT_EQ(ret, PARAM_INVALID);

  std::string optypelist_for_implmode;
  std::string op_select_implmode = "1";
  ret = CheckImplmodeParamValid(optypelist_for_implmode, op_select_implmode);
  EXPECT_EQ(ret, PARAM_INVALID);

  ret = CheckLogParamValidAndSetLogLevel(param_invalid);
}

// Get attr index failed, when set input shape range
TEST(UtestIrBuild, check_data_op_attr_index_invalid_0) {
  ComputeGraphPtr compute_graph = BuildComputeGraph();
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  const map<string, string> build_options = {
    {"input_shape_range", "[1, 2~3, -1],[4~5, 3~5, 10],[1, 2~3, -1]"}
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

// not set attr index, when set input shape range
TEST(UtestIrBuild, check_data_op_attr_index_invalid_1) {
  Graph graph = BuildIrGraph();
  const map<string, string> build_options = {
    {"input_shape_range", "[1, 2~3, -1],[4~5, 3~5, 10],[1, 2~3, -1]"}
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

// set attr index, but not valid, when set input shape range
TEST(UtestIrBuild, check_data_op_attr_index_invalid_2) {
  Graph graph = BuildIrGraph1();
  const map<string, string> build_options = {
    {"input_shape_range", "[1, 2~3, -1],[4~5, 3~5, 10],[1, 2~3, -1]"}
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_FAILED);

  Graph graph2 = BuildIrGraph2();
  ret = aclgrphBuildModel(graph2, build_options, model);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

// set attr index valid, when set input shape range
// only check data op attr index valid func.
TEST(UtestIrBuild, check_data_op_attr_index_valid) {
  Graph graph = BuildIrGraph3();
  const map<string, string> build_options = {
    {"input_shape_range", "[1, 2~3, -1],[4~5, 3~5, 10],[1, 2~3, -1]"}
  };
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, ge::FAILED);
}

// set attr index invalid, when not set input shape range
// only check data op attr index valid func.
TEST(UtestIrBuild, check_data_attr_index_succ_no_input_range) {
  Graph graph = BuildIrGraph1();
  const map<string, string> build_options;
  ModelBufferData model;
  graphStatus ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST(UtestIrBuild, check_modify_mixlist_param) {
  Graph graph = BuildIrGraph1();
  const std::map<std::string, std::string> build_options = {
    {"ge.exec.modify_mixlist", "/modify.json"}
  };
  ModelBufferData model;

  auto ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, check_op_precision_mode_param) {
  Graph graph = BuildIrGraph1();
  const std::map<std::string, std::string> build_options = {
    {"ge.exec.op_precision_mode", "./op_precision_mode.ini"}
  };
  ModelBufferData model;

  auto ret = aclgrphBuildModel(graph, build_options, model);
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}

TEST(UtestIrBuild, check_build_model_and_build_step) {
  Graph graph_1 = BuildIrGraph1();
  const std::map<std::string, std::string> build_options_1 = {
    {"ge.buildMode", "xxx"}
  };
  ModelBufferData model_1;
  auto ret_1 = aclgrphBuildModel(graph_1, build_options_1, model_1);
  EXPECT_NE(ret_1, GRAPH_SUCCESS);

  Graph graph_2 = BuildIrGraph1();
  const std::map<std::string, std::string> build_options_2 = {
    {"ge.buildStep", "xxx"}
  };
  ModelBufferData model_2;
  auto ret_2 = aclgrphBuildModel(graph_2, build_options_2, model_2);
  EXPECT_NE(ret_2, GRAPH_SUCCESS);

  Graph graph_3 = BuildIrGraph1();
  const std::map<std::string, std::string> build_options_3 = {
    {"ge.buildMode", "tuning"}
  };
  ModelBufferData model_3;
  auto ret_3 = aclgrphBuildModel(graph_3, build_options_3, model_3);
  EXPECT_NE(ret_3, GRAPH_SUCCESS);
}

TEST(UtestIrBuild, atc_cfg_optype_param) {
  ComputeGraphPtr graph = BuildComputeGraph1();
  FILE *fp = fopen("./keep.txt", "w+");
  if (fp) {
    fprintf(fp, "Test\n");
    fprintf(fp, "OpType::Mul\n");
    fprintf(fp, "Optype::Sub\n");
    fclose(fp);
  }
  auto ret = KeepDtypeFunc(graph, "./keep.txt");
  (void)remove("./keep.txt");
  EXPECT_EQ(ret, GRAPH_PARAM_INVALID);
}