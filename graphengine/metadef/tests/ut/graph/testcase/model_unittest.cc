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

#include <stdio.h>
#include <gtest/gtest.h>
#include <iostream>
#include "test_structs.h"
#include "func_counter.h"
#include "graph/buffer.h"
#include "graph/attr_store.h"
#include "graph/model.h"
#include "graph/node.h"
#include "graph_builder_utils.h"
#include "graph/utils/graph_utils.h"


namespace ge {
namespace {
class SubModel : public Model
{
public:

  SubModel();
  SubModel(const std::string &name, const std::string &custom_version);

  virtual ~SubModel();

};

SubModel::SubModel(){}
SubModel::SubModel(const std::string &name, const std::string &custom_version):Model(name,custom_version){}

SubModel::~SubModel() = default;

}

static Graph BuildGraph() {
  ge::OpDescPtr add_op(new ge::OpDesc("add1", "Add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  return graph;
}

class ModelUt : public testing::Test {};

TEST_F(ModelUt, SetGet) {
  auto md = SubModel();
  auto md2 = SubModel("md2", "test");
  EXPECT_EQ(md.GetName(),"");
  md.SetName("tt");
  EXPECT_EQ(md.GetName(),"tt");
  EXPECT_EQ(md2.GetName(),"md2");
  md2.SetName("md2tt");
  EXPECT_EQ(md2.GetName(),"md2tt");
  EXPECT_EQ(md.GetVersion(),0);
  EXPECT_EQ(md2.GetVersion(),0);
  EXPECT_EQ(md2.GetPlatformVersion(),"test");

  auto graph = BuildGraph();
  EXPECT_EQ(graph.IsValid(),true);
  md2.SetGraph(graph);
  auto g = md2.GetGraph();
  EXPECT_NE(&g, nullptr);
  Buffer buf = Buffer(1024);
  EXPECT_EQ(buf.GetSize(),1024);
  EXPECT_EQ(md2.IsValid(),true);
  ProtoAttrMap attr = AttrStore::Create(512);
  AttrId id = 1;
  int val = 100;
  attr.Set<int>(id, val);
  const int* v = attr.Get<int>(id);
  EXPECT_EQ(*v,val);
  md2.SetAttr(attr);
  EXPECT_EQ(md2.Save(buf,true), GRAPH_SUCCESS);

}

TEST_F(ModelUt, Load) {
  auto md = SubModel("md2", "test");
  auto graph = BuildGraph();
  md.SetGraph(graph);
  uint8_t b[5];
  memset(b,1,5);
  EXPECT_EQ(md.Load((const uint8_t*)b, 5, md),GRAPH_FAILED);

  const char* msg = "package lm;\nmessage helloworld{\nrequired int32     id = 1;\nrequired string    str = 2;\noptional int32     opt = 3;}";
  FILE *fp = NULL;
  fp = fopen("/tmp/hw.proto","w");
  fputs(msg, fp);
  fclose(fp);
  EXPECT_EQ(md.LoadFromFile("/tmp/hw.proto"),GRAPH_FAILED);

}

TEST_F(ModelUt, Save) {
  auto md = SubModel("md2", "test");
  auto graph = BuildGraph();
  md.SetGraph(graph);
  //EXPECT_EQ(md.SaveToFile("/tmp/hw2.proto"),GRAPH_SUCCESS);
}

}  // namespace ge
