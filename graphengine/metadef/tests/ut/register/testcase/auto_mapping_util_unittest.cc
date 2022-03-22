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
#include <iostream>
#include "graph_builder_utils.h"
#include "external/register/register.h"
#include <google/protobuf/message.h>
#include "graph/debug/ge_util.h"
#include "graph/debug/ge_op_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "proto/tensorflow/attr_value.pb.h"
#include "proto/tensorflow/node_def.pb.h"
#include "register/auto_mapping_util.h"
#include "register/op_registry.h"
#include "graph/graph.h"
#include "graph/utils/attr_utils.h"
#define private public
#define protected public
#include "external/register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"
#undef private
#undef protected

using namespace ge;
class AutoMappingUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

void CreateTFGraphDef(domi::tensorflow::GraphDef &graph_def) {
  // 1. add node
  auto placeholder0 = graph_def.add_node();
  auto placeholder1 = graph_def.add_node();
  auto add0 = graph_def.add_node();
  auto add1 = graph_def.add_node();
  auto mul0 = graph_def.add_node();
  auto mul1 = graph_def.add_node();
  auto add2 = graph_def.add_node();
  auto retval0 = graph_def.add_node();
  auto retval1 = graph_def.add_node();

  // 2. set info
  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");

  add0->set_name("add0");
  add0->set_op("Add");
  add1->set_name("add1");
  add1->set_op("Add");
  add2->set_name("add2");
  add2->set_op("Add");

  mul0->set_name("mul0");
  mul0->set_op("Mul");
  mul1->set_name("mul1");
  mul1->set_op("Mul");

  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");

  // 3. add edges
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  mul0->add_input("placeholder0");
  mul0->add_input("placeholder1");

  mul1->add_input("placeholder0");
  mul1->add_input("add0");
  mul1->add_input("^mul0");

  add1->add_input("mul0");
  add1->add_input("placeholder1");

  add2->add_input("mul1");
  add2->add_input("mul0");

  retval0->add_input("add2:0");
  retval1->add_input("add1:0");
}

TEST_F(AutoMappingUtils, FindAttrValue) {
  const std::string attr_name = "";
  domi::tensorflow::AttrValue attr_num;
  domi::tensorflow::NodeDef node1;
  ge::AutoMappingUtil::FindAttrValue(&node1, attr_name, attr_num);

  domi::tensorflow::GraphDef graph_def;
  CreateTFGraphDef(graph_def);
  for (int32_t i = 0; i < graph_def.node_size(); i++) {
    const domi::tensorflow::NodeDef *node2 = graph_def.mutable_node(i);
    const std::string &node_name = node2->name();
    ge::AutoMappingUtil::FindAttrValue(node2, node_name, attr_num);
  }
}

TEST_F(AutoMappingUtils, ConvertShape) {
  domi::tensorflow::TensorShapeProto shape;
  vector<int64_t> shape_dims;

  shape.set_unknown_rank(true);
  ge::AutoMappingUtil::ConvertShape(shape, shape_dims);

  shape.set_unknown_rank(false);
  shape.add_dim();
  ge::AutoMappingUtil::ConvertShape(shape, shape_dims);
}

TEST_F(AutoMappingUtils, ConvertTensor) {
  ge::graphStatus ret;
  domi::tensorflow::TensorProto tensor;
  ge::GeTensorPtr weight;

  tensor.set_dtype(domi::tensorflow::DataType_INT_MAX_SENTINEL_DO_NOT_USE_);
  ret = ge::AutoMappingUtil::ConvertTensor(tensor, weight);
  EXPECT_EQ(ret, GRAPH_FAILED);

  tensor.set_dtype(domi::tensorflow::DT_UINT16_REF);
  ge::AutoMappingUtil::ConvertTensor(tensor, weight);
}

TEST_F(AutoMappingUtils, ConvertTensorList) {
  domi::tensorflow::AttrValue_ListValue list;
  std::vector<ge::GeTensorPtr> vec;

  list.add_tensor();
  ge::AutoMappingUtil::ConvertTensorList(list, vec);
}

TEST_F(AutoMappingUtils, ConvertFunc) {
  domi::tensorflow::NameAttrList tf_func;
  ge::NamedAttrs ge_func;

  tf_func.set_name("test_fun");
  ge::AutoMappingUtil::ConvertFunc(tf_func, ge_func);
}

TEST_F(AutoMappingUtils, ConvertDataTypeList) {
  domi::tensorflow::AttrValue_ListValue list;
  std::vector<ge::DataType> vec;

  list.add_type(domi::tensorflow::DT_INT16);
  ge::AutoMappingUtil::ConvertDataTypeList(list, vec);
}

TEST_F(AutoMappingUtils, ConvertShapeList) {
  domi::tensorflow::AttrValue_ListValue list;
  std::vector<vector<int64_t>> vec;

  list.add_shape();
  ge::AutoMappingUtil::ConvertShapeList(list, vec);
}

TEST_F(AutoMappingUtils, ConvertFuncList) {
  domi::tensorflow::AttrValue_ListValue list;
  std::vector<ge::NamedAttrs> vec;

  list.add_func();
  ge::AutoMappingUtil::ConvertFuncList(list, vec);
}
