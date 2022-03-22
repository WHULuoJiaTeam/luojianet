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

#include <gtest/gtest.h>

#define protected public
#define private public

#include "external/graph/operator.h"
#include "graph/operator_impl.h"
#include "external/graph/tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/op_desc_impl.h"
#include "graph/tensor_type_impl.h"
#include "graph_builder_utils.h"
#include <string.h>

#undef private
#undef protected

namespace ge {
class UtestOperater : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOperater, GetInputConstData) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1);
  auto enter = builder.AddNode("Enter", "Enter", 1, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data2, 0, enter, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(enter, 0, transdata, 1);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  auto ge_tensor = std::make_shared<GeTensor>();
  auto op_desc = transdata->GetOpDesc();
  op_desc->impl_->input_name_idx_["Data"] = 0;
  op_desc->impl_->input_name_idx_["Enter"] = 1;
  auto tensor_desc = op_desc->MutableInputDesc(0);
  AttrUtils::SetTensor(tensor_desc, "_value", ge_tensor);

  Tensor tensor;
  auto op = OpDescUtils::CreateOperatorFromNode(transdata);
  ASSERT_EQ(op.GetInputConstData("Data", tensor), GRAPH_SUCCESS);
  ASSERT_EQ(op.GetInputConstData("Enter", tensor), GRAPH_FAILED);
}
/**                                   --------------------------
 *         const                     |   sub_data    sub_const  |
 *          |                        |         \    /           |
 *        case-----------------------|          Add             |
 *         |                         |          |               |
 *      netoutput                    |     sub_netoutput        |
 *                                   ---------------------------
 */
TEST_F(UtestOperater, GetInputConstData_subgraph) {
  auto ge_tensor = std::make_shared<GeTensor>();
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", ge_tensor);
  auto case_node = builder.AddNode("Case", "Case", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(const_node, 0, case_node, 0);
  builder.AddDataEdge(case_node, 0, netoutput, 0);
  auto parent_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder = ut::GraphBuilder("subgraph_graph");
  auto sub_data = sub_builder.AddNode("sub_data", "Data", 0, 1);
  auto sub_const = sub_builder.AddNode("sub_const", "Const", 0, 1);
  AttrUtils::SetTensor(sub_const->GetOpDesc(), "value", ge_tensor);
  auto add = sub_builder.AddNode("Add", "Add", 2, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_netoutput", "NetOutput", 1, 0);
  sub_builder.AddDataEdge(sub_data, 0, add, 0);
  sub_builder.AddDataEdge(sub_const, 0, add, 1);
  sub_builder.AddDataEdge(add, 0, sub_netoutput, 0);

  auto subgraph = sub_builder.GetGraph();
  subgraph->SetParentNode(case_node);
  subgraph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(subgraph->GetName(), subgraph);
  AttrUtils::SetInt(sub_data->GetOpDesc(), "_parent_node_index", 0);

  auto op_desc = add->GetOpDesc();
  op_desc->impl_->input_name_idx_["sub_data"] = 0;
  op_desc->impl_->input_name_idx_["sub_const"] = 1;

  Tensor tensor;
  auto op = OpDescUtils::CreateOperatorFromNode(add);
  ASSERT_EQ(op.GetInputConstData("sub_const", tensor), GRAPH_SUCCESS);
  ASSERT_EQ(op.GetInputConstData("sub_data", tensor), GRAPH_SUCCESS);
}

TEST_F(UtestOperater, TestOperatorSetInputs) {
  ge::Operator dst_op = ge::Operator("Mul");
  ge::Operator src_op = ge::Operator("Add");
  dst_op.InputRegister("x1");
  dst_op.InputRegister("x2");
  dst_op.OutputRegister("y");

  src_op.InputRegister("x1");
  src_op.InputRegister("x2");
  src_op.OutputRegister("y");

  ASSERT_EQ(src_op.GetInputsSize(), 2U);
  ASSERT_EQ(dst_op.GetInputsSize(), 2U);
  // src_index is illegal
  (void)dst_op.SetInput(0U, src_op, 3U);
  ASSERT_EQ(src_op.GetInputsSize(), 2U);
  // dst_index is illegal
  (void)dst_op.SetInput(3U, src_op, 0U);
  ASSERT_EQ(src_op.GetInputsSize(), 2U);

  (void)dst_op.SetInput(1U, src_op, 0U);
  ASSERT_EQ(src_op.GetInputsSize(), 2U);

  ge::Operator null_op;
  (void)null_op.SetInput(1U, src_op, 0U);
  ASSERT_EQ(null_op.GetInputsSize(), 0U);

  std::string dst_name = "x1";
  (void)dst_op.SetInput(dst_name, src_op, 0U);
  ASSERT_EQ(dst_op.GetInputsSize(), 2U);
}

TEST_F(UtestOperater, AttrRegister_Float) {
  auto op = Operator("Data");
  std::string attr = "attr";
  float value = 1.0;
  op.AttrRegister(attr.c_str(), value);
  float ret = 0;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_FLOAT_EQ(value, ret);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListFloat) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<float> value = {1.0, 2.0};
  op.AttrRegister(attr.c_str(), value);
  std::vector<float> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_FLOAT_EQ(value[0], ret[0]);
  ASSERT_FLOAT_EQ(value[1], ret[1]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_Int) {
  auto op = Operator("Data");
  std::string attr = "attr";
  int64_t value = 1;
  op.AttrRegister(attr.c_str(), value);
  int64_t ret = 0;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value, ret);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListInt) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<int64_t> value = {1, 2};
  op.AttrRegister(attr.c_str(), value);
  std::vector<int64_t> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0], ret[0]);
  ASSERT_EQ(value[1], ret[1]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_String) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::string value = "on";
  op.AttrRegister(attr.c_str(), value.c_str());
  std::string ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value, ret);
  op.AttrRegister(nullptr, value.c_str());
}

TEST_F(UtestOperater, AttrRegister_Bool) {
  auto op = Operator("Data");
  std::string attr = "attr";
  bool value = true;
  op.AttrRegister(attr.c_str(), value);
  bool ret = false;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value, ret);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListBool) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<bool> value = {false, true};
  op.AttrRegister(attr.c_str(), value);
  std::vector<bool> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0], ret[0]);
  ASSERT_EQ(value[1], ret[1]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_Tensor) {
  auto op = Operator("Data");
  auto value = Tensor();
  op.AttrRegister("attr", value);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListTensor) {
  auto op = Operator("Data");
  std::vector<Tensor> value = {Tensor()};
  op.AttrRegister("attr", value);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_OpBytes) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = OpBytes{1, 2, 3};
  op.AttrRegister(attr.c_str(), value);
  OpBytes ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0], ret[0]);
  ASSERT_EQ(value[1], ret[1]);
  ASSERT_EQ(value[2], ret[2]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListListInt) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<std::vector<int64_t>> value = {{1, 2}, {3}};
  op.AttrRegister(attr.c_str(), value);
  std::vector<std::vector<int64_t>> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0][0], ret[0][0]);
  ASSERT_EQ(value[0][1], ret[0][1]);
  ASSERT_EQ(value[1][0], ret[1][0]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListDataType) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<DataType> value = {DataType::DT_FLOAT, DataType::DT_INT64};
  op.AttrRegister(attr.c_str(), value);
  std::vector<DataType> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0], ret[0]);
  ASSERT_EQ(value[1], ret[1]);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_DataType) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = DataType::DT_FLOAT;
  op.AttrRegister(attr.c_str(), value);
  DataType ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value, ret);
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_NamedAttrs) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = NamedAttrs();
  value.SetName("name");
  op.AttrRegister(attr.c_str(), value);
  NamedAttrs ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value.GetName(), ret.GetName());
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListNamedAttrs) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<NamedAttrs> value = {NamedAttrs()};
  value[0].SetName("name");
  op.AttrRegister(attr.c_str(), value);
  std::vector<NamedAttrs> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(value[0].GetName(), ret[0].GetName());
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_AscendString) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = AscendString("1");
  op.AttrRegister(attr.c_str(), value);
  AscendString ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(std::string(value.GetString()), std::string(ret.GetString()));
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_AscendString2) {
  auto op = Operator("Data");
  std::string attr = "attr";
  auto value = AscendString("1");
  op.AttrRegister(attr, value);
  AscendString ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(std::string(value.GetString()), std::string(ret.GetString()));
  op.AttrRegister(attr, AscendString(""));
}

TEST_F(UtestOperater, AttrRegister_ListAscendString) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<AscendString> value = {AscendString("1")};
  op.AttrRegister(attr.c_str(), value);
  std::vector<AscendString> ret;
  op.GetAttr(attr.c_str(), ret);
  ASSERT_EQ(std::string(value[0].GetString()), std::string(ret[0].GetString()));
  op.AttrRegister(nullptr, value);
}

TEST_F(UtestOperater, AttrRegister_ListString) {
  auto op = Operator("Data");
  std::string attr = "attr";
  std::vector<std::string> value ;
  op.AttrRegister(attr, value);
  std::vector<std::string> ret;
  op.GetAttr(attr, ret);
  ASSERT_EQ(ret.size(), 0);
}

TEST_F(UtestOperater, InputRegister) {
  auto op = Operator("Data");
  std::string name = "data";
  op.InputRegister(name.c_str());
  op.InputRegister(nullptr);
}

TEST_F(UtestOperater, OptionalInputRegister) {
  auto op = Operator("Data");
  std::string name = "data";
  op.OptionalInputRegister(name.c_str());
  op.OptionalInputRegister(nullptr);
}

TEST_F(UtestOperater, OutputRegister) {
  auto op = Operator("Data");
  std::string name = "data";
  op.OutputRegister(name.c_str());
  op.OutputRegister(nullptr);
}

TEST_F(UtestOperater, DynamicInputRegister) {
  auto op = Operator("Data");
  std::string name = "data";
  op.DynamicInputRegister(name.c_str(), 1);
  op.DynamicInputRegister(nullptr, 1);
}

TEST_F(UtestOperater, DynamicInputRegisterByIndex) {
  auto op = Operator("Data");
  std::string name = "data";
  op.DynamicInputRegisterByIndex(name.c_str(), 1, 0);
  op.DynamicInputRegisterByIndex(nullptr, 1, 0);
}

TEST_F(UtestOperater, DynamicOutputRegister) {
  auto op = Operator("Data");
  std::string name = "data";
  op.DynamicOutputRegister(name.c_str(), 1);
  op.DynamicOutputRegister(nullptr, 1);
}

TEST_F(UtestOperater, RequiredAttrRegister) {
  auto op = Operator("Data");
  std::string name = "data";
  op.RequiredAttrRegister(name.c_str());
  op.RequiredAttrRegister(nullptr);
}

TEST_F(UtestOperater, SetInput_WithoutName) {
  auto op = Operator("Add");
  std::string dst_name = "data";
  uint32_t dst_index = 1;
  auto dst_op = Operator("Data");
  op.SetInput(dst_name.c_str(), dst_index, dst_op);
  op.SetInput(nullptr, dst_index, dst_op);
}

TEST_F(UtestOperater, SetInput_WithName) {
  std::string name = "add";
  auto op = Operator("Add");
  std::string dst_name = "data";
  uint32_t dst_index = 1;
  auto dst_op = Operator("Data");
  op.SetInput(dst_name.c_str(), dst_index, dst_op, name.c_str());
  op.SetInput(nullptr, dst_index, dst_op, name.c_str());
  op.SetInput(dst_name.c_str(), dst_index, dst_op, nullptr);
}

TEST_F(UtestOperater, SubgraphRegister) {
  std::string name = "add";
  auto op = Operator("Add");
  bool dynamic = true;
  op.SubgraphRegister(name.c_str(), dynamic);
  op.SubgraphRegister(nullptr, dynamic);
}

TEST_F(UtestOperater, SubgraphCountRegister) {
  std::string name = "add";
  auto op = Operator("Add");
  uint32_t count = 1;
  op.SubgraphCountRegister(name.c_str(), count);
  op.SubgraphCountRegister(nullptr, count);
}

TEST_F(UtestOperater, SetSubgraphBuilder) {
  std::string name = "add";
  auto op = Operator("Add");
  uint32_t index = 1;
  SubgraphBuilder builder = []() {return Graph();};
  op.SetSubgraphBuilder(name.c_str(), index, builder);
  op.SetSubgraphBuilder(nullptr, index, builder);
}

TEST_F(UtestOperater, GetSubgraphImpl) {
  std::string name = "add";
  auto op = Operator("Add");
  op.GetSubgraphImpl(name.c_str());
  op.GetSubgraphImpl(nullptr);
}

TEST_F(UtestOperater, SetInput_Handler) {
  std::string name = "add";
  std::string type = "Add";
  int index = 1;
  auto op = Operator(type);
  auto handler = OutHandler(nullptr);
  op.SetInput(name.c_str(), handler);
  op.SetInput(nullptr, handler);
}

TEST_F(UtestOperater, GetOutput) {
  std::string name = "add";
  auto op = Operator("Add");
  op.GetOutput(name.c_str());
  op.GetOutput(nullptr);
}

TEST_F(UtestOperater, GetInputConstDataOut) {
  std::string name = "add";
  auto op = Operator("Add");
  Tensor a = Tensor();
  ASSERT_EQ(op.GetInputConstDataOut(name.c_str(), a), GRAPH_FAILED);
  ASSERT_EQ(op.GetInputConstDataOut(nullptr, a), GRAPH_FAILED);
}

TEST_F(UtestOperater, testTensorType) {
  DataType dt(DT_INT16);
  TensorType tt1(dt);
  EXPECT_EQ(tt1.tensor_type_impl_->dt_vec_[0], DT_INT16);

  const std::initializer_list<DataType> types = {DT_INT8, DT_UINT8, DT_INT16};
  TensorType tt2(types);
  EXPECT_EQ(tt2.tensor_type_impl_->dt_vec_.size(), 3);
}

TEST_F(UtestOperater, CreateOperator) {
  Operator op;
  OpDescPtr op_desc;

  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  EXPECT_FALSE(op.IsEmpty());
}

TEST_F(UtestOperater, testGetName) {
  AscendString name;
  Operator op("one_op", "add");
  op.GetName(name);

  const char *str = name.GetString();
  EXPECT_EQ(strcmp(str, "one_op"), 0);
}

TEST_F(UtestOperater, GetInputConstData2) {
  Operator op;
  OpDescPtr op_desc;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  std::string dst_name("dst_name");
  Tensor td;

  EXPECT_NE(op.GetInputConstData(dst_name, td), GRAPH_SUCCESS);
}

TEST_F(UtestOperater, GetNode) {
  Operator op;
  OpDescPtr op_desc;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  EXPECT_EQ(op.GetNode(), nullptr);
}

TEST_F(UtestOperater, GetInputDesc) {
  Operator op;
  OpDescPtr op_desc;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  std::string str_name = "input_desc_name";
  TensorDesc td = op.GetInputDesc(str_name);

  EXPECT_EQ(td.GetName().length(), 0);
}

TEST_F(UtestOperater, TryGetInputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td;
  auto ret = op.TryGetInputDesc("input_name_1", td);
  EXPECT_EQ(ret, GRAPH_FAILED);

  std:string str = "input_name_2";
  ret = op.TryGetInputDesc(str, td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, UpdateInputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td;
  std:string str = "input_name";
  auto ret = op.UpdateInputDesc(str, td);
  EXPECT_EQ(ret, GRAPH_FAILED);

  ret = op.UpdateInputDesc("input_name", td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetOutputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  std:string str = "output_name";
  TensorDesc td = op.GetOutputDesc(str);
  EXPECT_EQ(td.GetName().length(), 0);
}

TEST_F(UtestOperater, UpdateOutputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td;
  std:string str = "output_name";
  auto ret = op.UpdateOutputDesc(str, td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetDynamicInputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  std:string str = "input_name";
  TensorDesc td_1 = op.GetDynamicInputDesc(str, 0);
  TensorDesc td_2 = op.GetDynamicInputDesc("input_name", 0);
  EXPECT_EQ(td_1.GetName().length(), 0);
  EXPECT_EQ(td_2.GetName().length(), 0);
}

TEST_F(UtestOperater, UpdateDynamicInputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td_1;
  std:string str = "input_name";
  auto ret = op.UpdateDynamicInputDesc(str, 0, td_1);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = op.UpdateDynamicInputDesc("input_name", 0, td_1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetDynamicOutputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  string str = "output_name";
  TensorDesc td_1 = op.GetDynamicOutputDesc(str, 0);
  TensorDesc td_2 = op.GetDynamicOutputDesc("output_name", 0);
  EXPECT_EQ(td_1.GetName().length(), 0);
  EXPECT_EQ(td_2.GetName().length(), 0);
}

TEST_F(UtestOperater, UpdateDynamicOutputDesc) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  TensorDesc td_1;
  string str = "output_name";
  auto ret = op.UpdateDynamicOutputDesc(str, 0, td_1);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = op.UpdateDynamicOutputDesc("output_name", 0, td_1);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, InferShapeAndType) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.InferShapeAndType();
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, VerifyAllAttr) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.VerifyAllAttr(true);
  EXPECT_EQ(ret, GRAPH_FAILED);

  ret = op.VerifyAllAttr(false);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetAllAttrNamesAndTypes) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.GetAllAttrNamesAndTypes();
  EXPECT_EQ(ret.size(), 0);

  std::map<AscendString, AscendString> attr_name_types;
  auto ret_2 = op.GetAllAttrNamesAndTypes(attr_name_types);
  EXPECT_EQ(ret_2, GRAPH_FAILED);
}

TEST_F(UtestOperater, FuncRegister) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  std::function<graphStatus(Operator &)> func;

  op.InferFuncRegister(func);

  if (op.operator_impl_->GetOpDescImpl() != nullptr) {
    printf("FuncRegister GetOpDescImpl is not null!\n");
    //auto ret1 = op.operator_impl_->GetOpDescImpl()->GetInferFunc();
    //EXPECT_EQ(ret1, nullptr);
  } else {
    printf("FuncRegister GetOpDescImpl is null!\n");
  }

  ASSERT_NE(op.operator_impl_, nullptr);
}

TEST_F(UtestOperater, FuncRegister2) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);
  std::function<graphStatus(Operator &)> func;

  op.InferFormatFuncRegister(func);
  op.VerifierFuncRegister(func);

  ASSERT_NE(op.operator_impl_, nullptr);
}

TEST_F(UtestOperater, GetDynamicInputNum) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  int num1 = op.GetDynamicInputNum("input_name");
  EXPECT_EQ(num1, 0);

  int num2 = op.GetDynamicInputNum(std::string("input_name"));
  EXPECT_EQ(num2, 0);
}

TEST_F(UtestOperater, GetDynamicOutputNum) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  int num1 = op.GetDynamicOutputNum("output_name");
  EXPECT_EQ(num1, 0);

  int num2 = op.GetDynamicOutputNum(std::string("output_name"));
  EXPECT_EQ(num2, 0);
}

TEST_F(UtestOperater, VerifyAll) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.VerifyAll();
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestOperater, GetOperatorImplPtr) {
  Operator op;
  OpDescPtr op_desc_1;
  op = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op.GetOperatorImplPtr();
  EXPECT_NE(ret, nullptr);
}

TEST_F(UtestOperater, AddControlInput_Exception) {
  Operator op1;
  Operator op2;
  OpDescPtr op_desc_1;
  op2 = OpDescUtils::CreateOperatorFromOpDesc(op_desc_1);

  auto ret = op1.AddControlInput(op2);
  EXPECT_EQ(op1.IsEmpty(), ret.IsEmpty());
}

TEST_F(UtestOperater, SetAttr1) {
  Operator op1;
  Operator op2;

  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  char_t *name = "data name";
  char_t *attr_value = "abc";

  op2 = op1.SetAttr(name, attr_value);
  std::string value1;

  op1.GetAttr(name, value1);
  printf("c_str1 = %s\n", value1.c_str());

  std::string value2;
  op2.GetAttr(name, value2);
  printf("c_str2 = %s\n", value2.c_str());
  EXPECT_EQ(value2, std::string("abc"));

  op1.SetAttr(nullptr, nullptr);
}

TEST_F(UtestOperater, SetAttr2) {
  Operator op1;
  Operator op2;

  std::string optype_str = "optype";
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  op1 = OpDescUtils::CreateOperatorFromOpDesc(op_desc);

  char_t *name = "data name";
  AscendString attr_value = "abc";

  op2 = op1.SetAttr(name, attr_value);

  std::string value2;
  op2.GetAttr(name, value2);

  EXPECT_EQ(value2, std::string("abc"));

  op1.SetAttr(nullptr, attr_value);
}




}  // namespace ge
