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
#include <memory>
#include <iostream>
#include <string>
#include <map>
#include "graph/serialization/attr_serializer_registry.h"
#include "graph/serialization/string_serializer.h"
#include "graph/serialization/int_serializer.h"
#include "graph/serialization/float_serializer.h"
#include "graph/serialization/bool_serializer.h"
#include "graph/serialization/buffer_serializer.h"
#include "graph/serialization/data_type_serializer.h"
#include "graph/serialization/tensor_serializer.h"
#include "graph/serialization/tensor_desc_serializer.h"
#include "graph/serialization/list_list_int_serializer.h"
#include "graph/serialization/list_value_serializer.h"
#include "graph/serialization/list_list_float_serializer.h"
#include "graph/serialization/graph_serializer.h"
#include "graph/serialization/named_attrs_serializer.h"
#include "graph/any_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/op_desc.h"
#include "proto/ge_ir.pb.h"
#include "graph/ge_attr_value.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/model_serialize.h"
#include "graph_builder_utils.h"
#include "test_std_structs.h"

namespace ge {
GeTensorPtr CreateTensor_1_1_224_224(float *tensor_data) {
  auto tensor = std::make_shared<GeTensor>();
  tensor->SetData(reinterpret_cast<uint8_t *>(tensor_data), 224 * 224 * sizeof(float));
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetFormat(FORMAT_NCHW);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT);
  td.SetOriginDataType(DT_FLOAT);
  AttrUtils::SetStr(&td, "bcd", "Hello world");
  tensor->SetTensorDesc(td);
  return tensor;
}

ComputeGraphPtr CreateGraph_1_1_224_224(float *tensor_data) {
  ut::GraphBuilder builder("graph1");
  auto data1 = builder.AddNode("data1", "Data", {}, {"y"});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto const1 = builder.AddNode("const1", "Const", {}, {"y"});
  GeTensorDesc const1_td;
  const1_td.SetShape(GeShape({1, 1, 224, 224}));
  const1_td.SetOriginShape(GeShape({1, 1, 224, 224}));
  const1_td.SetFormat(FORMAT_NCHW);
  const1_td.SetOriginFormat(FORMAT_NCHW);
  const1_td.SetDataType(DT_FLOAT);
  const1_td.SetOriginDataType(DT_FLOAT);
  GeTensor tensor(const1_td);
  tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data), sizeof(float) * 224 * 224);
  AttrUtils::SetTensor(const1->GetOpDesc(), "value", tensor);
  auto add1 = builder.AddNode("add1", "Add", {"x1", "x2"}, {"y"});
  auto netoutput1 = builder.AddNode("NetOutputNode", "NetOutput", {"x"}, {});

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, netoutput1, 0);

  return builder.GetGraph();
}

class AttrSerializerUt : public testing::Test {};

TEST_F(AttrSerializerUt, StringAttr) {
  REG_GEIR_SERIALIZER(StringSerializer, GetTypeId<std::string>(), proto::AttrDef::kS);

  auto op_desc = std::make_shared<OpDesc>();
  AttrUtils::SetStr(op_desc, "str_name", "test_string1");

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("str_name");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kS);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  std::string res;
  value2.GetValue(res);
  ASSERT_EQ(res, "test_string1");
}

TEST_F(AttrSerializerUt, IntAttr) {
  REG_GEIR_SERIALIZER(IntSerializer, GetTypeId<int64_t>(), proto::AttrDef::kI);

  auto op_desc = std::make_shared<OpDesc>();

  int64_t val = 12344321;
  AttrUtils::SetInt(op_desc, "int_val", val);

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("int_val");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kI);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  int64_t res;
  value2.GetValue(res);
  ASSERT_EQ(res, val);
}

TEST_F(AttrSerializerUt, FloatAttr) {
  REG_GEIR_SERIALIZER(FloatSerializer, GetTypeId<float>(), proto::AttrDef::kF);

  auto op_desc = std::make_shared<OpDesc>();

  float val = 123.321f;
  AttrUtils::SetFloat(op_desc, "float_val", val);

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("float_val");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kF);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  float res;
  value2.GetValue(res);
  ASSERT_EQ(res, val);
}

TEST_F(AttrSerializerUt, TensorDescAttr) {
  REG_GEIR_SERIALIZER(GeTensorDescSerializer, GetTypeId<GeTensorDesc>(), proto::AttrDef::kTd);

  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetFormat(FORMAT_NCHW);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT);
  td.SetOriginDataType(DT_FLOAT);

  auto op_desc = std::make_shared<OpDesc>();

  AttrUtils::SetTensorDesc(op_desc, "desc_val", td);

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("desc_val");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kTd);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  GeTensorDesc res;
  ASSERT_EQ(value2.GetValue(res), GRAPH_SUCCESS);
  ASSERT_EQ(res, td);
}

TEST_F(AttrSerializerUt, BoolAttr) {
  REG_GEIR_SERIALIZER(BoolSerializer, GetTypeId<bool>(), proto::AttrDef::kI);

  auto op_desc = std::make_shared<OpDesc>();

  bool val = true;
  AttrUtils::SetBool(op_desc, "bool_val", val);

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("bool_val");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kB);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  bool res;
  value2.GetValue(res);
  ASSERT_EQ(res, val);
}

TEST_F(AttrSerializerUt, DataTypeAttr) {
  REG_GEIR_SERIALIZER(DataTypeSerializer, GetTypeId<DataType>(), proto::AttrDef::kDt);

  auto op_desc = std::make_shared<OpDesc>();
  DataType dt = DT_DOUBLE;
  AttrUtils::SetDataType(op_desc, "val", dt);

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("val");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kDt);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  DataType res;
  value2.GetValue(res);
  ASSERT_EQ(res, dt);
}

TEST_F(AttrSerializerUt, NamedAttr) {
  REG_GEIR_SERIALIZER(NamedAttrsSerializer, GetTypeId<ge::NamedAttrs>(), proto::AttrDef::kFunc);
  REG_GEIR_SERIALIZER(GeTensorSerializer, GetTypeId<GeTensor>(), proto::AttrDef::kT);
  REG_GEIR_SERIALIZER(GeTensorDescSerializer, GetTypeId<GeTensorDesc>(), proto::AttrDef::kTd);
  REG_GEIR_SERIALIZER(GraphSerializer, GetTypeId<proto::GraphDef>(), proto::AttrDef::kG);

  AnyValue value;
  value.SetValue(1.2f);

  ge::NamedAttrs named_attrs;
  named_attrs.SetName("named_attr");

  float data[224 * 224] = {1.0f};
  GeTensorPtr ge_tensor = CreateTensor_1_1_224_224(data);
  AttrUtils::SetTensor(named_attrs, "tensor_attr", ge_tensor);

  float tensor[224 * 224] = {1.0f};
  auto compute_graph = CreateGraph_1_1_224_224(tensor);
  AttrUtils::SetGraph(named_attrs, "graph_attr", compute_graph);

  auto op_desc = std::make_shared<OpDesc>();
  AttrUtils::SetNamedAttrs(op_desc, "named_attr", named_attrs);

  ModelSerializeImp impl;
  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);
  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  EXPECT_TRUE(attr_map.count("named_attr") > 0);

  auto res_op_desc = std::make_shared<OpDesc>();
  EXPECT_TRUE(impl.UnserializeOpDesc(res_op_desc, op_def));

  ge::NamedAttrs res_named_attrs;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(res_op_desc, "named_attr", res_named_attrs));
}

TEST_F(AttrSerializerUt, OpToString) {
  REG_GEIR_SERIALIZER(GeTensorSerializer, GetTypeId<GeTensor>(), proto::AttrDef::kT);
  REG_GEIR_SERIALIZER(GeTensorDescSerializer, GetTypeId<GeTensorDesc>(), proto::AttrDef::kTd);

  auto op_desc = std::make_shared<OpDesc>();
  float data[224 * 224] = {1.0f};
  GeTensorPtr ge_tensor = CreateTensor_1_1_224_224(data);
  AttrUtils::SetTensor(op_desc, "tensor", ge_tensor);

  ModelSerializeImp impl;
  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();
  std::string op_str = op_def.SerializeAsString();
  EXPECT_TRUE(attr_map.count("tensor") > 0);
}

TEST_F(AttrSerializerUt, ListFloatAttr) {
  REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<float>>(), proto::AttrDef::kList);

  auto op_desc = std::make_shared<OpDesc>();
  vector<float> val = {1.2f, 1.3f, 1.4f};
  AttrUtils::SetListFloat(op_desc, "mem_size", val);

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("mem_size");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kList);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  std::vector<float> res;
  value2.GetValue(res);
  ASSERT_EQ(res, val);
}

TEST_F(AttrSerializerUt, ListIntAttr) {
  REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<int64_t>>(), proto::AttrDef::kList);

  auto op_desc = std::make_shared<OpDesc>();
  vector<int64_t> val = {-1, 224, 224, 224};
  AttrUtils::SetListInt(op_desc, "shapes", val);

  ModelSerializeImp impl;
  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("shapes");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kList);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  std::vector<int64_t> res;
  value2.GetValue(res);
  ASSERT_EQ(res, val);
}

TEST_F(AttrSerializerUt, ListListFloatAttr) {
  REG_GEIR_SERIALIZER(ListListFloatSerializer, GetTypeId<std::vector<std::vector<float>>>(),
                      proto::AttrDef::kListListFloat);

  auto op_desc = std::make_shared<OpDesc>();
  vector<vector<float>> val = {{1.2f, 1.3f, 1.4f}};
  AttrUtils::SetListListFloat(op_desc, "mem_size", val);

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("mem_size");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kListListFloat);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  std::vector<std::vector<float>> res;
  value2.GetValue(res);
  ASSERT_EQ(res, val);
}

TEST_F(AttrSerializerUt, ListListIntAttr) {
  REG_GEIR_SERIALIZER(ListListIntSerializer, GetTypeId<std::vector<std::vector<int64_t>>>(),
                      proto::AttrDef::kListListInt);

  auto op_desc = std::make_shared<OpDesc>();
  vector<vector<int64_t>> val = {{0, 1}, {-1, 1}};
  AttrUtils::SetListListInt(op_desc, "value_range", val);

  ModelSerializeImp impl;

  proto::OpDef op_def;
  impl.SerializeOpDesc(op_desc, &op_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = op_def.attr();

  auto iter = attr_map.find("value_range");
  EXPECT_TRUE(iter != attr_map.end());

  proto::AttrDef attr2 = iter->second;
  AnyValue value2;
  auto *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kListListInt);
  ASSERT_NE(deserializer, nullptr);
  deserializer->Deserialize(attr2, value2);
  std::vector<std::vector<int64_t>> res;
  value2.GetValue(res);
  ASSERT_EQ(res, val);
}

TEST_F(AttrSerializerUt, SetAttrToComputeGraph) {
  REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<DataType>>(), proto::AttrDef::kList);

  float tensor[224 * 224] = {1.0f};
  auto computer_graph = CreateGraph_1_1_224_224(tensor);

  std::vector<DataType> dts = {DT_DOUBLE, DT_BF16};
  std::vector<std::string> strings = {"str1", "str2", "str3"};
  std::vector<bool> bools = {true, false, true};
  std::vector<Buffer> buffers = {Buffer(128), Buffer(512)};

  AttrUtils::SetListDataType(computer_graph, "list_dt", dts);
  AttrUtils::SetListStr(computer_graph, "list_str", strings);
  AttrUtils::SetListBool(computer_graph, "list_bool", bools);
  AttrUtils::SetListBytes(computer_graph, "list_buffer", buffers);

  ModelSerializeImp impl;

  proto::GraphDef graph_def;
  impl.SerializeGraph(computer_graph, &graph_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = graph_def.attr();

  EXPECT_TRUE(attr_map.count("list_dt") > 0);
  EXPECT_TRUE(attr_map.count("list_str") > 0);
  EXPECT_TRUE(attr_map.count("list_bool") > 0);
  EXPECT_TRUE(attr_map.count("list_buffer") > 0);

  auto compute_graph_gen = std::make_shared<ComputeGraph>("res_graph");
  impl.UnserializeGraph(compute_graph_gen, graph_def);

  std::map<string, AnyValue> res_map = AttrUtils::GetAllAttrs(compute_graph_gen);

  EXPECT_TRUE(res_map.count("list_dt") > 0);
  EXPECT_TRUE(res_map.count("list_str") > 0);
  EXPECT_TRUE(res_map.count("list_bool") > 0);
  EXPECT_TRUE(res_map.count("list_buffer") > 0);

  std::vector<DataType> res_dts;
  ASSERT_EQ(res_map["list_dt"].GetValue(res_dts), GRAPH_SUCCESS);
  ASSERT_EQ(res_dts, dts);

  std::vector<std::string> res_strs;
  ASSERT_EQ(res_map["list_str"].GetValue(res_strs), GRAPH_SUCCESS);
  ASSERT_EQ(res_strs, strings);

  std::vector<bool> res_bools;
  ASSERT_EQ(res_map["list_bool"].GetValue(res_bools), GRAPH_SUCCESS);
  ASSERT_EQ(res_bools, bools);

  std::vector<Buffer> res_bts;
  ASSERT_EQ(res_map["list_buffer"].GetValue(res_bts), GRAPH_SUCCESS);
  ASSERT_EQ(res_bts.size(), buffers.size());
}

TEST_F(AttrSerializerUt, SetListAttrToComputeGraph) {
  REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<NamedAttrs>>(), proto::AttrDef::kList);
  REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<GeTensor>>(), proto::AttrDef::kList);
  REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<GeTensorDesc>>(), proto::AttrDef::kList);
  REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<proto::GraphDef>>(), proto::AttrDef::kList);

  REG_GEIR_SERIALIZER(NamedAttrsSerializer, GetTypeId<ge::NamedAttrs>(), proto::AttrDef::kFunc);
  REG_GEIR_SERIALIZER(GeTensorSerializer, GetTypeId<GeTensor>(), proto::AttrDef::kT);
  REG_GEIR_SERIALIZER(GeTensorDescSerializer, GetTypeId<GeTensorDesc>(), proto::AttrDef::kTd);
  REG_GEIR_SERIALIZER(GraphSerializer, GetTypeId<proto::GraphDef>(), proto::AttrDef::kG);

  float tensor[224 * 224] = {1.0f};
  auto computer_graph = CreateGraph_1_1_224_224(tensor);
  float tensor1[224 * 224] = {1.0f};
  GeTensorPtr ge_tensor = CreateTensor_1_1_224_224(tensor1);
  std::vector<GeTensor> ge_tensors = {*ge_tensor};

  GeTensorDesc const1_td;
  const1_td.SetShape(GeShape({1, 1, 224, 224}));
  const1_td.SetOriginShape(GeShape({1, 1, 224, 224}));
  const1_td.SetFormat(FORMAT_NCHW);
  const1_td.SetOriginFormat(FORMAT_NCHW);
  const1_td.SetDataType(DT_FLOAT);
  const1_td.SetOriginDataType(DT_FLOAT);
  std::vector<GeTensorDesc> tds = {const1_td};

  ge::NamedAttrs named_attrs;
  named_attrs.SetName("named_attr");
  std::vector<NamedAttrs> attrs = {named_attrs};

  float t[224 * 224] = {2.0f};
  auto graph_t = CreateGraph_1_1_224_224(t);
  graph_t->SetName("graph_t");

  std::vector<ComputeGraphPtr> graphs = {graph_t};

  AttrUtils::SetListTensor(computer_graph, "list_t", ge_tensors);
  AttrUtils::SetListTensorDesc(computer_graph, "list_td", tds);
  AttrUtils::SetListNamedAttrs(computer_graph, "list_n", attrs);
  AttrUtils::SetListGraph(computer_graph, "list_g", graphs);

  ModelSerializeImp impl;

  proto::GraphDef graph_def;
  impl.SerializeGraph(computer_graph, &graph_def);

  google::protobuf::Map<std::string, ge::proto::AttrDef> attr_map = graph_def.attr();

  EXPECT_TRUE(attr_map.count("list_t") > 0);
  EXPECT_TRUE(attr_map.count("list_g") > 0);
  EXPECT_TRUE(attr_map.count("list_td") > 0);
  EXPECT_TRUE(attr_map.size() == 4);
  EXPECT_TRUE(attr_map.count("list_n") > 0);

  auto compute_graph_gen = std::make_shared<ComputeGraph>("res_graph");
  impl.UnserializeGraph(compute_graph_gen, graph_def);

  std::map<string, AnyValue> res_map = AttrUtils::GetAllAttrs(compute_graph_gen);

  EXPECT_TRUE(res_map.count("list_t") > 0);
  EXPECT_TRUE(res_map.count("list_g") > 0);
  EXPECT_TRUE(res_map.count("list_td") > 0);
  EXPECT_TRUE(res_map.count("list_n") > 0);

}


TEST_F(AttrSerializerUt, TdAttrInOpDesc) {
  GeTensorDesc td = StandardTd_5d_1_1_224_224();

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("x1", td);
  op_desc->AddInputDesc("x2", td);
  op_desc->AddOutputDesc("y", td);
  AttrUtils::SetStr(op_desc, "padding", "SAME");

  auto op_def = std::make_shared<proto::OpDef>();
  EXPECT_NE(op_def, nullptr);
  ModelSerializeImp imp;
  EXPECT_TRUE(imp.SerializeOpDesc(op_desc, op_def.get()));

  EXPECT_EQ(op_def->attr().count("padding"), 1);
  EXPECT_EQ(op_def->attr().at("padding").value_case(), proto::AttrDef::ValueCase::kS);
  EXPECT_EQ(op_def->input_desc_size(), 2);
  EXPECT_EQ(op_def->output_desc_size(), 1);

  ExpectStandardTdProto_5d_1_1_224_224(op_def->input_desc(0));
  ExpectStandardTdProto_5d_1_1_224_224(op_def->input_desc(1));
  ExpectStandardTdProto_5d_1_1_224_224(op_def->output_desc(0));
}

TEST_F(AttrSerializerUt, ConstSerializer) {
  ut::GraphBuilder builder("graph1");
  auto data1 = builder.AddNode("data1", "Data", {}, {"y"});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto const1 = builder.AddNode("const1", "Const", {}, {"y"});
  GeTensorDesc const1_td;
  const1_td.SetShape(GeShape({1, 1, 224, 224}));
  const1_td.SetOriginShape(GeShape({1, 1, 224, 224}));
  const1_td.SetFormat(FORMAT_NCHW);
  const1_td.SetOriginFormat(FORMAT_NCHW);
  const1_td.SetDataType(DT_FLOAT);
  const1_td.SetOriginDataType(DT_FLOAT);
  GeTensor tensor(const1_td);
  float tensor_data[224*224] = {1.010101, 2.020202, 3.030303};
  tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data), sizeof(float) * 224 * 224);
  AttrUtils::SetTensor(const1->GetOpDesc(), "value", tensor);

  ComputeGraphPtr graph = builder.GetGraph();

  ModelSerializeImp impl;
  proto::GraphDef graph_def;
  impl.SerializeGraph(graph, &graph_def);
  auto compute_graph_gen = std::make_shared<ComputeGraph>("res_graph");

  impl.UnserializeGraph(compute_graph_gen, graph_def);

  NodePtr res_node = compute_graph_gen->FindNode("const1");
  EXPECT_TRUE(res_node != nullptr);

  ConstGeTensorPtr res_tensor;
  EXPECT_TRUE(AttrUtils::GetTensor(res_node->GetOpDesc(), "value", res_tensor));
}

}  // namespace ge