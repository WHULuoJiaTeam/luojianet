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
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define private public
#define protected public
#include "graph/model_serialize.h"

#include "graph/detail/model_serialize_imp.h"
#include "graph/node_impl.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#undef private
#undef protected

#include "proto/ge_ir.pb.h"

using namespace ge;
using std::string;
using std::vector;

bool LinkEdge(NodePtr src_node, int32_t src_index, NodePtr dst_node, int32_t dst_index) {
  if (src_index >= 0) {
    auto src_anchor = src_node->GetOutDataAnchor(src_index);
    auto dst_anchor = dst_node->GetInDataAnchor(dst_index);
    src_anchor->LinkTo(dst_anchor);
  } else {
    auto src_anchor = src_node->GetOutControlAnchor();
    auto dst_anchor = dst_node->GetInControlAnchor();
    src_anchor->LinkTo(dst_anchor);
  }
}

NodePtr CreateNode(OpDescPtr op, ComputeGraphPtr owner_graph) { return owner_graph->AddNode(op); }

void CompareShape(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
  EXPECT_EQ(shape1.size(), shape2.size());
  if (shape1.size() == shape2.size()) {
    for (size_t i = 0; i < shape1.size(); i++) {
      EXPECT_EQ(shape1[i], shape2[i]);
    }
  }
}

template <typename T>
void CompareList(const vector<T> &val1, const vector<T> &val2) {
  EXPECT_EQ(val1.size(), val2.size());
  if (val1.size() == val2.size()) {
    for (size_t i = 0; i < val1.size(); i++) {
      EXPECT_EQ(val1[i], val2[i]);
    }
  }
}

static bool NamedAttrsSimpleCmp(const GeAttrValue &left, const GeAttrValue &right) {
  GeAttrValue::NamedAttrs val1, val2;
  left.GetValue<GeAttrValue::NamedAttrs>(val1);
  right.GetValue<GeAttrValue::NamedAttrs>(val2);
  if (val1.GetName() != val2.GetName()) {
    return false;
  }
  auto attrs1 = val1.GetAllAttrs();
  auto attrs2 = val2.GetAllAttrs();
  if (attrs1.size() != attrs1.size()) {
    return false;
  }

  for (auto it : attrs1) {
    auto it2 = attrs2.find(it.first);
    if (it2 == attrs2.end()) {  // simple check
      return false;
    }
    if (it.second.GetValueType() != it2->second.GetValueType()) {
      return false;
    }
    switch (it.second.GetValueType()) {
      case GeAttrValue::VT_INT: {
        int64_t i1 = 0, i2 = 0;
        it.second.GetValue<GeAttrValue::INT>(i1);
        it2->second.GetValue<GeAttrValue::INT>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      case GeAttrValue::VT_FLOAT: {
        GeAttrValue::FLOAT i1 = 0, i2 = 0;
        it.second.GetValue<GeAttrValue::FLOAT>(i1);
        it2->second.GetValue<GeAttrValue::FLOAT>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      case GeAttrValue::VT_STRING: {
        string i1, i2;
        it.second.GetValue<GeAttrValue::STR>(i1);
        it2->second.GetValue<GeAttrValue::STR>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      case GeAttrValue::VT_BOOL: {
        bool i1 = false, i2 = false;
        it.second.GetValue<GeAttrValue::BOOL>(i1);
        it2->second.GetValue<GeAttrValue::BOOL>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      default: {
        return true;
      }
    }
  }
  return true;
}

static GeAttrValue::NamedAttrs CreateNamedAttrs(const string &name, std::map<string, GeAttrValue> map) {
  GeAttrValue::NamedAttrs named_attrs;
  named_attrs.SetName(name);
  for (auto it : map) {
    named_attrs.SetAttr(it.first, it.second);
  }
  return named_attrs;
}

TEST(UtestGeModelSerialize, simple) {
  Model model("model_name", "custom version3.0");
  model.SetAttr("model_key1", GeAttrValue::CreateFrom<GeAttrValue::INT>(123));
  model.SetAttr("model_key2", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(456.78f));
  model.SetAttr("model_key3", GeAttrValue::CreateFrom<GeAttrValue::STR>("abcd"));
  model.SetAttr("model_key4", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
  model.SetAttr("model_key5", GeAttrValue::CreateFrom<GeAttrValue::LIST_FLOAT>({456.78f, 998.90f}));
  model.SetAttr("model_key6", GeAttrValue::CreateFrom<GeAttrValue::LIST_STR>({"abcd", "happy"}));
  model.SetAttr("model_key7", GeAttrValue::CreateFrom<GeAttrValue::BOOL>(false));
  model.SetAttr("model_key8", GeAttrValue::CreateFrom<GeAttrValue::LIST_BOOL>({true, false}));

  auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

  // input
  auto input_op = std::make_shared<OpDesc>("input", "Input");
  input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  auto input = CreateNode(input_op, compute_graph);
  // w1
  auto w1_op = std::make_shared<OpDesc>("w1", "ConstOp");
  w1_op->AddOutputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  auto w1 = CreateNode(w1_op, compute_graph);

  // node1
  auto node1_op = std::make_shared<OpDesc>("node1", "Conv2D");
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  node1_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  auto node1 = CreateNode(node1_op, compute_graph);

  // Attr set
  node1_op->SetAttr("node_key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(Buffer(10)));
  node1_op->SetAttr("node_key2", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({Buffer(20), Buffer(30)}));
  auto named_attrs1 = GeAttrValue::CreateFrom<GeAttrValue::NAMED_ATTRS>(
      CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                   {"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                   {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}));

  node1_op->SetAttr("node_key3", std::move(named_attrs1));
  auto list_named_attrs = GeAttrValue::CreateFrom<GeAttrValue::LIST_NAMED_ATTRS>(
      {CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                    {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}),
       CreateNamedAttrs("my_name2", {{"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                     {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}})});
  node1_op->SetAttr("node_key4", std::move(list_named_attrs));
  // tensor
  auto tensor_data1 = "qwertyui";
  auto tensor1 =
      std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT8), (uint8_t *)tensor_data1, 8);
  auto tensor_data2 = "asdfqwertyui";
  auto tensor2 =
      std::make_shared<GeTensor>(GeTensorDesc(GeShape({3, 2, 2}), FORMAT_ND, DT_UINT8), (uint8_t *)tensor_data2, 12);
  auto tensor_data3 = "ghjkasdfqwertyui";
  auto tensor3 =
      std::make_shared<GeTensor>(GeTensorDesc(GeShape({4, 2, 2}), FORMAT_ND, DT_UINT16), (uint8_t *)tensor_data3, 16);
  node1_op->SetAttr("node_key5", GeAttrValue::CreateFrom<GeAttrValue::TENSOR>(tensor1));
  node1_op->SetAttr("node_key6", GeAttrValue::CreateFrom<GeAttrValue::LIST_TENSOR>({tensor2, tensor3}));

  auto tensor_desc = GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT16);
  TensorUtils::SetSize(tensor_desc, 100);
  node1_op->SetAttr("node_key7", GeAttrValue::CreateFrom<GeAttrValue::TENSOR_DESC>(tensor_desc));
  node1_op->SetAttr("node_key8", GeAttrValue::CreateFrom<GeAttrValue::LIST_TENSOR_DESC>(
                                    {GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT32),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_UINT32),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT64),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_UINT64),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_BOOL),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_DOUBLE)}));

  LinkEdge(input, 0, node1, 0);
  LinkEdge(w1, 0, node1, 1);

  Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  model.SetGraph(graph);

  Buffer buffer;
  ASSERT_EQ(model.Save(buffer), GRAPH_SUCCESS);
  EXPECT_TRUE(buffer.GetData() != nullptr);

  Model model2;
  ASSERT_EQ(Model::Load(buffer.GetData(), buffer.GetSize(), model2), GRAPH_SUCCESS);
  EXPECT_EQ(model2.GetName(), "model_name");
  GeAttrValue::INT model_val1;
  AttrUtils::GetInt(&model2, "model_key1", model_val1);
  EXPECT_EQ(model_val1, 123);

  GeAttrValue::FLOAT model_val2;
  AttrUtils::GetFloat(&model2, "model_key2", model_val2);
  EXPECT_EQ(model_val2, (float)456.78f);

  GeAttrValue::STR model_val3;
  AttrUtils::GetStr(&model2, "model_key3", model_val3);
  EXPECT_EQ(model_val3, "abcd");

  GeAttrValue::LIST_INT model_val4;
  AttrUtils::GetListInt(&model2, "model_key4", model_val4);
  CompareList(model_val4, {123, 456});

  GeAttrValue::LIST_FLOAT model_val5;
  AttrUtils::GetListFloat(&model2, "model_key5", model_val5);
  CompareList(model_val5, {456.78f, 998.90f});

  GeAttrValue::LIST_STR model_val6;
  AttrUtils::GetListStr(&model2, "model_key6", model_val6);
  CompareList(model_val6, {"abcd", "happy"});

  GeAttrValue::BOOL model_val7;
  EXPECT_EQ(AttrUtils::GetBool(&model2, "model_key7", model_val7), true);
  EXPECT_EQ(model_val7, false);

  GeAttrValue::LIST_BOOL model_val8;
  AttrUtils::GetListBool(&model2, "model_key8", model_val8);
  CompareList(model_val8, {true, false});

  auto graph2 = model2.GetGraph();
  const auto &s_graph = GraphUtils::GetComputeGraph(graph2);
  ASSERT_TRUE(s_graph != nullptr);
  auto s_nodes = s_graph->GetDirectNode();
  ASSERT_EQ(3, s_nodes.size());

  auto s_input = s_nodes.at(0);
  auto s_w1 = s_nodes.at(1);
  auto s_nod1 = s_nodes.at(2);
  {
    auto s_op = s_input->GetOpDesc();
    EXPECT_EQ(s_op->GetName(), "input");
    EXPECT_EQ(s_op->GetType(), "Input");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 0);
    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc1 = s_output_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto out_anchor = s_input->GetOutDataAnchor(0);
    auto peer_anchors = out_anchor->GetPeerInDataAnchors();
    ASSERT_EQ(peer_anchors.size(), 1);
    auto peer_anchor = peer_anchors.at(0);
    ASSERT_EQ(peer_anchor->GetIdx(), 0);
    ASSERT_EQ(peer_anchor->GetOwnerNode(), s_nod1);
  }

  {
    auto s_op = s_w1->GetOpDesc();
    EXPECT_EQ(s_op->GetName(), "w1");
    EXPECT_EQ(s_op->GetType(), "ConstOp");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 0);
    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc1 = s_output_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT16);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

    auto out_anchor = s_w1->GetOutDataAnchor(0);
    auto peer_anchors = out_anchor->GetPeerInDataAnchors();
    ASSERT_EQ(peer_anchors.size(), 1);
    auto peer_anchor = peer_anchors.at(0);
    ASSERT_EQ(peer_anchor->GetIdx(), 1);
    ASSERT_EQ(peer_anchor->GetOwnerNode(), s_nod1);
  }
  {
    auto s_op = s_nod1->GetOpDesc();
    EXPECT_EQ(s_op->GetName(), "node1");
    EXPECT_EQ(s_op->GetType(), "Conv2D");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 2);

    auto desc1 = s_input_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto desc2 = s_input_descs.at(1);
    EXPECT_EQ(desc2.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(desc2.GetDataType(), DT_FLOAT16);
    CompareShape(desc2.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc3 = s_output_descs.at(0);
    EXPECT_EQ(desc3.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc3.GetDataType(), DT_FLOAT);
    CompareShape(desc3.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto out_anchor = s_nod1->GetOutDataAnchor(0);
    auto peer_anchors = out_anchor->GetPeerInDataAnchors();
    ASSERT_EQ(peer_anchors.size(), 0);

    // node attrs
    GeAttrValue::BYTES node_val1;
    AttrUtils::GetBytes(s_op, "node_key1", node_val1);
    ASSERT_EQ(node_val1.GetSize(), 10);

    GeAttrValue::LIST_BYTES node_val2;
    AttrUtils::GetListBytes(s_op, "node_key2", node_val2);
    ASSERT_EQ(node_val2.size(), 2);
    ASSERT_EQ(node_val2[0].GetSize(), 20);
    ASSERT_EQ(node_val2[1].GetSize(), 30);

    GeAttrValue s_named_attrs;
    s_op->GetAttr("node_key3", s_named_attrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_named_attrs, named_attrs1));

    GeAttrValue s_list_named_attrs;
    s_op->GetAttr("node_key4", s_list_named_attrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_list_named_attrs, list_named_attrs));

    ConstGeTensorPtr s_tensor;
    AttrUtils::GetTensor(s_op, "node_key5", s_tensor);
    ASSERT_TRUE(s_tensor != nullptr);
    string str((char *)s_tensor->GetData().data(), s_tensor->GetData().size());
    EXPECT_EQ(str, "qwertyui");

    vector<ConstGeTensorPtr> s_list_tensor;
    AttrUtils::GetListTensor(s_op, "node_key6", s_list_tensor);
    ASSERT_EQ(s_list_tensor.size(), 2);
    string str2((char *)s_list_tensor[0]->GetData().data(), s_list_tensor[0]->GetData().size());
    EXPECT_EQ(str2, "asdfqwertyui");
    string str3((char *)s_list_tensor[1]->GetData().data(), s_list_tensor[1]->GetData().size());
    EXPECT_EQ(str3, "ghjkasdfqwertyui");

    GeTensorDesc s_tensor_desc;
    AttrUtils::GetTensorDesc(s_op, "node_key7", s_tensor_desc);
    EXPECT_EQ(s_tensor_desc.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(s_tensor_desc.GetDataType(), DT_INT16);
    int64_t size = 0;
    TensorUtils::GetSize(s_tensor_desc, size);
    EXPECT_EQ(size, 100);

    vector<GeTensorDesc> s_list_tensor_desc;
    AttrUtils::GetListTensorDesc(s_op, "node_key8", s_list_tensor_desc);
    ASSERT_EQ(s_list_tensor_desc.size(), 6);
    EXPECT_EQ(s_list_tensor_desc[0].GetDataType(), DT_INT32);
    EXPECT_EQ(s_list_tensor_desc[1].GetDataType(), DT_UINT32);
    EXPECT_EQ(s_list_tensor_desc[2].GetDataType(), DT_INT64);
    EXPECT_EQ(s_list_tensor_desc[3].GetDataType(), DT_UINT64);
    EXPECT_EQ(s_list_tensor_desc[4].GetDataType(), DT_BOOL);
    EXPECT_EQ(s_list_tensor_desc[5].GetDataType(), DT_DOUBLE);
  }
}

TEST(UtestGeModelSerialize, op_desc) {
  // node1_op
  auto node1_op = std::make_shared<OpDesc>("node1", "Conv2D");
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  node1_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

  // Attr set
  node1_op->SetAttr("node_key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(Buffer(10)));
  node1_op->SetAttr("node_key2", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({Buffer(20), Buffer(30)}));
  auto named_attrs1 = GeAttrValue::CreateFrom<GeAttrValue::NAMED_ATTRS>(
      CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                   {"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                   {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}));

  node1_op->SetAttr("node_key3", std::move(named_attrs1));
  auto list_named_attrs = GeAttrValue::CreateFrom<GeAttrValue::LIST_NAMED_ATTRS>(
      {CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                    {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}),
       CreateNamedAttrs("my_name2", {{"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                     {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}})});
  node1_op->SetAttr("node_key4", std::move(list_named_attrs));

  ModelSerialize model_serialize;
  Buffer buffer = model_serialize.SerializeOpDesc(node1_op);
  EXPECT_TRUE(buffer.GetData() != nullptr);

  auto s_op = model_serialize.UnserializeOpDesc(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(s_op != nullptr);

  {
    EXPECT_EQ(s_op->GetName(), "node1");
    EXPECT_EQ(s_op->GetType(), "Conv2D");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 2);

    auto desc1 = s_input_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto desc2 = s_input_descs.at(1);
    EXPECT_EQ(desc2.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(desc2.GetDataType(), DT_FLOAT16);
    CompareShape(desc2.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc3 = s_output_descs.at(0);
    EXPECT_EQ(desc3.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc3.GetDataType(), DT_FLOAT);
    CompareShape(desc3.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    // node attrs
    GeAttrValue::BYTES node_val1;
    AttrUtils::GetBytes(s_op, "node_key1", node_val1);
    ASSERT_EQ(node_val1.GetSize(), 10);

    GeAttrValue::LIST_BYTES node_val2;
    AttrUtils::GetListBytes(s_op, "node_key2", node_val2);
    ASSERT_EQ(node_val2.size(), 2);
    ASSERT_EQ(node_val2[0].GetSize(), 20);
    ASSERT_EQ(node_val2[1].GetSize(), 30);

    GeAttrValue s_named_attrs;
    s_op->GetAttr("node_key3", s_named_attrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_named_attrs, named_attrs1));

    GeAttrValue s_list_named_attrs;
    s_op->GetAttr("node_key4", s_list_named_attrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_list_named_attrs, list_named_attrs));
  }
}

TEST(UtestGeModelSerialize, opdesc_as_attr_value) {
  // node1_op
  auto node1_op = std::make_shared<OpDesc>("node1", "Conv2D");
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  node1_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

  // Attr set
  node1_op->SetAttr("node_key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(Buffer(10)));
  node1_op->SetAttr("node_key2", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({Buffer(20), Buffer(30)}));
  auto named_attrs1 = GeAttrValue::CreateFrom<GeAttrValue::NAMED_ATTRS>(
      CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                   {"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                   {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}));

  node1_op->SetAttr("node_key3", std::move(named_attrs1));
  auto list_named_attrs = GeAttrValue::CreateFrom<GeAttrValue::LIST_NAMED_ATTRS>(
      {CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                    {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}}),
       CreateNamedAttrs("my_name2", {{"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                     {"float_val", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(345.345)}})});
  node1_op->SetAttr("node_key4", std::move(list_named_attrs));

  Model model;
  EXPECT_TRUE(AttrUtils::SetListOpDesc(&model, "my_key", vector<OpDescPtr>{node1_op}));
  EXPECT_TRUE(AttrUtils::SetListInt(&model, "my_key2", {123}));
  EXPECT_TRUE(AttrUtils::SetListBytes(&model, "my_key3", {Buffer(100)}));

  vector<OpDescPtr> op_list;
  EXPECT_FALSE(AttrUtils::GetListOpDesc(&model, "my_error_key", op_list));
  EXPECT_FALSE(AttrUtils::GetListOpDesc(&model, "my_key2", op_list));

  EXPECT_TRUE(AttrUtils::GetListOpDesc(&model, "my_key", op_list));

  ASSERT_TRUE(op_list.size() > 0);
  auto s_op = op_list[0];

  {
    EXPECT_EQ(s_op->GetName(), "node1");
    EXPECT_EQ(s_op->GetType(), "Conv2D");
    auto s_input_descs = s_op->GetAllInputsDesc();
    ASSERT_EQ(s_input_descs.size(), 2);

    auto desc1 = s_input_descs.at(0);
    EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
    CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    auto desc2 = s_input_descs.at(1);
    EXPECT_EQ(desc2.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(desc2.GetDataType(), DT_FLOAT16);
    CompareShape(desc2.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

    auto s_output_descs = s_op->GetAllOutputsDesc();
    ASSERT_EQ(s_output_descs.size(), 1);
    auto desc3 = s_output_descs.at(0);
    EXPECT_EQ(desc3.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(desc3.GetDataType(), DT_FLOAT);
    CompareShape(desc3.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

    // node attrs
    GeAttrValue::BYTES node_val1;
    AttrUtils::GetBytes(s_op, "node_key1", node_val1);
    ASSERT_EQ(node_val1.GetSize(), 10);

    GeAttrValue::LIST_BYTES node_val2;
    AttrUtils::GetListBytes(s_op, "node_key2", node_val2);
    ASSERT_EQ(node_val2.size(), 2);
    ASSERT_EQ(node_val2[0].GetSize(), 20);
    ASSERT_EQ(node_val2[1].GetSize(), 30);

    GeAttrValue s_named_attrs;
    s_op->GetAttr("node_key3", s_named_attrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_named_attrs, named_attrs1));

    GeAttrValue s_list_named_attrs;
    s_op->GetAttr("node_key4", s_list_named_attrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_list_named_attrs, list_named_attrs));
  }
}

TEST(UtestGeModelSerialize, test_sub_graph) {
  Model model("model_name", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);

    auto sub_compute_graph = std::make_shared<ComputeGraph>("sub_graph");
    // input
    auto sub_graph_input_op = std::make_shared<OpDesc>("sub_graph_test", "TestOp2");
    sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_graph_input = CreateNode(sub_graph_input_op, sub_compute_graph);

    AttrUtils::SetGraph(input_op, "sub_graph", sub_compute_graph);
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);
  ASSERT_GE(serialize.GetSerializeModelSize(model), 0);

  auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(model2.GetGraph().IsValid());
  auto graph2 = GraphUtils::GetComputeGraph(model2.GetGraph());
  EXPECT_EQ(graph2->GetName(), "graph_name");
  auto nodes2 = graph2->GetDirectNode();
  ASSERT_EQ(nodes2.size(), 1);
  auto node2 = nodes2.at(0);
  EXPECT_EQ(node2->GetName(), "test");
  auto node2_op = node2->GetOpDesc();
  EXPECT_EQ(node2_op->GetType(), "TestOp");
  auto node2_input_descs = node2_op->GetAllInputsDesc();
  ASSERT_EQ(node2_input_descs.size(), 1);
  auto node2_input_desc = node2_input_descs.at(0);

  ComputeGraphPtr sub_compute_graph2;
  ASSERT_TRUE(AttrUtils::GetGraph(node2_op, "sub_graph", sub_compute_graph2));
  EXPECT_EQ(sub_compute_graph2->GetName(), "sub_graph");
  auto sub_nodes2 = sub_compute_graph2->GetDirectNode();
  ASSERT_EQ(sub_nodes2.size(), 1);
  auto sub_node2 = sub_nodes2.at(0);
  EXPECT_EQ(sub_node2->GetName(), "sub_graph_test");
  ASSERT_EQ(sub_node2->GetAllInDataAnchors().size(), 1);
  auto sub_node_op2 = sub_node2->GetOpDesc();
  EXPECT_EQ(sub_node_op2->GetType(), "TestOp2");
  ASSERT_EQ(sub_node_op2->GetAllInputsDesc().size(), 1);
  auto sub_node2_input_desc = sub_node_op2->GetAllInputsDesc().at(0);
  EXPECT_EQ(sub_node2_input_desc.GetShape().GetDim(1), 32);
}

TEST(UtestGeModelSerialize, test_list_sub_graph) {
  Model model("model_name", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);

    auto sub_compute_graph1 = std::make_shared<ComputeGraph>("sub_graph1");
    // input
    auto sub_graph_input_op1 = std::make_shared<OpDesc>("sub_graph_test1", "TestOp2");
    sub_graph_input_op1->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_graph_input1 = CreateNode(sub_graph_input_op1, sub_compute_graph1);

    auto sub_compute_graph2 = std::make_shared<ComputeGraph>("sub_graph2");
    // input
    auto sub_graph_input_op2 = std::make_shared<OpDesc>("sub_graph_test2", "TestOp2");
    sub_graph_input_op2->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_graph_input2 = CreateNode(sub_graph_input_op2, sub_compute_graph2);

    AttrUtils::SetListGraph(input_op, "sub_graph", vector<ComputeGraphPtr>{sub_compute_graph1, sub_compute_graph2});
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);

  auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(model2.GetGraph().IsValid());
  auto graph2 = GraphUtils::GetComputeGraph(model2.GetGraph());
  EXPECT_EQ(graph2->GetName(), "graph_name");
  auto nodes2 = graph2->GetDirectNode();
  ASSERT_EQ(nodes2.size(), 1);
  auto node2 = nodes2.at(0);
  auto node2_op = node2->GetOpDesc();

  vector<ComputeGraphPtr> list_sub_compute_graph;
  ASSERT_TRUE(AttrUtils::GetListGraph(node2_op, "sub_graph", list_sub_compute_graph));
  ASSERT_EQ(list_sub_compute_graph.size(), 2);

  EXPECT_EQ(list_sub_compute_graph[0]->GetName(), "sub_graph1");
  EXPECT_EQ(list_sub_compute_graph[1]->GetName(), "sub_graph2");

  auto sub_nodes21 = list_sub_compute_graph[0]->GetDirectNode();
  ASSERT_EQ(sub_nodes21.size(), 1);
  auto sub_node21 = sub_nodes21.at(0);
  EXPECT_EQ(sub_node21->GetName(), "sub_graph_test1");

  auto sub_nodes22 = list_sub_compute_graph[1]->GetDirectNode();
  ASSERT_EQ(sub_nodes22.size(), 1);
  auto sub_node22 = sub_nodes22.at(0);
  EXPECT_EQ(sub_node22->GetName(), "sub_graph_test2");
}

TEST(UtestGeModelSerialize, test_format) {
  Model model("model_name", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NHWC, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_ND, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NC1HWC0, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_Z, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NC1C0HWPAD, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NHWC1C0, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FSR_NCHW, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_DECONV, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_BN_WEIGHT, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_CHWN, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FILTER_HWCK, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_Z_C04, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(compute_graph));
  }
  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);
  auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(model2.GetGraph().IsValid());

  auto graph = model2.GetGraph();
  ASSERT_TRUE(GraphUtils::GetComputeGraph(graph) != nullptr);
  ASSERT_EQ(GraphUtils::GetComputeGraph(graph)->GetDirectNode().size(), 1);

  auto op = GraphUtils::GetComputeGraph(graph)->GetDirectNode().at(0)->GetOpDesc();
  auto input_descs = op->GetAllInputsDesc();
  ASSERT_EQ(input_descs.size(), 13);
  EXPECT_EQ(input_descs.at(0).GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(input_descs.at(1).GetFormat(), FORMAT_NHWC);
  EXPECT_EQ(input_descs.at(2).GetFormat(), FORMAT_ND);
  EXPECT_EQ(input_descs.at(3).GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(input_descs.at(4).GetFormat(), FORMAT_FRACTAL_Z);
  EXPECT_EQ(input_descs.at(5).GetFormat(), FORMAT_NC1C0HWPAD);
  EXPECT_EQ(input_descs.at(6).GetFormat(), FORMAT_NHWC1C0);
  EXPECT_EQ(input_descs.at(7).GetFormat(), FORMAT_FSR_NCHW);
  EXPECT_EQ(input_descs.at(8).GetFormat(), FORMAT_FRACTAL_DECONV);
  EXPECT_EQ(input_descs.at(9).GetFormat(), FORMAT_BN_WEIGHT);
  EXPECT_EQ(input_descs.at(10).GetFormat(), FORMAT_CHWN);
  EXPECT_EQ(input_descs.at(11).GetFormat(), FORMAT_FILTER_HWCK);
  EXPECT_EQ(input_descs.at(12).GetFormat(), FORMAT_FRACTAL_Z_C04);
}

TEST(UtestGeModelSerialize, test_control_edge) {
  Model model("model_name", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    // sink
    auto sink_op = std::make_shared<OpDesc>("test2", "Sink");
    auto sink = CreateNode(sink_op, compute_graph);
    LinkEdge(sink, -1, input, -1);

    // sink2
    auto sink_op2 = std::make_shared<OpDesc>("test3", "Sink");
    auto sink2 = CreateNode(sink_op2, compute_graph);
    LinkEdge(sink2, -1, input, -1);

    // dest
    auto dest_op = std::make_shared<OpDesc>("test4", "Dest");
    auto dest = CreateNode(dest_op, compute_graph);
    LinkEdge(input, -1, dest, -1);

    compute_graph->AddInputNode(sink);
    compute_graph->AddInputNode(sink2);
    compute_graph->AddOutputNode(dest);

    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);
  }
  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  EXPECT_GE(buffer.GetSize(), 0);

  auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(model2.GetGraph().IsValid());
  auto graph = GraphUtils::GetComputeGraph(model2.GetGraph());
  EXPECT_EQ(graph->GetName(), "graph_name");
  auto nodes = graph->GetDirectNode();
  ASSERT_EQ(nodes.size(), 4);

  auto node1 = nodes.at(0);
  auto sink = nodes.at(1);
  auto sink2 = nodes.at(2);
  auto dest = nodes.at(3);
  EXPECT_EQ(node1->GetName(), "test");
  EXPECT_EQ(sink->GetName(), "test2");
  ASSERT_EQ(node1->GetAllInDataAnchors().size(), 1);
  auto anchor1 = node1->GetAllInDataAnchors().at(0);
  EXPECT_EQ(anchor1->GetPeerAnchors().size(), 0);

  auto contorl_in_anchor1 = node1->GetInControlAnchor();
  ASSERT_EQ(contorl_in_anchor1->GetPeerAnchors().size(), 2);

  EXPECT_EQ(contorl_in_anchor1->GetPeerAnchors().at(0)->GetOwnerNode(), sink);
  EXPECT_EQ(contorl_in_anchor1->GetPeerAnchors().at(1)->GetOwnerNode(), sink2);

  auto contorl_out_anchor1 = node1->GetOutControlAnchor();
  ASSERT_EQ(contorl_out_anchor1->GetPeerAnchors().size(), 1);
  EXPECT_EQ(contorl_out_anchor1->GetPeerAnchors().at(0)->GetOwnerNode(), dest);

  auto input_nodes = graph->GetInputNodes();
  ASSERT_EQ(input_nodes.size(), 2);
  EXPECT_EQ(input_nodes.at(0), sink);
  EXPECT_EQ(input_nodes.at(1), sink2);

  auto output_nodes = graph->GetOutputNodes();
  ASSERT_EQ(output_nodes.size(), 1);
  EXPECT_EQ(output_nodes.at(0), dest);
}

TEST(UtestGeModelSerialize, test_serialize_graph) {
  auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
  {
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    // sink
    auto sink_op = std::make_shared<OpDesc>("test2", "Sink");
    auto sink = CreateNode(sink_op, compute_graph);
    LinkEdge(sink, -1, input, -1);

    // sink2
    auto sink_op2 = std::make_shared<OpDesc>("test3", "Sink");
    auto sink2 = CreateNode(sink_op2, compute_graph);
    LinkEdge(sink2, -1, input, -1);

    // dest
    auto dest_op = std::make_shared<OpDesc>("test4", "Dest");
    auto dest = CreateNode(dest_op, compute_graph);
    LinkEdge(input, -1, dest, -1);

    compute_graph->AddInputNode(sink);
    compute_graph->AddInputNode(sink2);
    compute_graph->AddOutputNode(dest);
  }
  ModelSerialize serialize;
  auto buffer = serialize.SerializeGraph(compute_graph);
  EXPECT_GE(buffer.GetSize(), 0);

  auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
  ASSERT_TRUE(graph != nullptr);
  EXPECT_EQ(graph->GetName(), "graph_name");
  auto nodes = graph->GetDirectNode();
  ASSERT_EQ(nodes.size(), 4);

  auto node1 = nodes.at(0);
  auto sink = nodes.at(1);
  auto sink2 = nodes.at(2);
  auto dest = nodes.at(3);
  EXPECT_EQ(node1->GetName(), "test");
  EXPECT_EQ(sink->GetName(), "test2");
  ASSERT_EQ(node1->GetAllInDataAnchors().size(), 1);
  auto anchor1 = node1->GetAllInDataAnchors().at(0);
  EXPECT_EQ(anchor1->GetPeerAnchors().size(), 0);

  auto contorl_in_anchor1 = node1->GetInControlAnchor();
  ASSERT_EQ(contorl_in_anchor1->GetPeerAnchors().size(), 2);

  EXPECT_EQ(contorl_in_anchor1->GetPeerAnchors().at(0)->GetOwnerNode(), sink);
  EXPECT_EQ(contorl_in_anchor1->GetPeerAnchors().at(1)->GetOwnerNode(), sink2);

  auto contorl_out_anchor1 = node1->GetOutControlAnchor();
  ASSERT_EQ(contorl_out_anchor1->GetPeerAnchors().size(), 1);
  EXPECT_EQ(contorl_out_anchor1->GetPeerAnchors().at(0)->GetOwnerNode(), dest);

  auto input_nodes = graph->GetInputNodes();
  ASSERT_EQ(input_nodes.size(), 2);
  EXPECT_EQ(input_nodes.at(0), sink);
  EXPECT_EQ(input_nodes.at(1), sink2);

  auto output_nodes = graph->GetOutputNodes();
  ASSERT_EQ(output_nodes.size(), 1);
  EXPECT_EQ(output_nodes.at(0), dest);
}

TEST(UtestGeModelSerialize, test_invalid_model) {
  {  // empty graph
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_EQ(buffer.GetSize(), 0);
  }
}

TEST(UtestGeModelSerialize, test_invalid_graph) {
  {  // empty graph

    ComputeGraphPtr graph = nullptr;

    ModelSerialize serialize;
    auto buffer = serialize.SerializeGraph(graph);
    EXPECT_EQ(buffer.GetSize(), 0);
  }
}

TEST(UtestGeModelSerialize, test_invalid_opdesc) {
  {  // empty OpDesc
    OpDescPtr op_desc = nullptr;
    ModelSerialize serialize;
    auto buffer = serialize.SerializeOpDesc(op_desc);
    EXPECT_EQ(buffer.GetSize(), 0);
  }
}

TEST(UtestGeModelSerialize, test_invalid_tensor_desc) {
  {  // valid test
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // invalid format
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_RESERVED, DT_FLOAT));  // invalid format
    auto input = CreateNode(input_op, compute_graph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_GE(buffer.GetSize(), 0);
    auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    ASSERT_TRUE(model2.IsValid());
    auto graph_new = GraphUtils::GetComputeGraph(model2.GetGraph());
    ASSERT_TRUE(graph_new != nullptr);
    auto node_list_new = graph_new->GetAllNodes();
    ASSERT_EQ(node_list_new.size(), 1);
    auto opdesc_new = node_list_new.at(0)->GetOpDesc();
    ASSERT_TRUE(opdesc_new != nullptr);
    auto output_desc_list_new = opdesc_new->GetAllOutputsDesc();
    ASSERT_EQ(output_desc_list_new.size(), 1);
    auto output_desc_new = output_desc_list_new.at(0);
    EXPECT_EQ(output_desc_new.GetDataType(), DT_FLOAT);
    EXPECT_EQ(output_desc_new.GetFormat(), FORMAT_RESERVED);
  }
  {  // DT_UNDEFINED datatype
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_UNDEFINED));
    auto input = CreateNode(input_op, compute_graph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_GE(buffer.GetSize(), 0);
    auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    ASSERT_TRUE(model2.IsValid());
    auto graph_new = GraphUtils::GetComputeGraph(model2.GetGraph());
    ASSERT_TRUE(graph_new != nullptr);
    auto node_list_new = graph_new->GetAllNodes();
    ASSERT_EQ(node_list_new.size(), 1);
    auto opdesc_new = node_list_new.at(0)->GetOpDesc();
    ASSERT_TRUE(opdesc_new != nullptr);
    auto output_desc_list_new = opdesc_new->GetAllOutputsDesc();
    ASSERT_EQ(output_desc_list_new.size(), 1);
    auto output_desc_new = output_desc_list_new.at(0);
    EXPECT_EQ(output_desc_new.GetDataType(), DT_UNDEFINED);
    EXPECT_EQ(output_desc_new.GetFormat(), FORMAT_NCHW);
  }
}

TEST(UtestGeModelSerialize, test_invalid_attrs) {
  {  // valid test
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs named_attrs;
    named_attrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::INT>(10));
    AttrUtils::SetNamedAttrs(input_op, "key", named_attrs);

    auto input = CreateNode(input_op, compute_graph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // none type
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs named_attrs;
    EXPECT_EQ(named_attrs.SetAttr("key1", GeAttrValue()), GRAPH_FAILED);
  }
  {  // bytes attr len is 0
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs named_attrs;
    named_attrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(GeAttrValue::BYTES(0)));
    AttrUtils::SetNamedAttrs(input_op, "key", named_attrs);

    auto input = CreateNode(input_op, compute_graph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);

    auto model2 = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    EXPECT_TRUE(model2.IsValid());
  }
  {  // invalid list bytes attr
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs named_attrs;
    named_attrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({GeAttrValue::BYTES(0)}));
    AttrUtils::SetNamedAttrs(input_op, "key", named_attrs);

    auto input = CreateNode(input_op, compute_graph);
    Graph graph = GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // invalid graph attr
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs named_attrs;
    EXPECT_EQ(named_attrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::GRAPH>(nullptr)), GRAPH_FAILED);
    GeAttrValue value;
    EXPECT_EQ(named_attrs.GetAttr("key1", value), GRAPH_FAILED);
    EXPECT_TRUE(value.IsEmpty());
  }
  {  // invalid list graph attr
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    GeAttrValue::NamedAttrs named_attrs;
    EXPECT_EQ(named_attrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::LIST_GRAPH>({nullptr})), GRAPH_FAILED);
    GeAttrValue value;
    EXPECT_EQ(named_attrs.GetAttr("key1", value), GRAPH_FAILED);
    EXPECT_TRUE(value.IsEmpty());
  }
}

TEST(UtestGeModelSerialize, test_model_serialize_imp_invalid_param) {
  ModelSerializeImp imp;
  EXPECT_FALSE(imp.SerializeModel(Model(), nullptr));
  EXPECT_FALSE(imp.SerializeGraph(nullptr, nullptr));
  EXPECT_FALSE(imp.SerializeNode(nullptr, nullptr));
  EXPECT_FALSE(imp.SerializeOpDesc(nullptr, nullptr));

  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(std::make_shared<OpDesc>());
  node->impl_->op_ = nullptr;
  ge::proto::ModelDef model_def;
  Model model;
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(graph));
  EXPECT_FALSE(imp.SerializeModel(model, &model_def));

  ModelSerialize serialize;
  EXPECT_EQ(serialize.GetSerializeModelSize(model), 0);
}

TEST(UtestGeModelSerialize, test_parse_node_false) {
  ModelSerializeImp imp;
  string node_index = "invalid_index";
  string node_name = "name";
  int32_t index = 1;
  EXPECT_EQ(imp.ParseNodeIndex(node_index, node_name, index), false);
}

TEST(UtestGeModelSerialize, test_invalid_tensor) {
  ModelSerializeImp imp;
  EXPECT_EQ(imp.SerializeTensor(nullptr, nullptr), false);

  try {
    ConstGeTensorPtr tensor_ptr = std::make_shared<GeTensor>();
    EXPECT_EQ(imp.SerializeTensor(tensor_ptr, nullptr), false);
  } catch (...) {
  }
}

TEST(UTEST_ge_model_unserialize, test_invalid_tensor) {
  ModelSerializeImp imp;
  EXPECT_EQ(imp.SerializeTensor(nullptr, nullptr), false);

  try {
    ConstGeTensorPtr tensor_ptr = std::make_shared<GeTensor>();
    EXPECT_EQ(imp.SerializeTensor(tensor_ptr, nullptr), false);
  } catch (...) {
  }
}

TEST(UTEST_ge_model_unserialize, test_invalid_TensorDesc) {
  {  // valid
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.mutable_attr();

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_td();
    tensor_desc_attr->set_layout("NCHW");
    tensor_desc_attr->set_dtype(ge::proto::DataType::DT_INT8);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
  }
  {  // invalid layout
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.mutable_attr();

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_td();
    tensor_desc_attr->set_layout("InvalidLayout");
    tensor_desc_attr->set_dtype(ge::proto::DataType::DT_INT8);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    GeTensorDesc tensor_desc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(model, "key1", tensor_desc));
    EXPECT_EQ(tensor_desc.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc.GetDataType(), DT_INT8);
  }
  {  // invalid datatype
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.mutable_attr();

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_td();  // tensor desc
    tensor_desc_attr->set_layout("NHWC");
    tensor_desc_attr->set_dtype((ge::proto::DataType)100);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    GeTensorDesc tensor_desc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(model, "key1", tensor_desc));
    EXPECT_EQ(tensor_desc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(tensor_desc.GetDataType(), DT_UNDEFINED);
  }
  {  // invalid datatype
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.mutable_attr();

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_t()->mutable_desc();  // tensor
    tensor_desc_attr->set_layout("NHWC");
    tensor_desc_attr->set_dtype((ge::proto::DataType)100);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(model, "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensor_desc = tensor->GetTensorDesc();
    EXPECT_EQ(tensor_desc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(tensor_desc.GetDataType(), DT_UNDEFINED);
  }
  {  // invalid attrmap
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->mutable_attr();  // graph attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_t()->mutable_desc();  // tensor
    tensor_desc_attr->set_layout("NCHW");
    tensor_desc_attr->set_dtype(ge::proto::DataType::DT_INT8);
    auto attrs1 = tensor_desc_attr->mutable_attr();
    auto attr1 = (*attrs1)["key2"];  // empty attr

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(graph, "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensor_desc = tensor->GetTensorDesc();
    GeAttrValue attr_value;
    EXPECT_EQ(tensor_desc.GetAttr("key2", attr_value), GRAPH_SUCCESS);
    EXPECT_EQ(attr_value.GetValueType(), GeAttrValue::VT_NONE);
  }
  {  // invalid attrmap2
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_t()->mutable_desc();  // tensor
    tensor_desc_attr->set_layout("NCHW");
    tensor_desc_attr->set_dtype(ge::proto::DataType::DT_INT8);
    auto attrs1 = tensor_desc_attr->mutable_attr();
    auto attr1 = (*attrs1)["key2"].mutable_list();  // empty list attr

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensor_desc = tensor->GetTensorDesc();
    GeAttrValue attr_value;
    EXPECT_EQ(tensor_desc.GetAttr("key2", attr_value), GRAPH_SUCCESS);
    EXPECT_EQ(attr_value.GetValueType(), GeAttrValue::VT_NONE);
  }
}
TEST(UTEST_ge_model_unserialize, test_invalid_attr) {
  {  // invalid graph
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto graph_attr = attr_def->mutable_g();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ComputeGraphPtr graph_attr_new;
    EXPECT_TRUE(AttrUtils::GetGraph(nodes.at(0)->GetOpDesc(), "key1", graph_attr_new));
    ASSERT_TRUE(graph_attr_new != nullptr);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(graph_attr_new, "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list graph
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_GRAPH);
    auto graph_attr = attr_def->mutable_list()->add_g();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<ComputeGraphPtr> graph_list_attr;
    EXPECT_TRUE(AttrUtils::GetListGraph(nodes.at(0)->GetOpDesc(), "key1", graph_list_attr));
    ASSERT_EQ(graph_list_attr.size(), 1);
    ASSERT_TRUE(graph_list_attr[0] != nullptr);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(graph_list_attr[0], "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid named_attrs
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto graph_attr = attr_def->mutable_func();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    GeAttrValue::NAMED_ATTRS named_attrs;
    EXPECT_TRUE(AttrUtils::GetNamedAttrs(nodes.at(0)->GetOpDesc(), "key1", named_attrs));
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(named_attrs, "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list named_attrs
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_NAMED_ATTRS);
    auto graph_attr = attr_def->mutable_list()->add_na();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    GeAttrValue::LIST_NAMED_ATTRS named_attrs;
    EXPECT_TRUE(AttrUtils::GetListNamedAttrs(nodes.at(0)->GetOpDesc(), "key1", named_attrs));
    ASSERT_EQ(named_attrs.size(), 1);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(named_attrs.at(0), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid tensor_desc
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto graph_attr = attr_def->mutable_td();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    GeTensorDesc tensor_desc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(nodes.at(0)->GetOpDesc(), "key1", tensor_desc));
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor_desc, "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensor_desc
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR_DESC);
    auto graph_attr = attr_def->mutable_list()->add_td();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<GeTensorDesc> tensor_desc;
    EXPECT_TRUE(AttrUtils::GetListTensorDesc(nodes.at(0)->GetOpDesc(), "key1", tensor_desc));
    ASSERT_EQ(tensor_desc.size(), 1);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor_desc.at(0), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid tensor
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto graph_attr = attr_def->mutable_t()->mutable_desc();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor->GetTensorDesc(), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensor
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR);
    auto graph_attr = attr_def->mutable_list()->add_t()->mutable_desc();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = GraphUtils::GetComputeGraph(model.GetGraph());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<ConstGeTensorPtr> tensor;
    EXPECT_TRUE(AttrUtils::GetListTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    ASSERT_EQ(tensor.size(), 1);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor.at(0)->GetTensorDesc(), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensor
    ge::proto::GraphDef graph_def;
    auto attrs = graph_def.add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR);
    auto graph_attr = attr_def->mutable_list()->add_t()->mutable_desc();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Buffer buffer(graph_def.ByteSizeLong());
    graph_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<ConstGeTensorPtr> tensor;
    EXPECT_TRUE(AttrUtils::GetListTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    ASSERT_EQ(tensor.size(), 1);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor.at(0)->GetTensorDesc(), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
}

TEST(UTEST_ge_model_unserialize, test_invalid_input_output) {
  // model invalid node input
  {
    // ge::proto::ModelDef model_def;
    // auto op_def = model_def.add_graph()->add_op();  // node attr
    // op_def->add_input("invalidNodeName:0");

    // Buffer buffer(model_def.ByteSizeLong());
    // model_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    // ModelSerialize serialize;
    // auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    // EXPECT_FALSE(model.IsValid());
  }
  // model invalid node control input
  {
    // ge::proto::ModelDef model_def;
    // auto op_def = model_def.add_graph()->add_op();  // node attr
    // op_def->add_input("invalidNodeName:-1");

    // Buffer buffer(model_def.ByteSizeLong());
    // model_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    // ModelSerialize serialize;
    // auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    // EXPECT_FALSE(model.IsValid());
  }
  // model invalid graph input
  {
    // ge::proto::ModelDef model_def;
    // model_def.add_graph()->add_input("invalidNodeName:0");

    // Buffer buffer(model_def.ByteSizeLong());
    // model_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    // ModelSerialize serialize;
    // auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    // EXPECT_FALSE(model.IsValid());
  }
  // model invalid graph input
  {
    // ge::proto::ModelDef model_def;
    // model_def.add_graph()->add_output("invalidNodeName:0");

    // Buffer buffer(model_def.ByteSizeLong());
    // model_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    // ModelSerialize serialize;
    // auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    // EXPECT_FALSE(model.IsValid());
  }
  // graph invalid node input
  {
    ge::proto::GraphDef graph_def;
    auto op_def = graph_def.add_op();  // node attr
    op_def->add_input("invalidNodeName:0");

    Buffer buffer(graph_def.ByteSizeLong());
    graph_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(graph != nullptr);
  }
  // graph invalid node control input
  {
    ge::proto::GraphDef graph_def;
    auto op_def = graph_def.add_op();  // node attr
    op_def->add_input("invalidNodeName:-1");

    Buffer buffer(graph_def.ByteSizeLong());
    graph_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(graph != nullptr);
  }
  // graph invalid graph input
  {
    ge::proto::GraphDef graph_def;
    graph_def.add_input("invalidNodeName:0");

    Buffer buffer(graph_def.ByteSizeLong());
    graph_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(graph != nullptr);
  }
  // graph invalid graph output
  {
    ge::proto::GraphDef graph_def;
    graph_def.add_output("invalidNodeName:0");

    Buffer buffer(graph_def.ByteSizeLong());
    graph_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph(buffer.GetData(), buffer.GetSize());
    EXPECT_FALSE(graph != nullptr);
  }
  // model invalid node input anchor
  {
    // ge::proto::ModelDef model_def;
    // auto graph_def = model_def.add_graph();
    // auto node_def1 = graph_def->add_op();  // node attr
    // node_def1->set_name("node1");

    // auto node_def2 = graph_def->add_op();  // node attr
    // node_def2->add_input("node1:0");

    // Buffer buffer(model_def.ByteSizeLong());
    // model_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));

    // ModelSerialize serialize;
    // auto model = serialize.UnserializeModel(buffer.GetData(), buffer.GetSize());
    // EXPECT_FALSE(model.IsValid());
  }
}

TEST(UTEST_ge_model_unserialize, test_invalid_CodeBuffer) {
  {
    char buffer[100] = "sdfasf";
    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph((uint8_t *)buffer, 100);
    EXPECT_FALSE(graph != nullptr);
  }
  {
    char buffer[100] = "sdfasf";
    ModelSerialize serialize;
    auto model = serialize.UnserializeModel((uint8_t *)buffer, 100);
    EXPECT_FALSE(model.IsValid());
  }
  {
    char buffer[100] = "sdfasf";
    ModelSerialize serialize;
    auto op_desc = serialize.UnserializeOpDesc((uint8_t *)buffer, 100);
    EXPECT_FALSE(op_desc != nullptr);
  }
  {
    ModelSerialize serialize;
    auto graph = serialize.UnserializeGraph((uint8_t *)nullptr, 100);
    EXPECT_FALSE(graph != nullptr);
  }
  {
    ModelSerialize serialize;
    auto model = serialize.UnserializeModel((uint8_t *)nullptr, 100);
    EXPECT_FALSE(model.IsValid());
  }
  {
    ModelSerialize serialize;
    auto op_desc = serialize.UnserializeOpDesc((uint8_t *)nullptr, 100);
    EXPECT_FALSE(op_desc != nullptr);
  }
}
