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
#include "graph/utils/attr_utils.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph_builder_utils.h"
#include "test_std_structs.h"

namespace ge {
namespace {
std::unique_ptr<float> GetRandomFloat(std::initializer_list<int64_t> shape) {
  int64_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  auto data = std::unique_ptr<float>(new float[size]);
  for (int64_t i = 0; i < size; ++i) {
    data.get()[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  return data;
}

GeTensorPtr CreateTensor_8_3_224_224(float *tensor_data) {
  auto tensor = std::make_shared<GeTensor>();
  tensor->SetData(reinterpret_cast<uint8_t *>(tensor_data), 8*3*224*224*sizeof(float));
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({8, 3, 224, 224})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({8, 3, 224, 224})));
  td.SetFormat(FORMAT_NCHW);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT);
  td.SetOriginDataType(DT_FLOAT);
  AttrUtils::SetStr(&td, "bcd", "Hello world");
  tensor->SetTensorDesc(td);
  return tensor;
}

void ExpectTensorEqual_8_3_224_224(ConstGeTensorPtr out_tensor, float *tensor_data) {
  EXPECT_NE(const_cast<uint8_t *>(out_tensor->GetData().data()), reinterpret_cast<uint8_t*>(tensor_data));
  EXPECT_EQ(out_tensor->GetData().size(), 8*3*224*224*sizeof(float));
  for (size_t i = 0; i < 8*3*224*224; ++i) {
    EXPECT_FLOAT_EQ(reinterpret_cast<const float *>(out_tensor->GetData().data())[i], tensor_data[i]);
  }
  EXPECT_EQ(out_tensor->GetTensorDesc().GetShape().GetDims(), std::vector<int64_t>({8,3,224,224}));
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginShape().GetDims(), std::vector<int64_t>({8,3,224,224}));
  EXPECT_EQ(out_tensor->GetTensorDesc().GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetDataType(), DT_FLOAT);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginDataType(), DT_FLOAT);
  std::string s;
  EXPECT_TRUE(AttrUtils::GetStr(&out_tensor->GetTensorDesc(), "bcd", s));
  EXPECT_EQ(s, "Hello world");
}

GeTensorPtr CreateTensor_5d_8_3_224_224(float *tensor_data) {
  auto tensor = std::make_shared<GeTensor>();
  tensor->SetData(reinterpret_cast<uint8_t *>(tensor_data), 8*1*224*224*16*sizeof(float));
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({8, 1, 224, 224, 16})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({8, 3, 224, 224})));
  td.SetFormat(FORMAT_NC1HWC0);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT);
  td.SetOriginDataType(DT_FLOAT);
  AttrUtils::SetStr(&td, "bcd", "Hello world");
  tensor->SetTensorDesc(td);
  return tensor;
}

void ExpectTensorEqual_5d_8_3_224_224(ConstGeTensorPtr out_tensor, float *tensor_data) {
  EXPECT_NE(const_cast<uint8_t *>(out_tensor->GetData().data()), reinterpret_cast<uint8_t*>(tensor_data));
  EXPECT_EQ(out_tensor->GetData().size(), 8*1*224*224*16*sizeof(float));
  for (size_t i = 0; i < 8*1*224*224*16; ++i) {
    EXPECT_FLOAT_EQ(reinterpret_cast<const float *>(out_tensor->GetData().data())[i], tensor_data[i]);
  }
  EXPECT_EQ(out_tensor->GetTensorDesc().GetShape().GetDims(), std::vector<int64_t>({8,1,224,224,16}));
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginShape().GetDims(), std::vector<int64_t>({8,3,224,224}));
  EXPECT_EQ(out_tensor->GetTensorDesc().GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetDataType(), DT_FLOAT);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginDataType(), DT_FLOAT);
  std::string s;
  EXPECT_TRUE(AttrUtils::GetStr(&out_tensor->GetTensorDesc(), "bcd", s));
  EXPECT_EQ(s, "Hello world");
}

ComputeGraphPtr CreateGraph_1_1_224_224(float *tensor_data) {
  ut::GraphBuilder builder("graph1");
  auto data1 = builder.AddNode("data1", "Data", {}, {"y"});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto const1 = builder.AddNode("const1", "Const", {}, {"y"});
  GeTensorDesc const1_td;
  const1_td.SetShape(GeShape({1,1,224,224}));
  const1_td.SetOriginShape(GeShape({1,1,224,224}));
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

bool ExpectConnected(const NodePtr &src, int src_index, const NodePtr &dst, int dst_index) {
  AnchorPtr src_anchor, dst_anchor;
  if (src_index >= 0 && dst_index >= 0) {
    src_anchor = src->GetOutDataAnchor(src_index);
    dst_anchor = dst->GetInDataAnchor(dst_index);
  } else if (src_index < 0 && dst_index < 0) {
    src_anchor = src->GetOutControlAnchor();
    dst_anchor = dst->GetInControlAnchor();
  } else {
    return false;
  }

  for (auto &peer_anchor : dst_anchor->GetPeerAnchors()) {
    if (src_anchor == peer_anchor) {
      return true;
    }
  }
  return false;
}

void ExpectEqGraph_1_1_224_224(const ConstComputeGraphPtr &graph, float *tensor_data) {
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
  auto data1 = graph->FindNode("data1");
  auto const1 = graph->FindNode("const1");
  auto add1 = graph->FindNode("add1");
  auto netoutput1 = graph->FindNode("NetOutputNode");
  EXPECT_NE(data1, nullptr);
  EXPECT_NE(const1, nullptr);
  EXPECT_NE(add1, nullptr);
  EXPECT_NE(netoutput1, nullptr);

  int data_index = 10;
  EXPECT_TRUE(AttrUtils::GetInt(data1->GetOpDesc(), "index", data_index));
  EXPECT_EQ(data_index, 0);

  EXPECT_EQ(data1->GetOpDesc()->GetName(), "data1");
  EXPECT_EQ(data1->GetType(), "Data");
  EXPECT_EQ(data1->GetOpDesc()->GetInputsSize(), 0);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputsSize(), 1);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetDataType(), DT_FLOAT);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetOriginDataType(), DT_FLOAT);

  EXPECT_EQ(const1->GetOpDesc()->GetName(), "const1");
  EXPECT_EQ(const1->GetType(), "Const");
  EXPECT_EQ(const1->GetOpDesc()->GetInputsSize(), 0);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputsSize(), 1);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetDataType(), DT_FLOAT);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetOriginDataType(), DT_FLOAT);

  ConstGeTensorPtr tensor;
  EXPECT_TRUE(AttrUtils::GetTensor(const1->GetOpDesc(), "value", tensor));
  EXPECT_EQ(tensor->GetTensorDesc().GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(tensor->GetTensorDesc().GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(tensor->GetTensorDesc().GetShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(tensor->GetTensorDesc().GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(tensor->GetTensorDesc().GetDataType(), DT_FLOAT);
  EXPECT_EQ(tensor->GetTensorDesc().GetOriginDataType(), DT_FLOAT);
  for (size_t i = 0; i < 224*224; ++i) {
    EXPECT_FLOAT_EQ(reinterpret_cast<const float *>(tensor->GetData().data())[i], tensor_data[i]);
  }


  EXPECT_EQ(add1->GetOpDesc()->GetName(), "add1");
  EXPECT_EQ(add1->GetType(), "Add");
  EXPECT_EQ(add1->GetOpDesc()->GetInputsSize(), 2);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputsSize(), 1);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetOriginDataType(), DT_FLOAT);

  EXPECT_EQ(netoutput1->GetOpDesc()->GetName(), "NetOutputNode");
  EXPECT_EQ(netoutput1->GetType(), "NetOutput");
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputsSize(), 1);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetOutputsSize(), 0);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetDataType(), DT_FLOAT);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetOriginDataType(), DT_FLOAT);

  EXPECT_EQ(data1->GetOutNodes().size(), 1);
  EXPECT_TRUE(ExpectConnected(data1, 0, add1, 0));
  EXPECT_EQ(const1->GetOutNodes().size(), 1);
  EXPECT_TRUE(ExpectConnected(const1, 0, add1, 1));
  EXPECT_EQ(add1->GetOutNodes().size(), 1);
  EXPECT_TRUE(ExpectConnected(add1, 0, netoutput1, 0));
}

ComputeGraphPtr CreateGraph_5d_1_1_224_224(float *tensor_data) {
  ut::GraphBuilder builder("graph1");
  auto data1 = builder.AddNode("data1", "Data", {}, {"y"});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  data1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1,1,224,224,16}));

  auto const1 = builder.AddNode("const1", "Const", {}, {"y"});
  const1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  const1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1,1,224,224,16}));
  GeTensorDesc const1_td;
  const1_td.SetShape(GeShape({1,1,224,224}));
  const1_td.SetOriginShape(GeShape({1,1,224,224}));
  const1_td.SetFormat(FORMAT_NCHW);
  const1_td.SetOriginFormat(FORMAT_NCHW);
  const1_td.SetDataType(DT_FLOAT);
  const1_td.SetOriginDataType(DT_FLOAT);
  GeTensor tensor(const1_td);
  tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data), sizeof(float) * 224 * 224);
  AttrUtils::SetTensor(const1->GetOpDesc(), "value", tensor);

  auto add1 = builder.AddNode("add1", "Add", {"x1", "x2"}, {"y"});
  add1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  add1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1,1,224,224,16}));
  add1->GetOpDesc()->MutableInputDesc(1)->SetFormat(FORMAT_NC1HWC0);
  add1->GetOpDesc()->MutableInputDesc(1)->SetShape(GeShape({1,1,224,224,16}));
  add1->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  add1->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1,1,224,224,16}));
  auto netoutput1 = builder.AddNode("NetOutputNode", "NetOutput", {"x"}, {});
  netoutput1->GetOpDesc()->MutableInputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  netoutput1->GetOpDesc()->MutableInputDesc(0)->SetShape(GeShape({1,1,224,224,16}));

  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, netoutput1, 0);

  return builder.GetGraph();
}

void ExpectEqGraph_5d_1_1_224_224(const ConstComputeGraphPtr &graph, float *tensor_data) {
  EXPECT_EQ(graph->GetAllNodesSize(), 4);
  auto data1 = graph->FindNode("data1");
  auto const1 = graph->FindNode("const1");
  auto add1 = graph->FindNode("add1");
  auto netoutput1 = graph->FindNode("NetOutputNode");
  EXPECT_NE(data1, nullptr);
  EXPECT_NE(const1, nullptr);
  EXPECT_NE(add1, nullptr);
  EXPECT_NE(netoutput1, nullptr);
  /*  todo 属性当前不支持序列化，支持序列化后，放开校验
  int data_index = 10;
  EXPECT_TRUE(AttrUtils::GetInt(data1->GetOpDesc(), "index", data_index));
  EXPECT_EQ(data_index, 0);
*/
  EXPECT_EQ(data1->GetOpDesc()->GetName(), "data1");
  EXPECT_EQ(data1->GetType(), "Data");
  EXPECT_EQ(data1->GetOpDesc()->GetInputsSize(), 0);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputsSize(), 1);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetShape(), GeShape({1,1,224,224,16}));
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetDataType(), DT_FLOAT);
  EXPECT_EQ(data1->GetOpDesc()->GetOutputDesc("y").GetOriginDataType(), DT_FLOAT);

  EXPECT_EQ(const1->GetOpDesc()->GetName(), "const1");
  EXPECT_EQ(const1->GetType(), "Const");
  EXPECT_EQ(const1->GetOpDesc()->GetInputsSize(), 0);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputsSize(), 1);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetShape(), GeShape({1,1,224,224,16}));
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetDataType(), DT_FLOAT);
  EXPECT_EQ(const1->GetOpDesc()->GetOutputDesc("y").GetOriginDataType(), DT_FLOAT);
//
//  ConstGeTensorPtr tensor;
//  EXPECT_TRUE(AttrUtils::GetTensor(const1->GetOpDesc(), "value", tensor));
//  EXPECT_EQ(tensor->GetTensorDesc().GetFormat(), FORMAT_NCHW);
//  EXPECT_EQ(tensor->GetTensorDesc().GetOriginFormat(), FORMAT_NCHW);
//  EXPECT_EQ(tensor->GetTensorDesc().GetShape(), GeShape({1,1,224,224}));
//  EXPECT_EQ(tensor->GetTensorDesc().GetOriginShape(), GeShape({1,1,224,224}));
//  EXPECT_EQ(tensor->GetTensorDesc().GetDataType(), DT_FLOAT);
//  EXPECT_EQ(tensor->GetTensorDesc().GetOriginDataType(), DT_FLOAT);
//  for (size_t i = 0; i < 224*224; ++i) {
//    EXPECT_FLOAT_EQ(reinterpret_cast<const float *>(tensor->GetData().data())[i], tensor_data[i]);
//  }


  EXPECT_EQ(add1->GetOpDesc()->GetName(), "add1");
  EXPECT_EQ(add1->GetType(), "Add");
  EXPECT_EQ(add1->GetOpDesc()->GetInputsSize(), 2);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputsSize(), 1);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetShape(), GeShape({1,1,224,224,16}));
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x1")->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetShape(), GeShape({1,1,224,224,16}));
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetInputDescPtr("x2")->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetShape(), GeShape({1,1,224,224,16}));
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetDataType(), DT_FLOAT);
  EXPECT_EQ(add1->GetOpDesc()->GetOutputDesc("y").GetOriginDataType(), DT_FLOAT);

  EXPECT_EQ(netoutput1->GetOpDesc()->GetName(), "NetOutputNode");
  EXPECT_EQ(netoutput1->GetType(), "NetOutput");
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputsSize(), 1);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetOutputsSize(), 0);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetShape(), GeShape({1,1,224,224,16}));
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetOriginShape(), GeShape({1,1,224,224}));
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetDataType(), DT_FLOAT);
  EXPECT_EQ(netoutput1->GetOpDesc()->GetInputDescPtr("x")->GetOriginDataType(), DT_FLOAT);

  EXPECT_EQ(data1->GetOutNodes().size(), 1);
  EXPECT_TRUE(ExpectConnected(data1, 0, add1, 0));
  EXPECT_EQ(const1->GetOutNodes().size(), 1);
  EXPECT_TRUE(ExpectConnected(const1, 0, add1, 1));
  EXPECT_EQ(add1->GetOutNodes().size(), 1);
  EXPECT_TRUE(ExpectConnected(add1, 0, netoutput1, 0));
}
}
class AttrUtilsUt : public testing::Test {};

TEST_F(AttrUtilsUt, HasAttrOk) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_FALSE(AttrUtils::HasAttr(op_desc, "abc"));
  EXPECT_FALSE(AttrUtils::HasAttr(op_desc, "bcd"));

  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "abc", 10));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, "bcd", "hello"));

  EXPECT_TRUE(AttrUtils::HasAttr(op_desc, "abc"));
  EXPECT_TRUE(AttrUtils::HasAttr(op_desc, "bcd"));
}

TEST_F(AttrUtilsUt, SetGetIntOk) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "abc", 10));
  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "bcd", 0xffffffffffff));

  int64_t i64;
  int32_t i32;
  uint32_t ui32;

  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "abc", i64));
  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "abc", i32));
  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "abc", ui32));
  EXPECT_EQ(i64, 10);
  EXPECT_EQ(i32, 10);
  EXPECT_EQ(ui32, 10);

  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "bcd", i64));
  EXPECT_EQ(i64, 0xffffffffffff);
  EXPECT_FALSE(AttrUtils::GetInt(op_desc, "bcd", i32));
  EXPECT_FALSE(AttrUtils::GetInt(op_desc, "bcd", ui32));
}

TEST_F(AttrUtilsUt, SetGetInt_ExceedsLimit) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetInt(op_desc, "bcd", 0xffffffff));

  int64_t i64;
  int32_t i32;
  uint32_t ui32;

  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "bcd", i64));
  EXPECT_FALSE(AttrUtils::GetInt(op_desc, "bcd", i32));
  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "bcd", ui32));
  EXPECT_EQ(i64, 0xffffffff);
  EXPECT_EQ(ui32, 0xffffffff);
}

TEST_F(AttrUtilsUt, SetGetListIntOk1) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "abc2", std::vector<int32_t>({1,2,3})));

  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  std::vector<uint32_t> ui32;

  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i64));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i32));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", ui32));
  EXPECT_EQ(i64, std::vector<int64_t>({1,2,3}));
  EXPECT_EQ(i32, std::vector<int32_t>({1,2,3}));
  EXPECT_EQ(ui32, std::vector<uint32_t>({1,2,3}));
}

TEST_F(AttrUtilsUt, SetGetListIntOk2) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "abc2", std::vector<uint32_t>({1,2,3})));

  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  std::vector<uint32_t> ui32;

  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i64));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i32));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", ui32));
  EXPECT_EQ(i64, std::vector<int64_t>({1,2,3}));
  EXPECT_EQ(i32, std::vector<int32_t>({1,2,3}));
  EXPECT_EQ(ui32, std::vector<uint32_t>({1,2,3}));
}

TEST_F(AttrUtilsUt, SetGetListIntOk3) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "abc2", std::vector<int64_t>({1,2,3})));

  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  std::vector<uint32_t> ui32;

  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i64));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i32));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", ui32));
  EXPECT_EQ(i64, std::vector<int64_t>({1,2,3}));
  EXPECT_EQ(i32, std::vector<int32_t>({1,2,3}));
  EXPECT_EQ(ui32, std::vector<uint32_t>({1,2,3}));
}

TEST_F(AttrUtilsUt, SetGetListIntOk4) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "abc2", {1,2,3}));

  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  std::vector<uint32_t> ui32;

  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i64));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i32));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", ui32));
  EXPECT_EQ(i64, std::vector<int64_t>({1,2,3}));
  EXPECT_EQ(i32, std::vector<int32_t>({1,2,3}));
  EXPECT_EQ(ui32, std::vector<uint32_t>({1,2,3}));
}

TEST_F(AttrUtilsUt, SetGetListInt_ExceedsLimit1) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "abc2", {1,2,3, 0xffffffffffff}));

  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  std::vector<uint32_t> ui32;

  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i64));
  EXPECT_FALSE(AttrUtils::GetListInt(op_desc, "abc2", i32));
  EXPECT_FALSE(AttrUtils::GetListInt(op_desc, "abc2", ui32));
  EXPECT_EQ(i64, std::vector<int64_t>({1,2,3,0xffffffffffff}));
}

TEST_F(AttrUtilsUt, SetGetListInt_ExceedsLimit2) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "abc2", {1,2,3, 0xffffffff}));

  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  std::vector<uint32_t> ui32;

  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i64));
  EXPECT_FALSE(AttrUtils::GetListInt(op_desc, "abc2", i32));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", ui32));
  EXPECT_EQ(i64, std::vector<int64_t>({1,2,3,0xffffffff}));
  EXPECT_EQ(ui32, std::vector<uint32_t>({1,2,3,0xffffffff}));
}

TEST_F(AttrUtilsUt, SetGetListInt_ExceedsLimit3) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "abc2", {1,2,3, -1}));

  std::vector<int64_t> i64;
  std::vector<int32_t> i32;
  //std::vector<uint32_t> ui32;

  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i64));
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "abc2", i32));
  //EXPECT_FALSE(AttrUtils::GetListInt(op_desc, "abc2", ui32));
  EXPECT_EQ(i64, std::vector<int64_t>({1,2,3,-1}));
  //EXPECT_EQ(i32, std::vector<int32_t>({1,2,3,-1}));
}

TEST_F(AttrUtilsUt, SetGetFloatOk) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetFloat(op_desc, "abc", 3.1415926));
  float f;
  EXPECT_TRUE(AttrUtils::GetFloat(op_desc, "abc", f));
  EXPECT_FLOAT_EQ(f, 3.1415926);
}

TEST_F(AttrUtilsUt, SetGetListFloatOk) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListFloat(op_desc, "abc", std::vector<float>({3.1415,4.1415,5.1415926})));
  std::vector<float> f;
  EXPECT_TRUE(AttrUtils::GetListFloat(op_desc, "abc", f));
  EXPECT_EQ(f.size(), 3);
  EXPECT_FLOAT_EQ(f[0], 3.1415);
  EXPECT_FLOAT_EQ(f[1], 4.1415);
  EXPECT_FLOAT_EQ(f[2], 5.1415926);
}

TEST_F(AttrUtilsUt, SetGetBoolOk) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetBool(op_desc, "abc", true));
  EXPECT_TRUE(AttrUtils::SetBool(op_desc, "bcd", false));
  bool b1 = false, b2 = true;
  EXPECT_TRUE(AttrUtils::GetBool(op_desc, "abc", b1));
  EXPECT_TRUE(AttrUtils::GetBool(op_desc, "bcd", b2));
  EXPECT_TRUE(b1);
  EXPECT_FALSE(b2);
}

TEST_F(AttrUtilsUt, SetGetListBoolOk) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListBool(op_desc, "abc", std::vector<bool>({true,false,false,true})));
  std::vector<bool> b;
  EXPECT_TRUE(AttrUtils::GetListBool(op_desc, "abc", b));
  EXPECT_EQ(b.size(), 4);
  EXPECT_TRUE(b[0]);
  EXPECT_FALSE(b[1]);
  EXPECT_FALSE(b[2]);
  EXPECT_TRUE(b[3]);
}

TEST_F(AttrUtilsUt, SetGetStrOk) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetStr(op_desc, "abc", "Hello"));
  EXPECT_TRUE(AttrUtils::SetStr(op_desc, "bcd", "World"));
  std::string s1, s2;
  EXPECT_TRUE(AttrUtils::GetStr(op_desc, "abc", s1));
  EXPECT_TRUE(AttrUtils::GetStr(op_desc, "bcd", s2));
  EXPECT_EQ(s1, "Hello");
  EXPECT_EQ(s2, "World");
}

TEST_F(AttrUtilsUt, SetGetListStrOk) {
  auto op_desc = std::make_shared<OpDesc>();

  EXPECT_TRUE(AttrUtils::SetListStr(op_desc, "abc", std::vector<std::string>({"Hello", "world", "!"})));
  std::vector<std::string> s;
  EXPECT_TRUE(AttrUtils::GetListStr(op_desc, "abc", s));
  EXPECT_EQ(s, std::vector<std::string>({"Hello", "world", "!"}));
}

TEST_F(AttrUtilsUt, SetGetTensorDescOk) {
  auto op_desc = std::make_shared<OpDesc>();
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({8,1,128,128,16})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({8,3,128,128})));
  td.SetFormat(FORMAT_NC1HWC0);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT16);
  td.SetOriginDataType(DT_FLOAT);
  AttrUtils::SetStr(&td, "bcd", "Hello world");

  EXPECT_TRUE(AttrUtils::SetTensorDesc(op_desc, "abc", td));

  GeTensorDesc td1;
  EXPECT_TRUE(AttrUtils::GetTensorDesc(op_desc, "abc", td1));
  EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>({8,1,128,128,16}));
  EXPECT_EQ(td1.GetOriginShape().GetDims(), std::vector<int64_t>({8,3,128,128}));
  EXPECT_EQ(td1.GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(td1.GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(td1.GetDataType(), DT_FLOAT16);
  EXPECT_EQ(td1.GetOriginDataType(), DT_FLOAT);
  std::string s;
  EXPECT_TRUE(AttrUtils::GetStr(&td1, "bcd", s));
  EXPECT_EQ(s, "Hello world");
}

TEST_F(AttrUtilsUt, SetGetTensorDescOk_CopyValidation1) {
  auto op_desc = std::make_shared<OpDesc>();
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({8,1,128,128,16})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({8,3,128,128})));
  td.SetFormat(FORMAT_NC1HWC0);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT16);
  td.SetOriginDataType(DT_FLOAT);
  AttrUtils::SetStr(&td, "bcd", "Hello world");

  EXPECT_TRUE(AttrUtils::SetTensorDesc(op_desc, "abc", td));
  td.SetShape(GeShape(std::vector<int64_t>({1})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({8})));
  td.SetFormat(FORMAT_ND);
  td.SetOriginFormat(FORMAT_ND);
  td.SetDataType(DT_INT16);
  td.SetOriginDataType(DT_INT16);
  AttrUtils::SetStr(&td, "bcd", "adasdfasdf");

  GeTensorDesc td1;
  EXPECT_TRUE(AttrUtils::GetTensorDesc(op_desc, "abc", td1));
  EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>({8,1,128,128,16}));
  EXPECT_EQ(td1.GetOriginShape().GetDims(), std::vector<int64_t>({8,3,128,128}));
  EXPECT_EQ(td1.GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(td1.GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(td1.GetDataType(), DT_FLOAT16);
  EXPECT_EQ(td1.GetOriginDataType(), DT_FLOAT);
  std::string s;
  EXPECT_TRUE(AttrUtils::GetStr(&td1, "bcd", s));
  EXPECT_EQ(s, "Hello world");
}

TEST_F(AttrUtilsUt, SetGetTensorDescOk_CopyValidation2) {
  auto op_desc = std::make_shared<OpDesc>();
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({8,1,128,128,16})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({8,3,128,128})));
  td.SetFormat(FORMAT_NC1HWC0);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT16);
  td.SetOriginDataType(DT_FLOAT);
  AttrUtils::SetStr(&td, "bcd", "Hello world");

  EXPECT_TRUE(AttrUtils::SetTensorDesc(op_desc, "abc", td));

  GeTensorDesc td1;
  EXPECT_TRUE(AttrUtils::GetTensorDesc(op_desc, "abc", td1));
  td1.SetShape(GeShape(std::vector<int64_t>({1})));
  td1.SetOriginShape(GeShape(std::vector<int64_t>({8})));
  td1.SetFormat(FORMAT_ND);
  td1.SetOriginFormat(FORMAT_ND);
  td1.SetDataType(DT_INT16);
  td1.SetOriginDataType(DT_INT16);
  AttrUtils::SetStr(&td1, "bcd", "adasdfasdf");

  EXPECT_EQ(td.GetShape().GetDims(), std::vector<int64_t>({8,1,128,128,16}));
  EXPECT_EQ(td.GetOriginShape().GetDims(), std::vector<int64_t>({8,3,128,128}));
  EXPECT_EQ(td.GetFormat(), FORMAT_NC1HWC0);
  EXPECT_EQ(td.GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(td.GetDataType(), DT_FLOAT16);
  EXPECT_EQ(td.GetOriginDataType(), DT_FLOAT);
  std::string s;
  EXPECT_TRUE(AttrUtils::GetStr(&td, "bcd", s));
  EXPECT_EQ(s, "Hello world");
}

TEST_F(AttrUtilsUt, SetGetListTensorDescOk) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<GeTensorDesc> tds(5);
  for (auto &td : tds) {
    td.SetShape(GeShape(std::vector<int64_t>({8,1,128,128,16})));
    td.SetOriginShape(GeShape(std::vector<int64_t>({8,3,128,128})));
    td.SetFormat(FORMAT_NC1HWC0);
    td.SetOriginFormat(FORMAT_NCHW);
    td.SetDataType(DT_FLOAT16);
    td.SetOriginDataType(DT_FLOAT);
    AttrUtils::SetStr(&td, "bcd", "Hello world");
  }

  EXPECT_TRUE(AttrUtils::SetListTensorDesc(op_desc, "abc", tds));

  std::vector<GeTensorDesc> tds1;
  EXPECT_TRUE(AttrUtils::GetListTensorDesc(op_desc, "abc", tds1));
  for (auto &td1 : tds1) {
    EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>({8,1,128,128,16}));
    EXPECT_EQ(td1.GetOriginShape().GetDims(), std::vector<int64_t>({8,3,128,128}));
    EXPECT_EQ(td1.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(td1.GetOriginFormat(), FORMAT_NCHW);
    EXPECT_EQ(td1.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(td1.GetOriginDataType(), DT_FLOAT);
    std::string s;
    EXPECT_TRUE(AttrUtils::GetStr(&td1, "bcd", s));
    EXPECT_EQ(s, "Hello world");
  }
}

TEST_F(AttrUtilsUt, SetGetListTensorDescOk_CopyValidation1) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<GeTensorDesc> tds(5);
  for (auto &td : tds) {
    td.SetShape(GeShape(std::vector<int64_t>({8,1,128,128,16})));
    td.SetOriginShape(GeShape(std::vector<int64_t>({8,3,128,128})));
    td.SetFormat(FORMAT_NC1HWC0);
    td.SetOriginFormat(FORMAT_NCHW);
    td.SetDataType(DT_FLOAT16);
    td.SetOriginDataType(DT_FLOAT);
    AttrUtils::SetStr(&td, "bcd", "Hello world");
  }

  EXPECT_TRUE(AttrUtils::SetListTensorDesc(op_desc, "abc", tds));
  for (auto &td : tds) {
    td.SetShape(GeShape(std::vector<int64_t>({1})));
    td.SetOriginShape(GeShape(std::vector<int64_t>({8})));
    td.SetFormat(FORMAT_ND);
    td.SetOriginFormat(FORMAT_ND);
    td.SetDataType(DT_INT16);
    td.SetOriginDataType(DT_INT16);
    AttrUtils::SetStr(&td, "bcd", "adasdfasdf");
  }

  std::vector<GeTensorDesc> tds1;
  EXPECT_TRUE(AttrUtils::GetListTensorDesc(op_desc, "abc", tds1));
  for (auto &td1 : tds1) {
    EXPECT_EQ(td1.GetShape().GetDims(), std::vector<int64_t>({8,1,128,128,16}));
    EXPECT_EQ(td1.GetOriginShape().GetDims(), std::vector<int64_t>({8,3,128,128}));
    EXPECT_EQ(td1.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(td1.GetOriginFormat(), FORMAT_NCHW);
    EXPECT_EQ(td1.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(td1.GetOriginDataType(), DT_FLOAT);
    std::string s;
    EXPECT_TRUE(AttrUtils::GetStr(&td1, "bcd", s));
    EXPECT_EQ(s, "Hello world");
  }
}

TEST_F(AttrUtilsUt, SetGetListTensorDescOk_CopyValidation2) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<GeTensorDesc> tds(5);
  for (auto &td : tds) {
    td.SetShape(GeShape(std::vector<int64_t>({8,1,128,128,16})));
    td.SetOriginShape(GeShape(std::vector<int64_t>({8,3,128,128})));
    td.SetFormat(FORMAT_NC1HWC0);
    td.SetOriginFormat(FORMAT_NCHW);
    td.SetDataType(DT_FLOAT16);
    td.SetOriginDataType(DT_FLOAT);
    AttrUtils::SetStr(&td, "bcd", "Hello world");
  }

  EXPECT_TRUE(AttrUtils::SetListTensorDesc(op_desc, "abc", tds));
  std::vector<GeTensorDesc> tds1;
  EXPECT_TRUE(AttrUtils::GetListTensorDesc(op_desc, "abc", tds1));
  for (auto &td1 : tds1) {
    td1.SetShape(GeShape(std::vector<int64_t>({1})));
    td1.SetOriginShape(GeShape(std::vector<int64_t>({8})));
    td1.SetFormat(FORMAT_ND);
    td1.SetOriginFormat(FORMAT_ND);
    td1.SetDataType(DT_INT16);
    td1.SetOriginDataType(DT_INT16);
    AttrUtils::SetStr(&td1, "bcd", "adasdfasdf");
  }
  for (auto &td : tds) {
    EXPECT_EQ(td.GetShape().GetDims(), std::vector<int64_t>({8,1,128,128,16}));
    EXPECT_EQ(td.GetOriginShape().GetDims(), std::vector<int64_t>({8,3,128,128}));
    EXPECT_EQ(td.GetFormat(), FORMAT_NC1HWC0);
    EXPECT_EQ(td.GetOriginFormat(), FORMAT_NCHW);
    EXPECT_EQ(td.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(td.GetOriginDataType(), DT_FLOAT);
    std::string s;
    EXPECT_TRUE(AttrUtils::GetStr(&td, "bcd", s));
    EXPECT_EQ(s, "Hello world");
  }
}

TEST_F(AttrUtilsUt, SetGetTensorOk1) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_data = GetRandomFloat({8, 3, 224, 224});
  {
    auto tensor = CreateTensor_8_3_224_224(tensor_data.get());
    ConstGeTensorPtr tensor1 = tensor;

    EXPECT_TRUE(AttrUtils::SetTensor(op_desc, "abc", tensor));
    EXPECT_TRUE(AttrUtils::SetTensor(op_desc, "bcd", *tensor));
    EXPECT_TRUE(AttrUtils::SetTensor(op_desc, "cde", tensor1));
  }

  ConstGeTensorPtr out_tensor;
  EXPECT_TRUE(AttrUtils::GetTensor(op_desc, "abc", out_tensor));
  EXPECT_NE(out_tensor, nullptr);
  ExpectTensorEqual_8_3_224_224(out_tensor, tensor_data.get());

  EXPECT_TRUE(AttrUtils::GetTensor(op_desc, "bcd", out_tensor));
  EXPECT_NE(out_tensor, nullptr);
  ExpectTensorEqual_8_3_224_224(out_tensor, tensor_data.get());

  EXPECT_TRUE(AttrUtils::GetTensor(op_desc, "cde", out_tensor));
  EXPECT_NE(out_tensor, nullptr);
  ExpectTensorEqual_8_3_224_224(out_tensor, tensor_data.get());
}

TEST_F(AttrUtilsUt, SetGetTensorOk1_CopyValidation1) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_data = GetRandomFloat({8, 3, 224, 224});
  auto tensor_data1 = GetRandomFloat({16, 3, 224, 224});
  auto tensor = CreateTensor_8_3_224_224(tensor_data.get());

  EXPECT_TRUE(AttrUtils::SetTensor(op_desc, "abc", tensor));
  tensor->MutableData().SetData(reinterpret_cast<uint8_t *>(tensor_data1.get()), 16*3*224*224*sizeof(float));
  tensor->MutableTensorDesc().SetShape(GeShape(std::vector<int64_t>({16,3,224,224})));
  tensor->MutableTensorDesc().SetOriginShape(GeShape(std::vector<int64_t>({16,3,224,224})));

  ConstGeTensorPtr out_tensor;
  EXPECT_TRUE(AttrUtils::GetTensor(op_desc, "abc", out_tensor));
  EXPECT_NE(out_tensor, nullptr);
  ExpectTensorEqual_8_3_224_224(out_tensor, tensor_data.get());
}

TEST_F(AttrUtilsUt, SetGetTensorOk1_MultipleGet) {
  auto op_desc = std::make_shared<OpDesc>();
  auto tensor_data = GetRandomFloat({8, 3, 224, 224});
  auto tensor = CreateTensor_8_3_224_224(tensor_data.get());

  EXPECT_TRUE(AttrUtils::SetTensor(op_desc, "abc", tensor));

  auto tensor_data1 = GetRandomFloat({16, 3, 224, 224});
  GeTensorPtr out_tensor = nullptr;
  EXPECT_TRUE(AttrUtils::MutableTensor(op_desc, "abc", out_tensor));
  EXPECT_NE(out_tensor, nullptr);
  out_tensor->MutableData().SetData(reinterpret_cast<uint8_t *>(tensor_data1.get()), 16*3*224*224*sizeof(float));
  out_tensor->MutableTensorDesc().SetShape(GeShape(std::vector<int64_t>({16,3,224,224})));
  out_tensor->MutableTensorDesc().SetOriginShape(GeShape(std::vector<int64_t>({16,3,224,224})));

  out_tensor = nullptr;
  EXPECT_TRUE(AttrUtils::MutableTensor(op_desc, "abc", out_tensor));
  EXPECT_NE(out_tensor, nullptr);

  EXPECT_NE(const_cast<uint8_t *>(out_tensor->GetData().data()), reinterpret_cast<uint8_t*>(tensor_data1.get()));
  EXPECT_EQ(out_tensor->GetData().size(), 16*3*224*224*sizeof(float));
  for (size_t i = 0; i < 16*3*224*224; ++i) {
    EXPECT_FLOAT_EQ(reinterpret_cast<const float *>(out_tensor->GetData().data())[i], tensor_data1.get()[i]);
  }
  EXPECT_EQ(out_tensor->GetTensorDesc().GetShape().GetDims(), std::vector<int64_t>({16,3,224,224}));
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginShape().GetDims(), std::vector<int64_t>({16,3,224,224}));
  EXPECT_EQ(out_tensor->GetTensorDesc().GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetDataType(), DT_FLOAT);
  EXPECT_EQ(out_tensor->GetTensorDesc().GetOriginDataType(), DT_FLOAT);
  std::string s;
  EXPECT_TRUE(AttrUtils::GetStr(&out_tensor->GetTensorDesc(), "bcd", s));
  EXPECT_EQ(s, "Hello world");
}

TEST_F(AttrUtilsUt, SetGetListTensor) {
  auto data1 = GetRandomFloat({8,3,224,224});
  auto data2 = GetRandomFloat({8,1,224,224,16});
  auto data3 = GetRandomFloat({8,3,224,224});
  auto tensor1 = CreateTensor_8_3_224_224(data1.get());
  auto tensor2 = CreateTensor_5d_8_3_224_224(data2.get());
  auto tensor3 = CreateTensor_8_3_224_224(data3.get());

  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_TRUE(AttrUtils::SetListTensor(op_desc, "abc", std::vector<GeTensorPtr>({tensor1, tensor2, tensor3})));
  EXPECT_TRUE(AttrUtils::SetListTensor(op_desc, "abc1", std::vector<ConstGeTensorPtr>({tensor1, tensor2, tensor3})));
  EXPECT_TRUE(AttrUtils::SetListTensor(op_desc, "abc2", std::vector<GeTensor>({*tensor1, *tensor2, *tensor3})));
  EXPECT_TRUE(AttrUtils::SetListTensor(op_desc, "abc3", {tensor1, tensor2, tensor3}));

  std::vector<ConstGeTensorPtr> out_tensors;
  EXPECT_TRUE(AttrUtils::GetListTensor(op_desc, "abc", out_tensors));
  EXPECT_EQ(out_tensors.size(), 3);
  ExpectTensorEqual_8_3_224_224(out_tensors[0], data1.get());
  ExpectTensorEqual_5d_8_3_224_224(out_tensors[1], data2.get());
  ExpectTensorEqual_8_3_224_224(out_tensors[2], data3.get());

  EXPECT_TRUE(AttrUtils::GetListTensor(op_desc, "abc1", out_tensors));
  EXPECT_EQ(out_tensors.size(), 3);
  ExpectTensorEqual_8_3_224_224(out_tensors[0], data1.get());
  ExpectTensorEqual_5d_8_3_224_224(out_tensors[1], data2.get());
  ExpectTensorEqual_8_3_224_224(out_tensors[2], data3.get());

  EXPECT_TRUE(AttrUtils::GetListTensor(op_desc, "abc2", out_tensors));
  EXPECT_EQ(out_tensors.size(), 3);
  ExpectTensorEqual_8_3_224_224(out_tensors[0], data1.get());
  ExpectTensorEqual_5d_8_3_224_224(out_tensors[1], data2.get());
  ExpectTensorEqual_8_3_224_224(out_tensors[2], data3.get());

  EXPECT_TRUE(AttrUtils::GetListTensor(op_desc, "abc3", out_tensors));
  EXPECT_EQ(out_tensors.size(), 3);
  ExpectTensorEqual_8_3_224_224(out_tensors[0], data1.get());
  ExpectTensorEqual_5d_8_3_224_224(out_tensors[1], data2.get());
  ExpectTensorEqual_8_3_224_224(out_tensors[2], data3.get());
}

TEST_F(AttrUtilsUt, SetGetListTensor_MutableOk) {
  auto data1 = GetRandomFloat({8,3,224,224});
  auto data2 = GetRandomFloat({8,1,224,224,16});
  auto data3 = GetRandomFloat({8,3,224,224});
  auto data4 = GetRandomFloat({8,1,224,224,16});
  auto tensor1 = CreateTensor_8_3_224_224(data1.get());
  auto tensor2 = CreateTensor_5d_8_3_224_224(data2.get());
  auto tensor3 = CreateTensor_8_3_224_224(data3.get());

  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_TRUE(AttrUtils::SetListTensor(op_desc, "abc", {tensor1, tensor2, tensor3}));

  std::vector<GeTensorPtr> out_tensors;
  EXPECT_TRUE(AttrUtils::MutableListTensor(op_desc, "abc", out_tensors));
  out_tensors[2]->MutableData().SetData(reinterpret_cast<uint8_t *>(data4.get()), 8*1*224*224*16*sizeof(float));
  out_tensors[2]->MutableTensorDesc().SetShape(GeShape(std::vector<int64_t>({8,1,224,224,16})));
  out_tensors[2]->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);
  out_tensors.clear();

  EXPECT_TRUE(AttrUtils::MutableListTensor(op_desc, "abc", out_tensors));
  EXPECT_EQ(out_tensors.size(), 3);
  ExpectTensorEqual_8_3_224_224(out_tensors[0], data1.get());
  ExpectTensorEqual_5d_8_3_224_224(out_tensors[1], data2.get());
  ExpectTensorEqual_5d_8_3_224_224(out_tensors[2], data4.get());
}

TEST_F(AttrUtilsUt, SetGetListTensor_CopyValidation) {
  auto data1 = GetRandomFloat({8,3,224,224});
  auto data2 = GetRandomFloat({8,1,224,224,16});
  auto data3 = GetRandomFloat({8,3,224,224});
  auto data4 = GetRandomFloat({8,1,224,224,16});
  auto tensor1 = CreateTensor_8_3_224_224(data1.get());
  auto tensor2 = CreateTensor_5d_8_3_224_224(data2.get());
  auto tensor3 = CreateTensor_8_3_224_224(data3.get());

  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_TRUE(AttrUtils::SetListTensor(op_desc, "abc", {tensor1, tensor2, tensor3}));
  tensor3->MutableData().SetData(reinterpret_cast<uint8_t *>(data4.get()), 8*1*224*224*16*sizeof(float));
  tensor3->MutableTensorDesc().SetShape(GeShape(std::vector<int64_t>({8,1,224,224,16})));
  tensor3->MutableTensorDesc().SetFormat(FORMAT_NC1HWC0);

  std::vector<ConstGeTensorPtr> out_tensors;
  EXPECT_TRUE(AttrUtils::GetListTensor(op_desc, "abc", out_tensors));
  EXPECT_EQ(out_tensors.size(), 3);
  ExpectTensorEqual_8_3_224_224(out_tensors[0], data1.get());
  ExpectTensorEqual_5d_8_3_224_224(out_tensors[1], data2.get());
  ExpectTensorEqual_8_3_224_224(out_tensors[2], data3.get());
}

TEST_F(AttrUtilsUt, SetGetGraphGraph) {
  auto const_data = GetRandomFloat({1,1,224,224});
  auto holder = std::make_shared<ComputeGraph>("holder");

  {
    auto graph = CreateGraph_1_1_224_224(const_data.get());
    EXPECT_TRUE(AttrUtils::SetGraph(holder, "abc", graph));
  }

  ComputeGraphPtr out_graph = nullptr;
  EXPECT_TRUE(AttrUtils::GetGraph(holder, "abc", out_graph));

  EXPECT_NE(out_graph, nullptr);
  ExpectEqGraph_1_1_224_224(out_graph, const_data.get());
}

TEST_F(AttrUtilsUt, SetGraphGraph_CopyValidation) {
  auto const_data = GetRandomFloat({1,1,224,224});
  auto holder = std::make_shared<ComputeGraph>("holder");

  auto graph = CreateGraph_1_1_224_224(const_data.get());
  EXPECT_TRUE(AttrUtils::SetGraph(holder, "abc", graph));
  graph->FindNode("data1")->GetOpDesc()->MutableOutputDesc(0)->SetFormat(FORMAT_NC1HWC0);
  graph->FindNode("data1")->GetOpDesc()->MutableOutputDesc(0)->SetShape(GeShape({1,1,224,224,16}));

  ComputeGraphPtr out_graph = nullptr;
  EXPECT_TRUE(AttrUtils::GetGraph(holder, "abc", out_graph));

  EXPECT_NE(out_graph, nullptr);
  ExpectEqGraph_1_1_224_224(out_graph, const_data.get());
}

TEST_F(AttrUtilsUt, SetGetListGraphGraph) {
  auto const_data1 = GetRandomFloat({1,1,224,224,16});
  auto const_data2 = GetRandomFloat({1,1,224,224,16});
  auto const_data3 = GetRandomFloat({1,1,224,224});
  auto holder = std::make_shared<ComputeGraph>("holder");

  {
    auto graph1 = CreateGraph_5d_1_1_224_224(const_data1.get());
    auto graph2 = CreateGraph_5d_1_1_224_224(const_data2.get());
    auto graph3 = CreateGraph_1_1_224_224(const_data3.get());
    EXPECT_TRUE(AttrUtils::SetListGraph(holder, "abc", std::vector<ComputeGraphPtr>({graph1, graph2, graph3})));
  }

  std::vector<ComputeGraphPtr> out_graphs;
  EXPECT_TRUE(AttrUtils::GetListGraph(holder, "abc", out_graphs));

  EXPECT_EQ(out_graphs.size(), 3);
  ExpectEqGraph_5d_1_1_224_224(out_graphs[0], const_data1.get());
  ExpectEqGraph_5d_1_1_224_224(out_graphs[1], const_data2.get());
  ExpectEqGraph_1_1_224_224(out_graphs[2], const_data3.get());
}

TEST_F(AttrUtilsUt, SimpleTest) {
  auto op_desc = std::make_shared<OpDesc>();
  {
    op_desc->SetAttr("Foo", GeAttrValue::CreateFrom<bool>(true));
  }
  EXPECT_TRUE(AttrUtils::SetBool(op_desc, "Foo", true));
  bool val = false;
  EXPECT_TRUE(AttrUtils::GetBool(op_desc, "Foo", val));
  EXPECT_TRUE(val);
}

TEST_F(AttrUtilsUt, CopyOpdesc) {
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1, 1, 224, 224})));
  td.SetFormat(FORMAT_NCHW);
  td.SetOriginFormat(FORMAT_NCHW);
  td.SetDataType(DT_FLOAT);
  td.SetOriginDataType(DT_FLOAT);
  vector<int64_t> input_size = {12};
  AttrUtils::SetListInt(td, "input_size", input_size);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("x1", td);
  op_desc->AddInputDesc("x2", td);
  op_desc->AddOutputDesc("y", td);
  AttrUtils::SetStr(op_desc, "padding", "SAME");

  auto new_desc = AttrUtils::CopyOpDesc(op_desc);

  std::string padding;
  EXPECT_TRUE(AttrUtils::GetStr(new_desc, "padding", padding));
  EXPECT_EQ(padding, "SAME");

  EXPECT_EQ(new_desc->GetInputsSize(), 2);
  EXPECT_EQ(new_desc->GetOutputsSize(), 1);

  EXPECT_EQ(new_desc->GetInputDescPtr("x1"), new_desc->GetInputDescPtr(0));
  EXPECT_EQ(new_desc->GetInputDescPtr("x2"), new_desc->GetInputDescPtr(1));
  EXPECT_EQ(new_desc->MutableOutputDesc("y"), new_desc->MutableOutputDesc(0));

  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetInputDescPtr(0)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  vector<int64_t> new_input_size;
  EXPECT_TRUE(AttrUtils::GetListInt(new_desc->GetInputDescPtr(0), "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));

  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetInputDescPtr(1)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  new_input_size.clear();
  auto new_input_desc = new_desc->GetInputDescPtr(1);
  EXPECT_TRUE(AttrUtils::GetListInt(new_input_desc, "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));

  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginDataType(), DT_FLOAT);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginFormat(), FORMAT_NCHW);
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  EXPECT_EQ(new_desc->GetOutputDescPtr(0)->GetOriginShape().GetDims(), std::vector<int64_t>({1, 1, 224, 224}));
  new_input_size.clear();
  EXPECT_TRUE(AttrUtils::GetListInt(new_desc->GetInputDescPtr(0), "input_size", new_input_size));
  EXPECT_EQ(new_input_size, std::vector<int64_t>({12}));
}


TEST_F(AttrUtilsUt, CopyOpdesc2) {
  GeTensorDesc td = StandardTd_5d_1_1_224_224();

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("x1", td);
  op_desc->AddInputDesc("x2", td);
  op_desc->AddOutputDesc("y", td);
  AttrUtils::SetStr(op_desc, "padding", "VALID");

  auto new_desc1 = AttrUtils::CopyOpDesc(op_desc);

  std::string padding;
  EXPECT_TRUE(AttrUtils::GetStr(new_desc1, "padding", padding));
  EXPECT_EQ(padding, "VALID");

  AttrUtils::SetStr(new_desc1, "padding", "SAME");
  padding.clear();
  EXPECT_TRUE(AttrUtils::GetStr(new_desc1, "padding", padding));
  EXPECT_EQ(padding, "SAME");

  auto new_desc2 = AttrUtils::CopyOpDesc(new_desc1);
  padding.clear();
  EXPECT_TRUE(AttrUtils::GetStr(new_desc2, "padding", padding));
  EXPECT_EQ(padding, "SAME");
}

TEST_F(AttrUtilsUt, CloneOpdesc) {
  GeTensorDesc td = StandardTd_5d_1_1_224_224();

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc("x1", td);
  op_desc->AddInputDesc("x2", td);
  op_desc->AddOutputDesc("y", td);
  AttrUtils::SetStr(op_desc, "padding", "VALID");

  auto new_desc1 = AttrUtils::CloneOpDesc(op_desc);

  std::string padding;
  EXPECT_TRUE(AttrUtils::GetStr(new_desc1, "padding", padding));
  EXPECT_EQ(padding, "VALID");

  AttrUtils::SetStr(new_desc1, "padding", "SAME");
  padding.clear();
  EXPECT_TRUE(AttrUtils::GetStr(new_desc1, "padding", padding));
  EXPECT_EQ(padding, "SAME");

  auto new_desc2 = AttrUtils::CloneOpDesc(new_desc1);
  padding.clear();
  EXPECT_TRUE(AttrUtils::GetStr(new_desc2, "padding", padding));
  EXPECT_EQ(padding, "SAME");
}

TEST_F(AttrUtilsUt, SetGetBytes) {
  GeTensorDesc td;
  auto data = GetRandomFloat({1,2,3,4});
  auto b1 = Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data.get()), sizeof(float) * 1 * 2 * 3 * 4);
  EXPECT_TRUE(AttrUtils::SetBytes(&td, "abc", b1));
  Buffer b2;
  EXPECT_TRUE(AttrUtils::GetBytes(&td, "abc", b2));
  EXPECT_EQ(b1.size(), b2.size());
  EXPECT_EQ(memcmp(b1.data(), b2.data(), b1.size()), 0);
  EXPECT_NE(b1.data(), b2.data());
}

TEST_F(AttrUtilsUt, SetGetBytes_ZeroCopy) {
  GeTensorDesc td;
  auto data = GetRandomFloat({1,2,3,4});
  auto b1 = Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data.get()), sizeof(float) * 1 * 2 * 3 * 4);
  auto addr = b1.data();
  EXPECT_TRUE(AttrUtils::SetZeroCopyBytes(&td, "abc", std::move(b1)));
  Buffer b2;
  EXPECT_TRUE(AttrUtils::GetZeroCopyBytes(&td, "abc", b2));
  EXPECT_EQ(addr, b2.data());
  EXPECT_EQ(b2.size(), sizeof(float) * 2 * 3 * 4);
}

TEST_F(AttrUtilsUt, SetGetBytes_CopyValidation) {
  GeTensorDesc td;
  auto data = GetRandomFloat({1,2,3,4});
  auto b1 = Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data.get()), sizeof(float) * 1 * 2 * 3 * 4);
  EXPECT_TRUE(AttrUtils::SetBytes(&td, "abc", b1));
  b1.ClearBuffer();
  Buffer b2;
  EXPECT_TRUE(AttrUtils::GetBytes(&td, "abc", b2));
  EXPECT_EQ(sizeof(float) * 1 * 2 * 3 * 4, b2.size());
  EXPECT_EQ(memcmp(data.get(), b2.data(), b2.size()), 0);
}

TEST_F(AttrUtilsUt, SetGetListBytes) {
  GeTensorDesc td;
  auto data1 = GetRandomFloat({20});
  auto data2 = GetRandomFloat({40});
  auto data3 = GetRandomFloat({90});
  std::vector<Buffer> bufs = {
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data1.get()), sizeof(float) * 20),
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data2.get()), sizeof(float) * 40),
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data3.get()), sizeof(float) * 90)
  };
  EXPECT_TRUE(AttrUtils::SetListBytes(&td, "abc", bufs));
  std::vector<Buffer> out_bufs;
  EXPECT_TRUE(AttrUtils::GetListBytes(&td, "abc", out_bufs));
  EXPECT_EQ(out_bufs.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(out_bufs[i].size(), bufs[i].size());
    EXPECT_EQ(memcmp(out_bufs[i].data(), bufs[i].data(), out_bufs[i].size()), 0);
    EXPECT_NE(out_bufs[i].data(), bufs[i].data());
  }
}

TEST_F(AttrUtilsUt, SetGetListBytes_CopyValidation) {
  GeTensorDesc td;
  auto data1 = GetRandomFloat({20});
  auto data2 = GetRandomFloat({40});
  auto data3 = GetRandomFloat({90});
  std::vector<Buffer> bufs = {
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data1.get()), sizeof(float) * 20),
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data2.get()), sizeof(float) * 40),
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data3.get()), sizeof(float) * 90)
  };

  EXPECT_TRUE(AttrUtils::SetListBytes(&td, "abc", bufs));
  bufs[0].ClearBuffer();
  bufs[1].ClearBuffer();
  bufs[2].ClearBuffer();

  std::vector<Buffer> out_bufs;
  EXPECT_TRUE(AttrUtils::GetListBytes(&td, "abc", out_bufs));
  EXPECT_EQ(out_bufs.size(), 3);

  EXPECT_EQ(out_bufs[0].size(), 20 * sizeof(float));
  EXPECT_EQ(memcmp(out_bufs[0].data(), data1.get(), out_bufs[0].size()), 0);
  EXPECT_EQ(out_bufs[1].size(), 40 * sizeof(float));
  EXPECT_EQ(memcmp(out_bufs[1].data(), data2.get(), out_bufs[1].size()), 0);
  EXPECT_EQ(out_bufs[2].size(), 90 * sizeof(float));
  EXPECT_EQ(memcmp(out_bufs[2].data(), data3.get(), out_bufs[2].size()), 0);
}

TEST_F(AttrUtilsUt, SetGetListBytes_ZeroCopy) {
  GeTensorDesc td;
  auto data1 = GetRandomFloat({20});
  auto data2 = GetRandomFloat({40});
  auto data3 = GetRandomFloat({90});
  std::vector<Buffer> bufs = {
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data1.get()), sizeof(float) * 20),
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data2.get()), sizeof(float) * 40),
      Buffer::CopyFrom(reinterpret_cast<uint8_t *>(data3.get()), sizeof(float) * 90)
  };
  EXPECT_TRUE(AttrUtils::SetZeroCopyListBytes(&td, "abc", bufs));
  std::vector<Buffer> out_bufs;
  EXPECT_TRUE(AttrUtils::GetZeroCopyListBytes(&td, "abc", out_bufs));
  EXPECT_EQ(out_bufs.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(out_bufs[i].data(), bufs[i].data());
  }
}

TEST_F(AttrUtilsUt, SetGetListListInt) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_TRUE(AttrUtils::SetListListInt(op_desc, "abc", std::vector<std::vector<int64_t>>({{1,2,3},{4,4,5},{2,2}})));
  std::vector<std::vector<int64_t>> vec;
  EXPECT_TRUE(AttrUtils::GetListListInt(op_desc, "abc", vec));
  EXPECT_EQ(vec, std::vector<std::vector<int64_t>>({{1,2,3},{4,4,5},{2,2}}));
}

TEST_F(AttrUtilsUt, SetGetListListFloat) {
  auto op_desc = std::make_shared<OpDesc>();
  EXPECT_TRUE(AttrUtils::SetListListFloat(op_desc, "abc", std::vector<std::vector<float>>({{1.1,2.9,3.14},{4.122,43.4,5.55},{2.1,2.0}})));
  std::vector<std::vector<float>> vec;
  EXPECT_TRUE(AttrUtils::GetListListFloat(op_desc, "abc", vec));
  EXPECT_EQ(vec.size(), 3);
  EXPECT_EQ(vec[0].size(), 3);
  EXPECT_EQ(vec[1].size(), 3);
  EXPECT_EQ(vec[2].size(), 2);
  EXPECT_FLOAT_EQ(vec[0][0], 1.1);
  EXPECT_FLOAT_EQ(vec[1][0], 4.122);
  EXPECT_FLOAT_EQ(vec[2][0], 2.1);
}

TEST_F(AttrUtilsUt, SetGetNamedAttrs) {
  auto op_desc = std::make_shared<OpDesc>();
  NamedAttrs nas;
  nas.SetName("Hello Name");
  nas.SetAttr("abc", AnyValue::CreateFrom(static_cast<int64_t>(10)));
  nas.SetAttr("bcd", AnyValue::CreateFrom(true));

  EXPECT_TRUE(AttrUtils::SetNamedAttrs(op_desc, "attr", nas));

  NamedAttrs out_nas;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(op_desc, "attr", out_nas));
  EXPECT_EQ(out_nas.GetName(), nas.GetName());
  AnyValue av;
  EXPECT_EQ(out_nas.GetAttr("abc", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<int64_t>(), nullptr);
  EXPECT_EQ(*av.Get<int64_t>(), 10);

  EXPECT_EQ(out_nas.GetAttr("bcd", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<bool>(), nullptr);
  EXPECT_EQ(*av.Get<bool>(), true);
}

TEST_F(AttrUtilsUt, SetGetNamedAttrs_CopyValidation) {
  auto op_desc = std::make_shared<OpDesc>();
  NamedAttrs nas;
  nas.SetName("Hello Name");
  nas.SetAttr("abc", AnyValue::CreateFrom(static_cast<int64_t>(10)));
  nas.SetAttr("bcd", AnyValue::CreateFrom(true));

  EXPECT_TRUE(AttrUtils::SetNamedAttrs(op_desc, "attr", nas));
  AnyValue tmp_av;
  nas.GetAttr("abc", tmp_av);
  tmp_av.SetValue(static_cast<int64_t>(1024));
  nas.SetAttr("bcd", AnyValue::CreateFrom(1243124));

  NamedAttrs out_nas;
  EXPECT_TRUE(AttrUtils::GetNamedAttrs(op_desc, "attr", out_nas));
  EXPECT_EQ(out_nas.GetName(), nas.GetName());
  AnyValue av;
  EXPECT_EQ(out_nas.GetAttr("abc", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<int64_t>(), nullptr);
  EXPECT_EQ(*av.Get<int64_t>(), 10);

  EXPECT_EQ(out_nas.GetAttr("bcd", av), GRAPH_SUCCESS);
  EXPECT_NE(av.Get<bool>(), nullptr);
  EXPECT_EQ(*av.Get<bool>(), true);
}

TEST_F(AttrUtilsUt, SetGetListNamedAttrs) {
  auto op_desc = std::make_shared<OpDesc>();
  std::vector<NamedAttrs> nass(5);
  for (size_t i = 0; i < nass.size(); ++i) {
    auto &nas = nass[i];
    nas.SetName(std::string("name_") + std::to_string(i));
    nas.SetAttr("abc", AnyValue::CreateFrom(static_cast<int32_t>(rand())));
  }

  EXPECT_TRUE(AttrUtils::SetListNamedAttrs(op_desc, "attr", nass));

  std::vector<NamedAttrs> out_nass;
  EXPECT_TRUE(AttrUtils::GetListNamedAttrs(op_desc, "attr", out_nass));
  EXPECT_EQ(out_nass.size(), 5);
  for (size_t i = 0; i < out_nass.size(); ++i) {
    auto &out_nas = out_nass[i];
    auto &nas = nass[i];
    EXPECT_EQ(out_nas.GetName(), nas.GetName());
    AnyValue out_av, av;
    EXPECT_EQ(out_nas.GetAttr("abc", out_av), GRAPH_SUCCESS);
    EXPECT_EQ(nas.GetAttr("abc", av), GRAPH_SUCCESS);
    EXPECT_EQ(*out_av.Get<int32_t>(), *av.Get<int32_t>());
  }
}
}
