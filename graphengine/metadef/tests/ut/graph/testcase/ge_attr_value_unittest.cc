/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "graph/op_desc.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include <string>

namespace ge {
class UtestGeAttrValue : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};


TEST_F(UtestGeAttrValue, GetAttrsStrAfterRid) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(AttrUtils::GetAttrsStrAfterRid(op_desc, {}), "");

  std::set<std::string> names ={"qazwsx", "d"};
  op_desc->SetAttr("qazwsx", GeAttrValue::CreateFrom<int64_t>(132));
  op_desc->SetAttr("xswzaq", GeAttrValue::CreateFrom<int64_t>(123));
  auto tensor = GeTensor();
  op_desc->SetAttr("value", GeAttrValue::CreateFrom<GeTensor>(tensor));
  std::string res = AttrUtils::GetAttrsStrAfterRid(op_desc, names);
  EXPECT_TRUE(res.find("qazwsx") == string::npos);
  EXPECT_TRUE(res.find("xswzaq") != string::npos);
}

TEST_F(UtestGeAttrValue, GetAllAttrsStr) {
 // 属性序列化
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  EXPECT_EQ(AttrUtils::GetAllAttrsStr(op_desc), "");
  op_desc->SetAttr("seri_i", GeAttrValue::CreateFrom<int64_t>(1));
  auto tensor = GeTensor();
  op_desc->SetAttr("seri_value", GeAttrValue::CreateFrom<GeTensor>(tensor));
  op_desc->SetAttr("seri_input_desc", GeAttrValue::CreateFrom<GeTensorDesc>(GeTensorDesc()));
  string attr = AttrUtils::GetAllAttrsStr(op_desc);

  EXPECT_TRUE(attr.find("seri_i") != string::npos);
  EXPECT_TRUE(attr.find("seri_value") != string::npos);
  EXPECT_TRUE(attr.find("seri_input_desc") != string::npos);

}
TEST_F(UtestGeAttrValue, GetAllAttrs) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  op_desc->SetAttr("i", GeAttrValue::CreateFrom<int64_t>(100));
  op_desc->SetAttr("input_desc", GeAttrValue::CreateFrom<GeTensorDesc>(GeTensorDesc()));
  auto attrs = AttrUtils::GetAllAttrs(op_desc);
  EXPECT_EQ(attrs.size(), 2);
  int64_t attr_value = 0;
  EXPECT_EQ(attrs["i"].GetValue(attr_value), GRAPH_SUCCESS);
  EXPECT_EQ(attr_value, 100);

}

TEST_F(UtestGeAttrValue, TrySetExists) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);

  int64_t attr_value = 0;

  EXPECT_FALSE(AttrUtils::GetInt(op_desc, "i", attr_value));
  op_desc->TrySetAttr("i", GeAttrValue::CreateFrom<int64_t>(100));
  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "i", attr_value));
  EXPECT_EQ(attr_value, 100);

  op_desc->TrySetAttr("i", GeAttrValue::CreateFrom<int64_t>(102));
  attr_value = 0;
  AttrUtils::GetInt(op_desc, "i", attr_value);
  EXPECT_EQ(attr_value, 100);
}


TEST_F(UtestGeAttrValue, SetGetListInt) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  EXPECT_TRUE(op_desc);

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "li1", std::vector<int64_t>({1,2,3,4,5})));
  std::vector<int64_t> li1_out0;
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "li1", li1_out0));
  EXPECT_EQ(li1_out0, std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(UtestGeAttrValue, SetListIntGetByGeAttrValue) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  EXPECT_TRUE(op_desc);

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "li1", std::vector<int64_t>({1,2,3,4,5})));
  auto names_to_value = AttrUtils::GetAllAttrs(op_desc);
  auto iter = names_to_value.find("li1");
  EXPECT_NE(iter, names_to_value.end());

  std::vector<int64_t> li1_out;
  auto &ge_value = iter->second;
  EXPECT_EQ(ge_value.GetValue(li1_out), GRAPH_SUCCESS);
  EXPECT_EQ(li1_out, std::vector<int64_t>({1,2,3,4,5}));

  li1_out.clear();
  EXPECT_EQ(ge_value.GetValue<std::vector<int64_t>>(li1_out), GRAPH_SUCCESS);
  EXPECT_EQ(li1_out, std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(UtestGeAttrValue, SetGetAttr_GeTensor) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1,100})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1,100})));
  td.SetDataType(DT_FLOAT);
  td.SetFormat(FORMAT_ND);
  float data[100];
  for (size_t i = 0; i < 100; ++i) {
    data[i] = 1.0 * i;
  }
  auto tensor = std::make_shared<GeTensor>(td, reinterpret_cast<const uint8_t *>(data), sizeof(data));
  EXPECT_NE(tensor, nullptr);

  EXPECT_TRUE(AttrUtils::SetTensor(op_desc, "t", tensor));
  tensor = nullptr;

  EXPECT_TRUE(AttrUtils::MutableTensor(op_desc, "t", tensor));
  EXPECT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->GetData().GetSize(), sizeof(data));
  auto attr_data = reinterpret_cast<const float *>(tensor->GetData().GetData());
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(attr_data[i], data[i]);
  }
  tensor = nullptr;

  EXPECT_TRUE(AttrUtils::MutableTensor(op_desc, "t", tensor));
  EXPECT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->GetData().GetSize(), sizeof(data));
  attr_data = reinterpret_cast<const float *>(tensor->GetData().GetData());
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(attr_data[i], data[i]);
  }
  tensor = nullptr;
}
}
