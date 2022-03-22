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
#include "gtest/gtest.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
GE_NS_BEGIN

class OpDescCfgTest : public testing::Test {};

TEST_F(OpDescCfgTest, test_attr_set_string_success) {
  auto op_ptr = OP_CFG(DATA).Attr(ENTER_ATTR_FRAME_NAME, "1").Build("data1");

  ge::GeAttrValue ret;
  op_ptr->GetAttr(ENTER_ATTR_FRAME_NAME, ret);
  std::string value;
  ret.GetValue<std::string>(value);

  ASSERT_EQ(value, "1");
}

TEST_F(OpDescCfgTest, test_attr_set_int_success) {
  auto op_ptr = OP_CFG(DATA).Attr(ENTER_ATTR_FRAME_NAME, 2).Build("data1");

  ge::GeAttrValue ret;
  op_ptr->GetAttr(ENTER_ATTR_FRAME_NAME, ret);
  int64_t value;
  ret.GetValue<int64_t>(value);

  ASSERT_EQ(value, 2);
}

TEST_F(OpDescCfgTest, test_attr_set_perent_node_index_success) {
  auto op_ptr = OP_CFG(DATA).ParentNodeIndex(2).Build("data1");

  ge::GeAttrValue ret;
  op_ptr->GetAttr(ATTR_NAME_PARENT_NODE_INDEX, ret);
  int64_t value;
  ret.GetValue<int64_t>(value);

  ASSERT_EQ(value, 2);
}

TEST_F(OpDescCfgTest, test_attr_set_weight_success) {
  int64_t dims_size = 1;
  vector<int64_t> data_vec = {5};
  for_each(data_vec.begin(), data_vec.end(), [&](int64_t &data) { dims_size *= data; });
  vector<int32_t> data_value_vec(dims_size, 1);
  GeTensorDesc data_tensor_desc(GeShape(data_vec), FORMAT_NCHW, DT_INT32);
  GeTensorPtr data_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec.data(),
                                                       data_value_vec.size() * sizeof(int32_t));

  auto op_ptr = OP_CFG(CONSTANT).Weight(data_tensor).Build("const1");

  ConstGeTensorPtr tensor_value;
  ASSERT_TRUE(AttrUtils::GetTensor(op_ptr, ge::ATTR_NAME_WEIGHTS, tensor_value));
  ASSERT_EQ(tensor_value->GetTensorDesc().GetDataType(), DT_INT32);
}

GE_NS_END
