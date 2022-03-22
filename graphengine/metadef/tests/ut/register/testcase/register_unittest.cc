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
#include "external/register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"

using namespace domi;
using namespace ge;
class UtestRegister : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestRegister, base) {
 
}

TEST_F(UtestRegister, test_register_dynamic_outputs_op_only_has_partial_output) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node_src = builder.AddNode("ParseSingleExample", "ParseSingleExample", 
                    {"serialized", "dense_defaults_0", "dense_defaults_1", "dense_defaults_2"}, 
                    {"dense_values_0", "dense_values_1", "dense_values_2"});
  // build op_src attrs
  vector<string> dense_keys = {"image/class/lable","image/encode", "image/format"};
  vector<DataType> t_dense = {DT_INT64, DT_STRING, DT_STRING};
  AttrUtils::SetListStr(node_src->GetOpDesc(), "dense_keys", dense_keys);
  AttrUtils::SetListStr(node_src->GetOpDesc(), "dense_shapes", {});
  AttrUtils::SetInt(node_src->GetOpDesc(), "num_sparse", 0);
  AttrUtils::SetListStr(node_src->GetOpDesc(), "sparse_keys", {});
  AttrUtils::SetListStr(node_src->GetOpDesc(), "sparse_types", {});
  AttrUtils::SetListDataType(node_src->GetOpDesc(), "Tdense", t_dense);
  auto graph = builder.GetGraph();

  // get op_src
  ge::Operator op_src = OpDescUtils::CreateOperatorFromNode(node_src);
  ge::Operator op_dst = ge::Operator("ParseSingleExample");
  std::shared_ptr<ge::OpDesc> op_desc_dst = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  op_desc_dst->AddRegisterInputName("dense_defaults");
  op_desc_dst->AddRegisterOutputName("sparse_indices");
  op_desc_dst->AddRegisterOutputName("sparse_values");
  op_desc_dst->AddRegisterOutputName("sparse_shapes");
  op_desc_dst->AddRegisterOutputName("dense_values");

  // simulate parse_single_example plugin
  std::vector<DynamicInputOutputInfo> value;
  DynamicInputOutputInfo input(kInput, "dense_defaults", 14, "Tdense", 6);
  value.push_back(input);
  DynamicInputOutputInfo output(kOutput, "sparse_indices", 14, "num_sparse", 10);
  value.push_back(output);
  DynamicInputOutputInfo output1(kOutput, "sparse_values", 13, "sparse_types", 12);
  value.push_back(output1);
  DynamicInputOutputInfo output2(kOutput, "sparse_shapes", 13, "num_sparse", 10);
  value.push_back(output2);
  DynamicInputOutputInfo output3(kOutput, "dense_values", 12, "Tdense", 6);
  value.push_back(output3);
  // pre_check
  EXPECT_EQ(op_dst.GetOutputsSize(), 0);
  auto ret = AutoMappingByOpFnDynamic(op_src, op_dst, value);

  // check add 3 output to op_dst
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(op_dst.GetOutputsSize(), 3);
}
