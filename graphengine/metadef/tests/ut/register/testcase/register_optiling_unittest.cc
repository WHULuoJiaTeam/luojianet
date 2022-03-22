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
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling.cc"

using namespace std;
using namespace ge;

namespace optiling {
using ByteBuffer = std::stringstream;
class RegisterOpTilingUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(RegisterOpTilingUT, byte_buffer_test) {
  ByteBuffer stream;
  std::string str = "test";

  ByteBuffer &str_stream1 = ByteBufferPut(stream, str);

  string value;
  ByteBuffer &str_stream2 = ByteBufferGet(stream, str);

  char *dest = nullptr;
  size_t size = ByteBufferGetAll(stream, dest, 2);
  cout << size << endl;
}

TEST_F(RegisterOpTilingUT, op_run_info_test) {
  std::shared_ptr<utils::OpRunInfo> run_info = make_shared<utils::OpRunInfo>(8, true, 64);
  int64_t work_space;
  graphStatus ret = run_info->GetWorkspace(0, work_space);
  EXPECT_EQ(ret, GRAPH_FAILED);
  vector<int64_t> work_space_vec = {10, 20, 30, 40};
  run_info->SetWorkspaces(work_space_vec);
  ret = run_info->GetWorkspace(1, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 20);
  EXPECT_EQ(run_info->GetWorkspaceNum(), 4);
  string str = "test";
  run_info->AddTilingData(str);

  const ByteBuffer &tiling_data = run_info->GetAllTilingData();

  std::shared_ptr<utils::OpRunInfo> run_info_2 = make_shared<utils::OpRunInfo>(*run_info);
  ret = run_info_2->GetWorkspace(2, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 30);

  utils::OpRunInfo run_info_3 = *run_info;
  ret = run_info_3.GetWorkspace(3, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 40);

  utils::OpRunInfo &run_info_4 = *run_info;
  ret = run_info_4.GetWorkspace(0, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 10);
}

TEST_F(RegisterOpTilingUT, op_compile_info_test) {
  std::shared_ptr<utils::OpCompileInfo> compile_info = make_shared<utils::OpCompileInfo>();
  string str_key = "key";
  string str_value = "value";
  AscendString key(str_key.c_str());
  AscendString value(str_value.c_str());
  compile_info->SetKey(key);
  compile_info->SetValue(value);

  std::shared_ptr<utils::OpCompileInfo> compile_info_2 = make_shared<utils::OpCompileInfo>(key, value);
  EXPECT_EQ(compile_info_2->GetKey() == key, true);
  EXPECT_EQ(compile_info_2->GetValue() == value, true);

  std::shared_ptr<utils::OpCompileInfo> compile_info_3 = make_shared<utils::OpCompileInfo>(str_key, str_value);
  EXPECT_EQ(compile_info_3->GetKey() == key, true);
  EXPECT_EQ(compile_info_3->GetValue() == value, true);

  std::shared_ptr<utils::OpCompileInfo> compile_info_4 = make_shared<utils::OpCompileInfo>(*compile_info);
  EXPECT_EQ(compile_info_4->GetKey() == key, true);
  EXPECT_EQ(compile_info_4->GetValue() == value, true);

  utils::OpCompileInfo compile_info_5 = *compile_info;
  EXPECT_EQ(compile_info_5.GetKey() == key, true);
  EXPECT_EQ(compile_info_5.GetValue() == value, true);

  utils::OpCompileInfo &compile_info_6 = *compile_info;
  EXPECT_EQ(compile_info_6.GetKey() == key, true);
  EXPECT_EQ(compile_info_6.GetValue() == value, true);
}

TEST_F(RegisterOpTilingUT, te_op_paras_test) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape({1,4,1,1});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  int32_t attr_value = 1024;
  AttrUtils::SetInt(op_desc, "some_int_attr", attr_value);
  vector<int64_t> attr_vec = {11, 22, 33, 44};
  AttrUtils::SetListInt(op_desc, "some_int_vec", attr_vec);
  TeOpParas op_param;
  op_param.op_type = op_desc->GetType();
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  op_param.var_attrs.GetData("some_int_attr", "xxx", size);
  op_param.var_attrs.GetData("some_int_attr", "Int32", size);
  op_param.var_attrs.GetData("some_int_vec", "ListInt32", size);
}
}  // namespace ge