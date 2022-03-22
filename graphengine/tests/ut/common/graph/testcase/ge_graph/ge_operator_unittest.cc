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

#define private public
#define protected public
#include "graph/operator.h"

#include "graph/def_types.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/graph.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#undef private
#undef protected

using namespace std;
using namespace ge;

class UtestGeOperator : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
  string vec2str(vector<uint8_t> &vec) {
    string str((char *)vec.data(), vec.size());
    return str;
  }
};

TEST_F(UtestGeOperator, try_get_input_desc) {
  Operator data("data0");

  TensorDesc td;
  graphStatus ret = data.TryGetInputDesc("const", td);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGeOperator, get_dynamic_input_num) {
  Operator const_node("constNode");

  (void)const_node.DynamicInputRegister("data", 2, 1);
  int num = const_node.GetDynamicInputNum("data");
  EXPECT_EQ(num, 2);
}

TEST_F(UtestGeOperator, infer_format_func_register) {
  Operator add("add");
  std::function<graphStatus(Operator &)> func = nullptr;
  add.InferFormatFuncRegister(func);
}

graphStatus TestFunc(Operator &op) { return 0; }
TEST_F(UtestGeOperator, get_infer_format_func_register) {
  (void)OperatorFactoryImpl::GetInferFormatFunc("add");
  std::function<graphStatus(Operator &)> func = TestFunc;
  OperatorFactoryImpl::RegisterInferFormatFunc("add", TestFunc);
  (void)OperatorFactoryImpl::GetInferFormatFunc("add");
}

TEST_F(UtestGeOperator, get_attr_names_and_types) {
  Operator attr("attr");
  (void)attr.GetAllAttrNamesAndTypes();
}