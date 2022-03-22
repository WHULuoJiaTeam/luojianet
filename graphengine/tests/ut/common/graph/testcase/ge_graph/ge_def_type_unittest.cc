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

#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

using namespace std;
using namespace ge;

class UtestGeTestDefType : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeTestDefType, quant) {
  OpDescPtr desc_ptr1 = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr1->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr1->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(desc_ptr1), false);
  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(*desc_ptr1), false);
}
