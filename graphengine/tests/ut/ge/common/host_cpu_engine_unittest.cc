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

#define protected public
#define private public
#include "ge_local_engine/engine/host_cpu_engine.h"
#undef private
#undef protected

namespace ge {
class UTEST_host_cpu_engine : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTEST_host_cpu_engine, PrepareOutputs_success) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  op_desc->AddOutputDesc("1", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_BOOL));
  op_desc->AddOutputDesc("2", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT8));
  op_desc->AddOutputDesc("3", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT16));
  op_desc->AddOutputDesc("4", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc("5", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT64));
  op_desc->AddOutputDesc("6", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_UINT8));
  op_desc->AddOutputDesc("7", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_UINT16));
  op_desc->AddOutputDesc("8", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_UINT32));
  op_desc->AddOutputDesc("9", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_UINT64));
  op_desc->AddOutputDesc("10", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_FLOAT16));
  op_desc->AddOutputDesc("11", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_FLOAT));
  op_desc->AddOutputDesc("12", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_DOUBLE));
  op_desc->AddOutputDesc("13", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT4));

  vector<GeTensorPtr> outputs;
  GeTensorPtr value = std::make_shared<GeTensor>();
  for (int32_t i = 0; i < 13; i++) {
    outputs.push_back(value);
  }

  map<std::string, Tensor> named_outputs;
  auto ret = HostCpuEngine::GetInstance().PrepareOutputs(op_desc, outputs, named_outputs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(named_outputs.size(), 13);
}

TEST_F(UTEST_host_cpu_engine, PrepareOutputs_need_create_success) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("name", "type");
  op_desc->AddOutputDesc("output_1", GeTensorDesc(GeShape({2, 2}), FORMAT_NCHW, DT_INT32));

  vector<GeTensorPtr> outputs;
  map<std::string, Tensor> named_outputs;
  auto ret = HostCpuEngine::GetInstance().PrepareOutputs(op_desc, outputs, named_outputs);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(named_outputs.size(), 1);
  EXPECT_EQ(named_outputs["output_1"].GetSize(), 16);
  EXPECT_EQ(named_outputs["output_1"].GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(named_outputs["output_1"].GetTensorDesc().GetShape().GetShapeSize(), 4);
}
}  // namespace ge