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

#define private public
#define protected public
#include "graph/load/model_manager/data_dumper.h"
#include "graph/load/model_manager/davinci_model.h"
#undef private
#undef protected

using namespace std;

namespace ge {
class UtestDataDumper : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

std::vector<void *> stub_get_output_addrs(const RuntimeParam &model_param, ConstOpDescPtr op_desc) {
  std::vector<void *> res;
  res.emplace_back(reinterpret_cast<void *>(23333));
  return res;
}

static ge::OpDescPtr CreateOpDesc(string name = "", string type = "") {
  auto op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  op_desc->SetId(0);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetInputOffset({});
  op_desc->SetOutputOffset({100, 200});
  return op_desc;
}

TEST_F(UtestDataDumper, LoadDumpInfo_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);
  std::shared_ptr<OpDesc> op_desc_1(new OpDesc());
  op_desc_1->AddOutputDesc("test", GeTensorDesc());
  data_dumper.SaveDumpTask(0, 0, op_desc_1, 0);
  string dump_mode = "output";
  data_dumper.is_op_debug_ = true;
  data_dumper.dump_properties_.SetDumpMode(dump_mode);
  EXPECT_EQ(data_dumper.LoadDumpInfo(), SUCCESS);
  EXPECT_EQ(data_dumper.UnloadDumpInfo(), SUCCESS);
}

TEST_F(UtestDataDumper, DumpOutputWithTask_success) {
  RuntimeParam rts_param;
  DataDumper data_dumper(&rts_param);
  data_dumper.SetModelName("test");
  data_dumper.SetModelId(2333);

  toolkit::aicpu::dump::Task task;
  OpDescPtr op_desc = CreateOpDesc("conv", CONVOLUTION);
  GeTensorDesc tensor_0(GeShape(), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc tensor_1(GeShape(), FORMAT_NCHW, DT_FLOAT);
  int32_t calc_type = 1;
  ge::AttrUtils::SetInt(tensor_1, ATTR_NAME_MEMORY_SIZE_CALC_TYPE, calc_type);
  op_desc->AddOutputDesc(tensor_0);
  op_desc->AddOutputDesc(tensor_1);
  DataDumper::InnerDumpInfo inner_dump_info;
  inner_dump_info.op = op_desc;
  Status ret = data_dumper.DumpOutputWithTask(inner_dump_info, task);
  EXPECT_EQ(ret, SUCCESS);
  int64_t task_size = 1;
  data_dumper.GenerateOpBuffer(task_size, task);
}
}  // namespace ge
