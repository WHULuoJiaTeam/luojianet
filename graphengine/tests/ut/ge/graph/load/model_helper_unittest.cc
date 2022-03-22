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
#include "framework/common/helper/model_helper.h"
#include "framework/omg/model_tool.h"
#include "framework/omg/ge_init.h"
#include "ge/common/model/ge_model.h"
#undef private
#undef protected

#include "proto/task.pb.h"

using namespace std;

namespace ge {
class UtestModelHelper : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestModelHelper, save_size_to_modeldef)
{
  GeModelPtr ge_model = ge::MakeShared<ge::GeModel>();
  std::shared_ptr<domi::ModelTaskDef> task = ge::MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(task);
  ModelHelper model_helper;
  EXPECT_EQ(SUCCESS, model_helper.SaveSizeToModelDef(ge_model));
}

TEST_F(UtestModelHelper, atc_test)
{
  ge::proto::ModelDef model_def;
  uint32_t modeldef_size = 0;

  GEInit::Finalize();
  char buffer[1024];
  getcwd(buffer, 1024);
  string path=buffer;
  string file_path=path + "/Makefile";

  ModelTool::GetModelInfoFromOm(file_path.c_str(), model_def, modeldef_size);
  ModelTool::GetModelInfoFromOm("123.om", model_def, modeldef_size);
  ModelTool::GetModelInfoFromPbtxt(file_path.c_str(), model_def);
  ModelTool::GetModelInfoFromPbtxt("123.pbtxt", model_def);
}
}  // namespace ge
