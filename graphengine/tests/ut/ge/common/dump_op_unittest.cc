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
#include "common/dump/dump_op.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#include "common/dump/dump_properties.h"
#undef private
#undef protected

namespace ge {
class UTEST_dump_op : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTEST_dump_op, launch_dump_op_success) {
  DumpOp dump_op;
  DumpProperties dump_properties;
  OpDescPtr op_desc = std::make_shared<OpDesc>("GatherV2", "GatherV2");
  std::set<std::string> temp;
  dump_properties.model_dump_properties_map_.emplace("model1", temp);
  dump_properties.enable_dump_ = "1";
  dump_op.SetDynamicModelInfo("model1", "model2", 1);
  dump_op.SetDumpInfo(dump_properties, op_desc, {}, {}, nullptr);
  auto ret = dump_op.LaunchDumpOp();
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_dump_op, launch_dump_op_success_2) {
  DumpOp dump_op;
  DumpProperties dump_properties;
  OpDescPtr op_desc = std::make_shared<OpDesc>("GatherV2", "GatherV2");
  std::set<std::string> temp;
  dump_properties.model_dump_properties_map_.emplace("model1", temp);
  dump_properties.enable_dump_ = "1";
  dump_op.SetDynamicModelInfo("modle2", "model2", 1);
  dump_op.SetDumpInfo(dump_properties, op_desc, {}, {}, nullptr);
  auto ret = dump_op.LaunchDumpOp();
  EXPECT_EQ(ret, ge::SUCCESS);
}

}  // namespace ge