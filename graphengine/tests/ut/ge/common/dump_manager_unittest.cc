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
#include "common/dump/dump_manager.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"

namespace ge {
class UTEST_dump_manager : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
 TEST_F(UTEST_dump_manager, is_dump_open_success) {
   DumpConfig dump_config;
   dump_config.dump_path = "/test";
   dump_config.dump_mode = "all";
   dump_config.dump_status = "on";
   dump_config.dump_op_switch = "on";
   auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
   auto dump = DumpManager::GetInstance().GetDumpProperties(0);
   bool result = dump.IsDumpOpen();
   dump.ClearDumpInfo();
   EXPECT_EQ(result, true);
 }

 TEST_F(UTEST_dump_manager, is_dump_op_success) {
   DumpConfig dump_config;
   dump_config.dump_path = "/test";
   dump_config.dump_mode = "all";
   dump_config.dump_status = "off";
   auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
   EXPECT_EQ(ret, ge::SUCCESS);
 }

TEST_F(UTEST_dump_manager, is_dump_single_op_close_success) {
   DumpConfig dump_config;
   dump_config.dump_path = "/test";
   dump_config.dump_mode = "all";
   dump_config.dump_status = "on";
   dump_config.dump_op_switch = "off";
   auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
   EXPECT_EQ(ret, ge::PARAM_INVALID);
 }

 TEST_F(UTEST_dump_manager, dump_status_empty) {
   DumpConfig dump_config;
   dump_config.dump_path = "/test";
   dump_config.dump_mode = "all";
   dump_config.dump_op_switch = "off";
   auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
   EXPECT_EQ(ret, ge::SUCCESS);
 }

 // dump_debug and debug_status are on
  TEST_F(UTEST_dump_manager, dump_op_debug_on) {
    DumpConfig dump_config;
    dump_config.dump_debug = "on";
    dump_config.dump_status = "on";
    auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
    EXPECT_EQ(ret, ge::SUCCESS);
  }

  // just dump_status is on
  TEST_F(UTEST_dump_manager, dump_status_without_dump_list) {
    DumpConfig dump_config;
    dump_config.dump_status = "on";
    auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
    EXPECT_EQ(ret, ge::PARAM_INVALID);
  }

  // dump_status is on with dump_list
  TEST_F(UTEST_dump_manager, dump_status_with_dump_list) {
    DumpConfig dump_config;
    dump_config.dump_status = "on";
    ModelDumpConfig dump_list;
    dump_list.model_name = "test";
    dump_list.layers.push_back("first");
    dump_config.dump_list.push_back(dump_list);
    auto ret = DumpManager::GetInstance().SetDumpConf(dump_config);
    EXPECT_EQ(ret, ge::PARAM_INVALID);
  }

 TEST_F(UTEST_dump_manager, add_dump_properties_success) {
   DumpProperties dump_properties;
   DumpManager::GetInstance().AddDumpProperties(0, dump_properties);
   auto dump = DumpManager::GetInstance().GetDumpProperties(0);
   DumpManager::GetInstance().RemoveDumpProperties(0);
 }

 TEST_F(UTEST_dump_manager, not_need_do_dump) {
   DumpConfig dump_config;
   dump_config.dump_status = "off";
   dump_config.dump_debug = "off";
   DumpProperties dump_properties;
   bool ret = DumpManager::GetInstance().NeedDoDump(dump_config, dump_properties);
   EXPECT_EQ(ret, false);
 }
}  // namespace ge