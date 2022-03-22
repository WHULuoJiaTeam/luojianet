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

#include "common/dump/dump_properties.h"
#include "ge_local_context.h"
#include "ge/ge_api_types.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"

namespace ge {
class UTEST_dump_properties : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTEST_dump_properties, check_dump_step) {
  DumpProperties dp;
  std::string dump_step{"0|3-5|10"};
  std::string unsupport_input1{"0|5-3|10"};
  std::string unsupport_input2{"one"};
  std::string unsupport_input3;
  for (int i = 0; i < 200; ++i) {
    unsupport_input3 += std::to_string(i) + "|";
  }
  unsupport_input3.pop_back();
  Status st = dp.CheckDumpStep(dump_step);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpStep(unsupport_input1);
  EXPECT_NE(st, SUCCESS);
  st = dp.CheckDumpStep(unsupport_input2);
  EXPECT_NE(st, SUCCESS);
  st = dp.CheckDumpStep(unsupport_input3);
  EXPECT_NE(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, check_dump_mode) {
  DumpProperties dp;
  std::string dump_mode_1{"input"};
  std::string dump_mode_2{"output"};
  std::string dump_mode_3{"all"};
  std::string unsupport_input1{"mode1"};
  Status st = dp.CheckDumpMode(dump_mode_1);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpMode(dump_mode_2);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpMode(dump_mode_3);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpMode(unsupport_input1);
  EXPECT_NE(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, check_dump_path) {
  DumpProperties dp;
  std::string dump_path{"/tmp/"};
  std::string unsupport_input1{"  \\unsupported"};
  Status st = dp.CheckDumpPath(dump_path);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckDumpPath(unsupport_input1);
  EXPECT_NE(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, check_enable_dump) {
  DumpProperties dp;
  std::string enable_dump_t{"1"};
  std::string enable_dump_f{"0"};
  std::string unsupport_input1{"true"};
  std::string unsupport_input2{"false"};
  Status st = dp.CheckEnableDump(enable_dump_t);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckEnableDump(enable_dump_f);
  EXPECT_EQ(st, SUCCESS);
  st = dp.CheckEnableDump(unsupport_input1);
  EXPECT_NE(st, SUCCESS);
  st = dp.CheckEnableDump(unsupport_input2);
  EXPECT_NE(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, init_by_options_success_1) {
  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_STEP, "0|1-3|10"},
                                              {OPTION_EXEC_DUMP_MODE, "all"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, init_by_options_success_2) {
  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"},
                                              {OPTION_EXEC_DUMP_DEBUG_MODE, "aicore_overflow"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
}

TEST_F(UTEST_dump_properties, init_by_options_success_3) {
  DumpProperties dp;
  std::map<std::string, std::string> options {{OPTION_EXEC_ENABLE_DUMP_DEBUG, "1"},
                                              {OPTION_EXEC_DUMP_PATH, "/tmp/"}};
  GetThreadLocalContext().SetGlobalOption(options);
  Status st = dp.InitByOptions();
  EXPECT_EQ(st, SUCCESS);
}
}  // namespace ge