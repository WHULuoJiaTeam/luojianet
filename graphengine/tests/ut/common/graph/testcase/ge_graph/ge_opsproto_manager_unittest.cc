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
#include <vector>
#include "graph/opsproto_manager.h"
#undef protected
#undef private

using namespace ge;
using namespace testing;
using namespace std;

class UtestOpsprotoManager : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

TEST_F(UtestOpsprotoManager, initialize_failure) {
  OpsProtoManager *manager = OpsProtoManager::Instance();
  std::map<std::string, std::string> options;
  options["a"] = "a";
  bool ret = manager->Initialize(options);
  EXPECT_EQ(ret, false);

  options["ge.opsProtoLibPath"] = "";
  ret = manager->Initialize(options);
  EXPECT_EQ(ret, true);

  options["ge.opsProtoLibPath"] = "path1:path2";
  ret = manager->Initialize(options);
  EXPECT_EQ(ret, true);

  options["ge.opsProtoLibPath"] = "/usr/local/HiAI/path1.so:$ASCEND_HOME/path2";
  EXPECT_EQ(ret, true);

  mkdir("test_ops_proto_manager", S_IRUSR);

  options["ge.opsProtoLibPath"] = "test_ops_proto_manager";
  ret = manager->Initialize(options);
  EXPECT_EQ(ret, true);
  rmdir("test_proto_manager");

  manager->Finalize();
}