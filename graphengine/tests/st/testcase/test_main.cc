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
#include "external/ge/ge_api.h"
#include "ge_graph_dsl/assert/check_utils.h"
#include "ge_running_env/include/ge_running_env/ge_running_env_faker.h"

using namespace std;
using namespace ge;

int main(int argc, char **argv) {
  // init the logging
  map<AscendString, AscendString> options;
  auto init_status = ge::GEInitialize(options);
  if (init_status != SUCCESS) {
    std::cout << "ge init failed , ret code:" << init_status << endl;
  }
  GeRunningEnvFaker::BackupEnv();
  CheckUtils::init();
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  return ret;
}
