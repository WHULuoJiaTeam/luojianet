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

#include <cce/compiler_stub.h>
#include <gtest/gtest.h>
#include <sys/time.h>
#include <unistd.h>

#include "common/debug/log.h"
#include "common/l2_cache_optimize.h"
#include "common/model_parser/model_parser.h"
#include "common/properties_manager.h"
#include "common/types.h"

#define private public
#define protected public
#include "common/helper/om_file_helper.h"
#include "common/op/ge_op_utils.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
//#include "new_op_test_utils.h"
#undef private
#undef protected

using namespace std;
using namespace testing;

namespace ge {

const static std::string ENC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

class UtestModelManagerModelManagerAicpu : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestModelManagerModelManagerAicpu, checkAicpuOptype) {
  ModelManager model_manager;
  uint32_t model_id = 0;
  std::vector<std::string> aicpu_op_list;
  std::vector<std::string> aicpu_tf_list;
  aicpu_tf_list.emplace_back("FrameworkOp");
  aicpu_tf_list.emplace_back("Unique");

  model_manager.LaunchKernelCheckAicpuOp(aicpu_op_list, aicpu_tf_list);
  // Load allow listener is null
  // EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, nullptr, nullptr));
}

TEST_F(UtestModelManagerModelManagerAicpu, DestroyAicpuKernel) {
  ModelManager model_manager;
  uint32_t model_id = 0;
  std::vector<std::string> aicpu_op_list;
  std::vector<std::string> aicpu_tf_list;
  aicpu_tf_list.emplace_back("FrameworkOp");
  aicpu_tf_list.emplace_back("Unique");

  model_manager.DestroyAicpuKernel(0,0,0);
  // Load allow listener is null
  // EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, nullptr, nullptr));
}

// test GenSessionId
TEST_F(UtestModelManagerModelManagerAicpu, gen_session_id) {
  ModelManager manager;
  uint64_t session_id;
  manager.GenSessionId(session_id);

  struct timeval tv;
  gettimeofday(&tv, nullptr);
  uint64_t timestamp = static_cast<uint64_t>(tv.tv_sec * 1000000);

  const uint64_t kSessionTimeMask = 0xfffffff000000000; // 不比us
  const uint64_t kSessionPidMask  = 0x000000000000ff00;
  const uint64_t kSessionBiasMask = 0x00000000000000ff;

  uint32_t pid = getpid();

  EXPECT_EQ(1, kSessionBiasMask & session_id);
  EXPECT_EQ(pid<<8 & kSessionPidMask, kSessionPidMask & session_id);
  //EXPECT_EQ(timestamp<<16 & kSessionTimeMask, kSessionTimeMask & session_id);
}


}  // namespace ge
