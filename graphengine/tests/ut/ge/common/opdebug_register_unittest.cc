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

#include "common/dump/opdebug_register.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"

namespace ge {
class UTEST_opdebug_register : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
 
TEST_F(UTEST_opdebug_register, register_debug_for_model_success) {
  OpdebugRegister opdebug_register;
  rtModel_t model_handle = (void*)0x111;
  uint32_t op_debug_mode = 1;
  DataDumper data_dumper({});
  auto ret = opdebug_register.RegisterDebugForModel(model_handle, op_debug_mode, data_dumper);
  opdebug_register.UnregisterDebugForModel(model_handle);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_opdebug_register, register_debug_for_stream_success) {
  OpdebugRegister opdebug_register;
  rtStream_t stream = (void*)0x111;
  uint32_t op_debug_mode = 1;
  DataDumper data_dumper({});
  auto ret = opdebug_register.RegisterDebugForStream(stream, op_debug_mode, data_dumper);
  opdebug_register.UnregisterDebugForStream(stream);
  EXPECT_EQ(ret, ge::SUCCESS);
}


}  // namespace ge