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
#include <memory>



#define protected public
#define private public
#include "graph/manager/host_mem_manager.h"
#include "inc/framework/memory/memory_api.h"
#undef protected
#undef private
#include "metadef/inc/graph/aligned_ptr.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestMemoryApiTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestMemoryApiTest, query_mem_info_success) {
string var_name = "host_params";
SharedMemInfo info;
uint8_t tmp(0);
info.device_address = &tmp;

std::shared_ptr<AlignedPtr> aligned_ptr = std::make_shared<AlignedPtr>(100, 16);

info.host_aligned_ptr = aligned_ptr;
info.fd=0;
info.mem_size = 100;
info.op_name = var_name;
HostMemManager::Instance().var_memory_base_map_[var_name] = info;
uint64_t base_addr;
uint64_t var_size;
Status ret = GetVarBaseAddrAndSize(var_name, base_addr, var_size);
EXPECT_EQ(ret, SUCCESS);
EXPECT_EQ(var_size, 100);
HostMemManager::Instance().var_memory_base_map_.clear();
}

TEST_F(UtestMemoryApiTest, query_mem_info_failed) {
string var_name = "host_params";
uint64_t base_addr;
uint64_t var_size;
Status ret = GetVarBaseAddrAndSize(var_name, base_addr, var_size);
EXPECT_NE(ret, SUCCESS);
}
