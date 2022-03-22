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
#include "common/dump/exception_dumper.h"
#include "common/debug/log.h"
#include "common/ge_inner_error_codes.h"
#undef private
#undef protected

namespace ge {
class UTEST_dump_exception : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UTEST_dump_exception, save_dump_op_info_success) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("GatherV2", "GatherV2");
  uint32_t task_id = 1;
  uint32_t stream_id = 233;
  vector<void *> input_addr;
  vector<void *> output_addr;
  ExceptionDumper exception_dumper;
  exception_dumper.SaveDumpOpInfo(op_desc, task_id, stream_id, input_addr, output_addr);
}

TEST_F(UTEST_dump_exception, dump_exception_info) {
  rtExceptionInfo exception_info = {1, 2, 3, 4, 5};
  std::vector<rtExceptionInfo> exception_infos = { exception_info };
  OpDescInfo op_desc_info = {"Save", "Save", 1, 2, {FORMAT_NCHW}, {{1}}, {DT_FLOAT}, {}, {2},
                             {FORMAT_NCHW}, {{1}}, {DT_FLOAT}, {}, {2}};

  ExceptionDumper exception_dumper;
  exception_dumper.op_desc_info_ = { op_desc_info };
  exception_dumper.DumpExceptionInfo(exception_infos);
}
}  // namespace ge