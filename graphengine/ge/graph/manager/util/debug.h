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

#ifndef GE_GRAPH_MANAGER_UTIL_DEBUG_H_
#define GE_GRAPH_MANAGER_UTIL_DEBUG_H_

#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "framework/common/debug/log.h"
#include "common/debug/memory_dumper.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "mmpa/mmpa_api.h"
#include "proto/om.pb.h"
#include "runtime/kernel.h"

using google::protobuf::Message;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileOutputStream;

namespace ge {
// MemoryDumper: used for output memory data in test case
class Debug {
 public:
  Debug();
  ~Debug();

  static void DumpProto(const Message &proto, const char *file);
  static Status DumpDevMem(const char *file, const void *addr, int64_t size);
};
}  // namespace ge

#endif  // GE_GRAPH_MANAGER_UTIL_DEBUG_H_
