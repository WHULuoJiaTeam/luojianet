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

#include "graph/manager/util/debug.h"

#include <string>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"

using google::protobuf::Message;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileOutputStream;

namespace ge {
Debug::Debug() = default;

Debug::~Debug() = default;

void Debug::DumpProto(const Message &proto, const char *file) {
  std::string file_path = RealPath(file);
  int fd = mmOpen2(file_path.c_str(), M_WRONLY | M_CREAT | O_TRUNC, M_IRUSR | M_IWUSR | M_UMASK_GRPREAD |
                   M_UMASK_OTHREAD);
  if (fd == -1) {
    GELOGW("Write %s failed. errmsg:%s", file_path.c_str(), strerror(errno));
    return;
  }
  auto output = ge::MakeShared<FileOutputStream>(fd);
  if (output == nullptr) {
    GELOGW("create output failed.");
    if (mmClose(fd) != 0) {
      GELOGW("close fd failed. errmsg:%s", strerror(errno));
    }
    return;
  }
  bool ret = google::protobuf::TextFormat::Print(proto, output.get());
  if (!ret) {
    GELOGW("dump proto failed.");
  }
  if (mmClose(fd) != 0) {
    GELOGW("close fd failed. errmsg:%s", strerror(errno));
  }
}

Status Debug::DumpDevMem(const char *file, const void *addr, int64_t size) {
  if (size == 0) {
    GELOGI("Dump data failed because the size is 0.");
    return SUCCESS;
  }
  uint8_t *host_addr = nullptr;
  rtError_t ret = rtMallocHost(reinterpret_cast<void **>(&host_addr), size);
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMallocHost failed, size:%zu, ret:0x%X", size, ret);
    GELOGE(FAILED, "[Call][RtMallocHost] failed, size:%zu, ret:0x%X", size, ret);
    return FAILED;
  }
  GE_MAKE_GUARD_RTMEM(host_addr);
  ret = rtMemcpy(host_addr, size, addr, size, RT_MEMCPY_DEVICE_TO_HOST);
  if (ret != RT_ERROR_NONE) {
    REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, size:%zu, ret:0x%X", size, ret);
    GELOGE(FAILED, "[Call][RtMemcpy] failed, size:%zu, ret:0x%X", size, ret);
    return FAILED;
  }

  GE_CHK_STATUS_RET(MemoryDumper::DumpToFile(file, host_addr, size));
  return SUCCESS;
}
}  // namespace ge
