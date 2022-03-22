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

#ifndef GE_COMMON_DEBUG_MEMORY_DUMPER_H_
#define GE_COMMON_DEBUG_MEMORY_DUMPER_H_

#include <stdint.h>

#include "framework/common/types.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/ge_inner_error_codes.h"

namespace ge {
// MemoryDumper：dump memory data for internal test
// Output in one time: using DumpToFile
// Open file at one time and output multiple times: create  MemoryDumper object first, and using Open/Dump/Close
class MemoryDumper {
 public:
  MemoryDumper();
  ~MemoryDumper();

  // Assignment/copy is not allowed to avoid repeated release
  MemoryDumper &operator=(const MemoryDumper &dumper) = delete;
  MemoryDumper(const MemoryDumper &dumper) = delete;

  /** @ingroup domi_common
   *  @brief write memory data to file, if the filename is not exist, create it first
   *  @param [in] filename  the output file path, specific to filename
   *  @param [in] data the memory data
   *  @param [in] len length of data
   *  @return SUCCESS  output success
   *  @return FAILED   output failed
   *  @author
   */
  static Status DumpToFile(const char *filename, void *data, int64_t len);

  /** @ingroup domi_common
   *  @brief open the dump file
   *  @param [in] filename the output file path, specific to filename
   *  @return SUCCESS  open file success
   *  @return FAILED   open file failed
   *  @author
   */
  Status Open(const char *filename);

  /** @ingroup domi_common
   *  @brief write the Memory data to file
   *  @param [in] data the memory data
   *  @param [in] len length of data
   *  @return SUCCESS  success
   *  @return FAILED   failed
   *  @author
   */
  Status Dump(void *data, uint32_t len) const;

  /** @ingroup domi_common
   *  @brief close the Dump file
   *  @return SUCCESS  success
   *  @return FAILED   failed
   *  @author
   */
  void Close() noexcept;

 private:
  /** @ingroup domi_common
   *  @brief open the dump file
   *  @param [in] filename the output file path, specific to filename
   *  @return int the file handle after file open, -1 means open file failed
   *  @author
   */
  static int OpenFile(const char *filename);

  int fd_;
};
}  // namespace ge

#endif  // GE_COMMON_DEBUG_MEMORY_DUMPER_H_
