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

#ifndef GE_COMMON_DUMP_EXCEPTION_DUMPER_H_
#define GE_COMMON_DUMP_EXCEPTION_DUMPER_H_

#include <vector>

#include "graph/op_desc.h"
#include "framework/common/ge_types.h"
#include "graph/load/model_manager/task_info/task_info.h"

namespace ge {
class ExceptionDumper {
 public:
  ExceptionDumper() = default;
  ~ExceptionDumper();

  void SaveDumpOpInfo(const OpDescPtr &op, uint32_t task_id, uint32_t stream_id,
                      std::vector<void *> &input_addrs, std::vector<void *> &output_addrs);
  void SaveDumpOpInfo(const RuntimeParam &model_param, const OpDescPtr &op, uint32_t task_id, uint32_t stream_id);
  Status DumpExceptionInfo(const std::vector<rtExceptionInfo> &exception_infos) const;
  bool GetOpDescInfo(uint32_t stream_id, uint32_t task_id, OpDescInfo &op_desc_info) const;
  OpDescInfo *MutableOpDescInfo(uint32_t task_id, uint32_t stream_id);

 private:
  void SaveOpDescInfo(const OpDescPtr &op, uint32_t task_id, uint32_t stream_id, OpDescInfo &op_desc_info);
  Status DumpExceptionInput(const OpDescInfo &op_desc_info, const std::string &dump_file) const;
  Status DumpExceptionOutput(const OpDescInfo &op_desc_info, const std::string &dump_file) const;

  std::vector<OpDescInfo> op_desc_info_;
};
}  // namespace ge

#endif // GE_COMMON_DUMP_EXCEPTION_DUMPER_H_
