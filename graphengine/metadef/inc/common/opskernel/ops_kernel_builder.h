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

#ifndef INC_COMMON_OPSKERNELUTILS_OPS_KERNEL_INFO_UTILS_H_
#define INC_COMMON_OPSKERNELUTILS_OPS_KERNEL_INFO_UTILS_H_

#include "external/ge/ge_api_error_codes.h"
#include "cce/aicpu_engine_struct.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph/node.h"
#include "proto/task.pb.h"

namespace ge {
class OpsKernelBuilder {
 public:
  enum class Mode {
    kNormal,
    kFfts,
    kFftsPlus
  };
  OpsKernelBuilder() = default;
  virtual ~OpsKernelBuilder() = default;

  // initialize OpsKernelBuilder
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  // finalize OpsKernelBuilder
  virtual Status Finalize() = 0;

  // memory allocation requirement
  virtual Status CalcOpRunningParam(Node &node) = 0;

  // generate task for op
  virtual Status GenerateTask(const Node &node, RunContext &context,
                              std::vector<domi::TaskDef> &tasks) = 0;

  // generate task for op with different mode
  virtual Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks,
                              OpsKernelBuilder::Mode) {
    return SUCCESS;
  }

  // only call aicpu interface to generate task struct
  virtual Status GenSingleOpRunTask(const NodePtr &node, STR_FWK_OP_KERNEL &task, std::string &task_info) {
    return FAILED;
  }

  // only call aicpu interface to generate task struct
  virtual Status GenMemCopyTask(uint64_t count, STR_FWK_OP_KERNEL &task, std::string &task_info) {
    return FAILED;
  }
};
}  // namespace ge
#endif // INC_COMMON_OPSKERNELUTILS_OPS_KERNEL_INFO_UTILS_H_
