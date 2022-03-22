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

#ifndef GE_SINGLE_OP_TASK_AICPU_TASK_BUILDER_H_
#define GE_SINGLE_OP_TASK_AICPU_TASK_BUILDER_H_

#include <vector>
#include "graph/op_desc.h"
#include "single_op/single_op.h"
#include "single_op/single_op_model.h"
#include "cce/aicpu_engine_struct.h"

namespace ge {
  class AiCpuTaskBuilder {
  public:
    AiCpuTaskBuilder(const OpDescPtr &op_desc, const domi::KernelExDef &kernel_def);
    ~AiCpuTaskBuilder() = default;

    Status BuildTask(AiCpuTask &task, const SingleOpModelParam &param, uint64_t kernel_id);

  private:
    static Status SetKernelArgs(void **args, STR_FWK_OP_KERNEL &kernel);
    Status SetFmkOpKernel(void *io_addr, void *ws_addr, STR_FWK_OP_KERNEL &kernel);
    Status InitWorkspaceAndIO(AiCpuTask &task, const SingleOpModelParam &param);

    const OpDescPtr op_desc_;
    const domi::KernelExDef &kernel_def_;
  };
}  // namespace ge

#endif  // GE_SINGLE_OP_TASK_AICPU_TASK_BUILDER_H_