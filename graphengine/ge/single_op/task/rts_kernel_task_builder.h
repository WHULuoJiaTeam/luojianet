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

#ifndef GE_SINGLE_OP_TASK_RTS_KERNEL_TASK_BUILDER_H_
#define GE_SINGLE_OP_TASK_RTS_KERNEL_TASK_BUILDER_H_

#include <vector>
#include "graph/op_desc.h"
#include "single_op/single_op.h"
#include "single_op/single_op_model.h"

namespace ge {
class RtsKernelTaskBuilder {
 public:
  static Status BuildMemcpyAsyncTask(const OpDescPtr &op_desc,
                                     const domi::MemcpyAsyncDef &kernel_def,
                                     const SingleOpModelParam &param,
                                     std::unique_ptr<MemcpyAsyncTask> &task);
};
}  // namespace ge
#endif  // GE_SINGLE_OP_TASK_RTS_KERNEL_TASK_BUILDER_H_