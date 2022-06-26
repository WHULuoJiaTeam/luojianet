/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/control_flow/exit_subgraph_kernel.h"
#include "src/tensor.h"

namespace mindspore::kernel {
int ExitSubGraphKernel::Execute(const KernelCallBack &before, const KernelCallBack &after) { return lite::RET_OK; }

SubGraphKernel *ExitSubGraphKernel::Create(Kernel *kernel) {
  auto sub_kernel = new kernel::ExitSubGraphKernel(kernel);
  if (sub_kernel == nullptr) {
    MS_LOG(ERROR) << "create entrance subgraph failed!";
    return nullptr;
  }
  return sub_kernel;
}

void ExitSubGraphKernel::SetPartial(kernel::LiteKernel *partial_node) { partials_.insert(partial_node); }
}  // namespace mindspore::kernel
