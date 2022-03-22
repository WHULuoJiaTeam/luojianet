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

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_UTILS_GE_LOCAL_OPS_KERNEL_UTILS_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_UTILS_GE_LOCAL_OPS_KERNEL_UTILS_H_

#include "external/ge/ge_api_error_codes.h"
#include "common/opskernel/ops_kernel_builder.h"

namespace ge {
namespace ge_local {
class GE_FUNC_VISIBILITY GeLocalOpsKernelBuilder : public OpsKernelBuilder {
 public:
  ~GeLocalOpsKernelBuilder() override;
  Status Initialize(const map<std::string, std::string> &options) override;

  Status Finalize() override;

  Status CalcOpRunningParam(Node &node) override;

  Status GenerateTask(const Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) override;

 private:
  /**
   * Calc memSize for constant which type is DT_STRING.
   * @param op_desc OpDesc information
   * @param mem_size output size
   * @return whether this operation success
   */
  Status CalcConstantStrMemSize(const OpDescPtr &op_desc, int64_t &mem_size);
};
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_UTILS_GE_LOCAL_OPS_KERNEL_UTILS_H_
