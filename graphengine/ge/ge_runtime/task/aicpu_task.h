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

#ifndef GE_GE_RUNTIME_TASK_AICPU_TASK_H_
#define GE_GE_RUNTIME_TASK_AICPU_TASK_H_

#include <memory>
#include <string>
#include "ge_runtime/task/task.h"

namespace ge {
namespace model_runner {
class AicpuTask : public TaskRepeater<AicpuTaskInfo> {
 public:
  AicpuTask(const ModelContext &model_context, const std::shared_ptr<AicpuTaskInfo> &task_info);

  ~AicpuTask() override;

  bool Distribute() override;

  void *Args() override { return input_output_addr_; }

  std::string task_name() const override { return task_info_->op_name(); }

 private:
  static void ReleaseRtMem(void **ptr) noexcept;

  std::shared_ptr<AicpuTaskInfo> task_info_;
  void *stream_;
  void *args_;
  void *ext_info_;
  void *input_output_addr_;
};
}  // namespace model_runner
}  // namespace ge

#endif  // GE_GE_RUNTIME_TASK_AICPU_TASK_H_
