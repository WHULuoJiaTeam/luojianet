/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_STREAM_SWITCH_TASK_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_STREAM_SWITCH_TASK_H_

#include <memory>
#include <vector>
#include "plugin/device/ascend/hal/device/ge_runtime/task/task.h"

namespace mindspore::ge::model_runner {
class StreamSwitchTask : public TaskRepeater<StreamSwitchTaskInfo> {
 public:
  StreamSwitchTask(const ModelContext &model_context, const std::shared_ptr<StreamSwitchTaskInfo> &task_info);

  ~StreamSwitchTask() override;

  void Distribute() override;

 private:
  std::shared_ptr<StreamSwitchTaskInfo> task_info_;

  void *stream_;
  std::vector<rtStream_t> stream_list_;
};
}  // namespace mindspore::ge::model_runner
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_GE_RUNTIME_TASK_STREAM_SWITCH_TASK_H_
