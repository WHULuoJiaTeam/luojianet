/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "common/graph_kernel/split_model/split_model_factory.h"
#include <memory>
#include "utils/ms_context.h"
#include "common/graph_kernel/split_model/split_model_cpu.h"

namespace mindspore::graphkernel::inner {
SplitModelPtr SplitModelFactory::CreateSplitModel(const std::string &processor) {
  if (processor == kCPUDevice) {
    return std::make_shared<SplitModelCpu>();
  }
  return nullptr;
}
}  // namespace mindspore::graphkernel::inner
