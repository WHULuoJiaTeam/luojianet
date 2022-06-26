/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/hal/device/distribution/collective_fake_init.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace gpu {
void CollectiveFakeInitializer::InitCollective() {
  MS_LOG(EXCEPTION) << "You are trying to call 'init('nccl')', Please check "
                       "this MindSpore package is GPU version and built with NCCL.";
}

void CollectiveFakeInitializer::FinalizeCollective() {
  MS_LOG(EXCEPTION) << "You are trying to call 'init('nccl')', Please check "
                       "this MindSpore package is GPU version and built with NCCL.";
}

uint32_t CollectiveFakeInitializer::GetRankID(const std::string &) {
  MS_LOG(EXCEPTION) << "You are trying to call 'GetRankID', Please check "
                       "this MindSpore package is GPU version and built with NCCL.";
}

uint32_t CollectiveFakeInitializer::GetRankSize(const std::string &) {
  MS_LOG(EXCEPTION) << "You are trying to call 'GetRankSize', Please check "
                       "this MindSpore package is GPU version and built with NCCL.";
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
