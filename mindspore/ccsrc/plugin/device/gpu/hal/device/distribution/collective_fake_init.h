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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_FAKE_INIT_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_FAKE_INIT_H_

#include <string>
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
namespace gpu {
class BACKEND_EXPORT CollectiveFakeInitializer {
 public:
  CollectiveFakeInitializer() = default;
  ~CollectiveFakeInitializer() = default;
  CollectiveFakeInitializer(CollectiveFakeInitializer const &) = delete;
  CollectiveFakeInitializer &operator=(const CollectiveFakeInitializer &) = delete;
  static void InitCollective();
  static void FinalizeCollective();
  static uint32_t GetRankID(const std::string &group_name);
  static uint32_t GetRankSize(const std::string &group_name);
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_DISTRIBUTION_COLLECTIVE_FAKE_INIT_H_
