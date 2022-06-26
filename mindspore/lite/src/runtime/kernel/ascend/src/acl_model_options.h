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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_OPTIONS_H_
#define MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_OPTIONS_H_

#include <string>
#include <set>
#include <utility>

namespace mindspore::kernel {
namespace acl {
const uint64_t kBatchSizeInvalid = 0;

typedef struct AclModelOptions {
  int32_t device_id;
  std::string dump_cfg_path;
  std::set<uint64_t> batch_size;
  std::set<std::pair<uint64_t, uint64_t>> image_size;

  AclModelOptions() : device_id(0) {}
} AclModelOptions;
}  // namespace acl
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_AGENT_ACL_MODEL_OPTIONS_H_
