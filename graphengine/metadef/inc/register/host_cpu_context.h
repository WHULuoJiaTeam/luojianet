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

#ifndef INC_REGISTER_HOST_CPU_CONTEXT_H_
#define INC_REGISTER_HOST_CPU_CONTEXT_H_

#include "external/ge/ge_api_error_codes.h"
#include "register/register_types.h"

namespace ge {
class HostCpuContext {
 public:
  HostCpuContext() = default;
  ~HostCpuContext() = default;
 private:
  class Impl;
  Impl *impl_;
};
} // namespace ge

extern "C" {
// Unified definition for registering host_cpu_kernel_wrapper when so is opened
FMK_FUNC_HOST_VISIBILITY ge::Status Initialize(const ge::HostCpuContext &ctx);
}

#endif //INC_REGISTER_HOST_CPU_CONTEXT_H_
