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

#include "plugin/device/ascend/hal/device/executor/rts/memcpy_rts_dynamic_kernel.h"

#include "runtime/mem.h"
#include "acl/acl_rt.h"

namespace luojianet_ms {
namespace device {
namespace ascend {
MemcpyRtsDynamicKernel::~MemcpyRtsDynamicKernel() {
  dst_ = nullptr;
  src_ = nullptr;
}

void MemcpyRtsDynamicKernel::Execute() {
  auto status = aclrtMemcpyAsync(dst_, dest_max_, src_, count_, ACL_MEMCPY_DEVICE_TO_DEVICE, stream_);
  if (status != RT_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "MemCpyAsync op execute aclrtMemcpyAsync failed!";
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace luojianet_ms
