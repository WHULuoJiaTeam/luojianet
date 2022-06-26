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

#ifndef LUOJIANET_MS_LITE_SRC_RUNTIME_AGENT_ACL_OPTIONS_PARSERS_H_
#define LUOJIANET_MS_LITE_SRC_RUNTIME_AGENT_ACL_OPTIONS_PARSERS_H_

#include <memory>
#include <string>
#include "include/api/context.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/ascend/src/acl_model_options.h"

namespace luojianet_ms::kernel {
namespace acl {
using luojianet_ms::lite::STATUS;

class AclOptionsParser {
 public:
  STATUS ParseAclOptions(const luojianet_ms::Context *ctx, AclModelOptions *acl_options);

 private:
  STATUS ParseOptions(const std::shared_ptr<DeviceInfoContext> &device_info, AclModelOptions *acl_options);
  STATUS CheckDeviceId(int32_t *device_id);
};
}  // namespace acl
}  // namespace luojianet_ms::kernel

#endif  // LUOJIANET_MS_LITE_SRC_RUNTIME_AGENT_ACL_OPTIONS_PARSERS_H_
