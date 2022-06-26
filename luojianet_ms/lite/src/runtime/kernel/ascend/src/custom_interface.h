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
#ifndef LUOJIANET_MS_LITE_ACL_CUSTOM_INTERFACE_H_
#define LUOJIANET_MS_LITE_ACL_CUSTOM_INTERFACE_H_

#include <vector>
#include <string>
#include "include/kernel_interface.h"

namespace luojianet_ms::kernel {
namespace acl {
class CustomInterface : public luojianet_ms::kernel::KernelInterface {
 public:
  CustomInterface() {}
  ~CustomInterface() = default;

  Status Infer(std::vector<luojianet_ms::MSTensor> *inputs, std::vector<luojianet_ms::MSTensor> *outputs,
               const luojianet_ms::schema::Primitive *primitive) override;

 private:
  Status GetCustomAttr(char *buf, uint32_t buf_size, const luojianet_ms::schema::Custom *op, const std::string &attr_name);
};
}  // namespace acl
}  // namespace luojianet_ms::kernel
#endif  // LUOJIANET_MS_LITE_ACL_CUSTOM_INTERFACE_H_
