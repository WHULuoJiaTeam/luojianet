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

#include "src/runtime/kernel/ascend/src/custom_interface.h"
#include <memory>
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"
#include "common/log_adapter.h"

namespace luojianet_ms::kernel {
namespace acl {
Status CustomInterface::Infer(std::vector<luojianet_ms::MSTensor> *inputs, std::vector<luojianet_ms::MSTensor> *outputs,
                              const luojianet_ms::schema::Primitive *primitive) {
  if (inputs == nullptr || (*inputs).empty()) {
    MS_LOG(ERROR) << "Inputs is invalid.";
    return kLiteError;
  }
  if (outputs == nullptr || (*outputs).empty()) {
    MS_LOG(ERROR) << "Outputs is invalid.";
    return kLiteError;
  }
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "Primitive is nullptr.";
    return kLiteError;
  }
  if (primitive->value_type() != schema::PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type is not PrimitiveType_Custom.";
    return kLiteError;
  }
  return kSuccess;
}

std::shared_ptr<luojianet_ms::kernel::KernelInterface> CustomInferCreater() {
  auto infer = std::make_shared<CustomInterface>();
  if (infer == nullptr) {
    MS_LOG(ERROR) << "New custom infer is nullptr";
    return nullptr;
  }
  return infer;
}
}  // namespace acl
}  // namespace luojianet_ms::kernel
namespace luojianet_ms {
namespace kernel {
REGISTER_CUSTOM_KERNEL_INTERFACE(ACL, ACL, acl::CustomInferCreater);
}  // namespace kernel
}  // namespace luojianet_ms
