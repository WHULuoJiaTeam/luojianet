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

#ifndef INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H_
#define INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H_

#include <memory>
#include "register/register_types.h"
#include "common/opskernel/ops_kernel_builder.h"

namespace ge {
using OpsKernelBuilderPtr = std::shared_ptr<OpsKernelBuilder>;

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpsKernelBuilderRegistry {
 public:
  ~OpsKernelBuilderRegistry();
  static OpsKernelBuilderRegistry &GetInstance();

  void Register(const std::string &lib_name, const OpsKernelBuilderPtr &instance);

  void Unregister(const std::string &lib_name);

  void UnregisterAll();

  const std::map<std::string, OpsKernelBuilderPtr> &GetAll() const;

 private:
  OpsKernelBuilderRegistry() = default;
  std::map<std::string, OpsKernelBuilderPtr> kernel_builders_;
};

class FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY OpsKernelBuilderRegistrar {
 public:
  using CreateFn = OpsKernelBuilder *(*)();
  OpsKernelBuilderRegistrar(const std::string &kernel_lib_name, CreateFn const fn);
  ~OpsKernelBuilderRegistrar();

private:
  std::string kernel_lib_name_;
};

#define REGISTER_OPS_KERNEL_BUILDER(kernel_lib_name, builder) \
    REGISTER_OPS_KERNEL_BUILDER_UNIQ_HELPER(__COUNTER__, kernel_lib_name, builder)

#define REGISTER_OPS_KERNEL_BUILDER_UNIQ_HELPER(ctr, kernel_lib_name, builder) \
    REGISTER_OPS_KERNEL_BUILDER_UNIQ(ctr, kernel_lib_name, builder)

#define REGISTER_OPS_KERNEL_BUILDER_UNIQ(ctr, kernel_lib_name, builder)                         \
  static ::ge::OpsKernelBuilderRegistrar register_op_kernel_builder_##ctr                       \
      __attribute__((unused)) =                                                                 \
          ::ge::OpsKernelBuilderRegistrar(kernel_lib_name, []()->::ge::OpsKernelBuilder* {      \
            return new (std::nothrow) builder();                                                \
          })
}  // namespace ge

#endif // INC_REGISTER_OPS_KERNEL_BUILDER_REGISTRY_H_
