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

#ifndef LUOJIANET_MS_CCSRC_PLUGIN_FACTORY_MS_FACTORY_H_
#define LUOJIANET_MS_CCSRC_PLUGIN_FACTORY_MS_FACTORY_H_

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ir/dtype/type_id.h"

namespace luojianet_ms {
namespace kernel {
template <class C>
class Factory {
  using CreatorFunc = std::function<std::shared_ptr<C>()>;

 public:
  Factory(const Factory &) = delete;
  void operator=(const Factory &) = delete;

  static Factory &Instance() {
    static Factory instance;
    return instance;
  }

  void Register(const std::string &name, CreatorFunc &&creator) {
    if (IsRegistered(name)) {
      MS_LOG(EXCEPTION) << "Kernel " << name << " is already registered!";
    }
    (void)kernel_mod_creators_.emplace(name, creator);
  }

  std::shared_ptr<C> Create(const std::string &name) {
    auto iter = kernel_mod_creators_.find(name);
    if (iter != kernel_mod_creators_.end()) {
      return (iter->second)();
    }

    MS_LOG(WARNING) << "Unsupported kernel type " << name;
    return nullptr;
  }

  bool IsRegistered(const std::string &name) {
    if (kernel_mod_creators_.find(name) != kernel_mod_creators_.end()) {
      return true;
    }
    return false;
  }

 private:
  Factory() = default;
  ~Factory() = default;
  std::map<std::string, CreatorFunc> kernel_mod_creators_;
};

template <class C>
class KernelRegistrar {
 public:
  explicit KernelRegistrar(const std::string &name, std::function<std::shared_ptr<C>()> creator) {
    Factory<C>::Instance().Register(name, std::move(creator));
  }
  ~KernelRegistrar() = default;
};

// Helper macro for factory registration.
#define MS_KERNEL_FACTORY_REG(BASE_CLASS, NAME, DERIVE_CLASS)                                                          \
  static_assert(std::is_base_of<BASE_CLASS, DERIVE_CLASS>::value, #DERIVE_CLASS " must be derived from " #BASE_CLASS); \
  static const KernelRegistrar<BASE_CLASS> g_##NAME##_##BASE_CLASS##_reg(                                              \
    #NAME, []() { return std::make_shared<DERIVE_CLASS>(); })

#define MS_KERNEL_FACTORY_REG_BY_CREATOR(BASE_CLASS, NAME, CREATOR) \
  static const KernelRegistrar<BASE_CLASS> g_##NAME##_##BASE_CLASS##_reg(#NAME, CREATOR)
}  // namespace kernel
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_PLUGIN_FACTORY_MS_FACTORY_H_
