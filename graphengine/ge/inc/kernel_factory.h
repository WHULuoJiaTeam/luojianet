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

#ifndef GE_INC_KERNEL_FACTORY_H_
#define GE_INC_KERNEL_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>

#include "common/ge/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "external/graph/graph.h"

using std::string;

namespace ge {
class Kernel;

///
/// @ingroup domi_omg
/// @brief kernel create factory
/// @author
///
class KernelFactory {
 public:
  // KernelCreator（function）, type definition
  using KERNEL_CREATOR_FUN = std::function<std::shared_ptr<Kernel>(void)>;

  ///
  /// Get singleton instance
  ///
  static KernelFactory &Instance() {
    static KernelFactory instance;
    return instance;
  }

  ///
  /// create Kernel
  /// @param [in] op_type operation type
  ///
  std::shared_ptr<Kernel> Create(const std::string &op_type) {
    std::map<std::string, KERNEL_CREATOR_FUN>::iterator iter = creator_map_.find(op_type);
    if (iter != creator_map_.end()) {
      return iter->second();
    }

    return nullptr;
  }

  // Kernel registration function to register different types of kernel to the factory
  class Registerar {
   public:
    ///
    /// @ingroup domi_omg
    /// @brief Constructor
    /// @param [in] type operation type
    /// @param [in| fun kernel function of the operation
    ///
    Registerar(const string &type, const KERNEL_CREATOR_FUN &fun) {
      KernelFactory::Instance().RegisterCreator(type, fun);
    }
    ~Registerar() {}
  };

 protected:
  KernelFactory() {}
  ~KernelFactory() {}

  // register creator, this function will call in the constructor
  void RegisterCreator(const string &type, const KERNEL_CREATOR_FUN &fun) {
    std::map<std::string, KERNEL_CREATOR_FUN>::iterator iter = creator_map_.find(type);
    if (iter != creator_map_.end()) {
      GELOGD("KernelFactory::RegisterCreator: %s creator already exist", type.c_str());
      return;
    }

    creator_map_[type] = fun;
  }

 private:
  std::map<std::string, KERNEL_CREATOR_FUN> creator_map_;
};

#define REGISTER_KERNEL(type, clazz)                  \
  std::shared_ptr<Kernel> Creator_##type##_Kernel() { \
    std::shared_ptr<clazz> ptr = nullptr;             \
    ptr = MakeShared<clazz>();                        \
    return ptr;                                       \
  }                                                   \
  KernelFactory::Registerar g_##type##_Kernel_Creator(type, Creator_##type##_Kernel)
}  // namespace ge
#endif  // GE_INC_KERNEL_FACTORY_H_
