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

#ifndef SUPER_KERNEL_FACTORY_H
#define SUPER_KERNEL_FACTORY_H

#include <vector>
#include "graph/load/model_manager/task_info/super_kernel/super_kernel.h"
#include "framework/common/debug/log.h"

namespace ge {
namespace skt {
class SuperKernelFactory {
 private:
  void *func_stub_ = nullptr;
  void *func_ptr_ = nullptr;
  void *handle_ = nullptr;
  std::string sk_stub_name_ = "super_kernel_template";
  bool is_init_ = false;
  SuperKernelFactory() {};
  ~SuperKernelFactory() {
    if (handle_ != nullptr) {
      GELOGI("SKT: SKT LIB PATH release.");
      if (mmDlclose(handle_) != 0) {
        const char *error = mmDlerror();
        GE_IF_BOOL_EXEC(error == nullptr, error = "");
        GELOGW("failed to close handle, message: %s", error);
      }
    }
  };

 public:
  SuperKernelFactory(SuperKernelFactory const &) = delete;
  void operator=(SuperKernelFactory const &) = delete;
  static SuperKernelFactory &GetInstance();
  Status Init();
  Status Uninitialize();
  Status FuseKernels(const std::vector<void *> &stub_func_list, const std::vector<void *> &args_addr_list,
                     uint32_t block_dim, std::unique_ptr<skt::SuperKernel> &h);
};
}  // namespace skt
}  // namespace ge
#endif
