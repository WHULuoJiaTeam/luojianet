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

#ifndef H1D9F4FDE_BB21_4DE4_AC7E_751920B45039
#define H1D9F4FDE_BB21_4DE4_AC7E_751920B45039

#include "fake_ns.h"
#include "opskernel_manager/ops_kernel_manager.h"
#include "register/ops_kernel_builder_registry.h"

FAKE_NS_BEGIN

struct EnvInstaller {
  virtual void InstallTo(std::map<string, OpsKernelInfoStorePtr>&) const {}
  virtual void InstallTo(std::map<string, GraphOptimizerPtr>&) const {}
  virtual void InstallTo(std::map<string, OpsKernelBuilderPtr>&) const {}
  virtual void Install() const {}
};

FAKE_NS_END

#endif
