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
#ifndef H99C11FC4_700E_4D4D_B073_7808FA88BEBC
#define H99C11FC4_700E_4D4D_B073_7808FA88BEBC

#include "ge_running_env/fake_engine.h"
#include "fake_ns.h"
#include "opskernel_manager/ops_kernel_manager.h"
#include "register/ops_kernel_builder_registry.h"

FAKE_NS_BEGIN

struct GeRunningEnvFaker {
  GeRunningEnvFaker();
  GeRunningEnvFaker &Reset();
  GeRunningEnvFaker &Install(const EnvInstaller &);
  GeRunningEnvFaker &InstallDefault();
  static void BackupEnv();

 private:
  void flush();

 private:
  std::map<string, vector<OpInfo>> &op_kernel_info_;
  std::map<string, OpsKernelInfoStorePtr> &ops_kernel_info_stores_;
  std::map<string, GraphOptimizerPtr> &ops_kernel_optimizers_;
  std::map<string, OpsKernelBuilderPtr> &ops_kernel_builders_;
};

FAKE_NS_END

#endif /* H99C11FC4_700E_4D4D_B073_7808FA88BEBC */
