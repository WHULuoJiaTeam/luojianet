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
#ifndef H737AD661_27C0_400F_8B08_29701308C5D0
#define H737AD661_27C0_400F_8B08_29701308C5D0

#include <string>
#include <set>
#include "fake_ns.h"
#include "ge_running_env/env_installer.h"
#include "graph/operator_factory.h"

FAKE_NS_BEGIN

struct FakeOp : EnvInstaller {
  FakeOp(const std::string& op_type);

  FakeOp& Inputs(const std::vector<std::string>&);
  FakeOp& Outputs(const std::vector<std::string>&);
  FakeOp& InferShape(InferShapeFunc);
  FakeOp& InfoStoreAndBuilder(const std::string&);

 private:
  void Install() const override;
  void InstallTo(std::map<string, OpsKernelInfoStorePtr>&) const override;

 private:
  const std::string op_type_;
  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;
  InferShapeFunc info_fun_;
  std::set<std::string> info_store_names_;
};

FAKE_NS_END

#endif /* H737AD661_27C0_400F_8B08_29701308C5D0 */
