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

#include "ge_running_env/fake_op.h"
#include "fake_op_repo.h"
#include "ge_running_env/info_store_holder.h"
#include "graph/operator_factory.h"

FAKE_NS_BEGIN

FakeOp::FakeOp(const std::string& op_type) : op_type_(op_type) {}

FakeOp& FakeOp::Inputs(const std::vector<std::string>& inputs) {
  inputs_ = inputs;
  return *this;
}

FakeOp& FakeOp::Outputs(const std::vector<std::string>& outputs) {
  outputs_ = outputs;
  return *this;
}

FakeOp& FakeOp::InferShape(InferShapeFunc infer_fun) {
  info_fun_ = infer_fun;
  return *this;
}

FakeOp& FakeOp::InfoStoreAndBuilder(const std::string& name) {
  info_store_names_.insert(name);
  return *this;
}

namespace {

void RegistOpToInfoStore(OpsKernelInfoStorePtr& info_store, const std::string& op_type) {
  if (info_store == nullptr) {
    return;
  }
  auto holder = dynamic_cast<InfoStoreHolder*>(info_store.get());
  holder->RegistOp(op_type);
}

struct FakeOperator : Operator {
  FakeOperator(const std::string& op_type) : Operator(op_type) {}

  FakeOperator& RegistInputs(const std::vector<std::string>& inputs) {
    for (auto& input : inputs) {
      Operator::InputRegister(input);
    }
    return *this;
  }

  FakeOperator& RegistOutputs(const std::vector<std::string>& outputs) {
    for (auto& output : outputs) {
      Operator::OutputRegister(output);
    }
    return *this;
  }
};
}  // namespace

void FakeOp::InstallTo(std::map<string, OpsKernelInfoStorePtr>& info_stores) const {
  std::for_each(info_store_names_.begin(), info_store_names_.end(), [=, &info_stores](auto& info_store_name) {
    auto iter = info_stores.find(info_store_name);
    if (iter != info_stores.end()) {
      RegistOpToInfoStore(iter->second, op_type_);
    }
  });
}

void FakeOp::Install() const {
  FakeOpRepo::Regist(
    op_type_,
    [op_type = this->op_type_, inputs = this->inputs_, outputs = this->outputs_](const std::string&) -> Operator {
      return FakeOperator(op_type).RegistInputs(inputs).RegistOutputs(outputs);
    });
  if (info_fun_) {
    FakeOpRepo::Regist(op_type_, info_fun_);
  }
}

FAKE_NS_END
