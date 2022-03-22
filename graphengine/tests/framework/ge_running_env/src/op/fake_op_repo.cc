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

#include "graph/operator_factory_impl.h"
#include "ge_running_env/fake_op.h"
#include "fake_op_repo.h"

FAKE_NS_BEGIN

void FakeOpRepo::Reset() {
  if (OperatorFactoryImpl::operator_creators_) {
    OperatorFactoryImpl::operator_creators_->clear();
  }
  if (OperatorFactoryImpl::operator_infershape_funcs_) {
    OperatorFactoryImpl::operator_infershape_funcs_->clear();
  }
}

void FakeOpRepo::Regist(const std::string &operator_type, const OpCreator creator) {
  OperatorFactoryImpl::RegisterOperatorCreator(operator_type, creator);
}
void FakeOpRepo::Regist(const std::string &operator_type, const InferShapeFunc infer_fun) {
  OperatorFactoryImpl::RegisterInferShapeFunc(operator_type, infer_fun);
}

FAKE_NS_END