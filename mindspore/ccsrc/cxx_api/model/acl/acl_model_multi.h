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
#ifndef MINDSPORE_CCSRC_CXX_API_ACL_MODEL_MULTI_H
#define MINDSPORE_CCSRC_CXX_API_ACL_MODEL_MULTI_H

#include "cxx_api/model/acl/acl_model.h"
#include <memory>
#include <optional>
#include <vector>
#include <string>
#include <map>

namespace mindspore {
namespace compile {
class MsBackend;
class FinalVM;
}  // namespace compile

class AclModelMulti : public AclModel {
 public:
  AclModelMulti() : AclModel(), is_multi_graph_(std::nullopt) {}
  ~AclModelMulti() = default;

  Status Build() override;
  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) override;

  std::vector<MSTensor> GetInputs() override;
  std::vector<MSTensor> GetOutputs() override;

 private:
  void SetInputs();
  void SetOutput();

  std::optional<bool> is_multi_graph_;
  std::shared_ptr<compile::MsBackend> backend_;
  std::shared_ptr<compile::FinalVM> vm_;
  std::vector<MSTensor> inputs_ = {};
  std::vector<MSTensor> outputs_ = {};
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_CXX_API_ACL_MODEL_MULTI_H
