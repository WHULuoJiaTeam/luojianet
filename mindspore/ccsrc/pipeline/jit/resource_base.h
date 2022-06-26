/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_JIT_RESOURCE_BASE_H_
#define MINDSPORE_CCSRC_PIPELINE_JIT_RESOURCE_BASE_H_

#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include "utils/hash_map.h"
#include "utils/any.h"
#include "ir/manager.h"

namespace mindspore {
namespace pipeline {
class ResourceBase {
 public:
  ResourceBase() { manager_ = MakeManager(); }

  virtual ~ResourceBase() = default;

  FuncGraphManagerPtr manager() { return manager_; }
  // set a manager defined outside which will not manage the graphs.
  void set_manager(const FuncGraphManagerPtr &manager) { manager_ = manager; }

  void SetResult(const std::string &key, const Any &value) { results_[key] = value; }

  Any GetResult(const std::string &key) const {
    auto iter = results_.find(key);
    if (iter == results_.end()) {
      MS_LOG(EXCEPTION) << "this key is not in resource list:" << key;
    }
    return iter->second;
  }

  bool HasResult(const std::string &key) const { return results_.count(key) != 0; }

 protected:
  FuncGraphManagerPtr manager_;
  mindspore::HashMap<std::string, Any> results_;
};

using ResourceBasePtr = std::shared_ptr<pipeline::ResourceBase>;
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_JIT_RESOURCE_BASE_H_
