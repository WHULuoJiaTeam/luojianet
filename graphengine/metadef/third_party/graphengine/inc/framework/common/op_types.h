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

#ifndef INC_FRAMEWORK_COMMON_OP_TYPES_H_
#define INC_FRAMEWORK_COMMON_OP_TYPES_H_

#include <set>
#include <string>

namespace ge {
class GE_FUNC_VISIBILITY OpTypeContainer {
 public:
  static OpTypeContainer *Instance() {
    static OpTypeContainer instance;
    return &instance;
  }
  ~OpTypeContainer() = default;

  void Register(const std::string &op_type) { op_type_list_.insert(op_type); }

  bool IsExisting(const std::string &op_type) {
    auto iter_find = op_type_list_.find(op_type);
    return iter_find != op_type_list_.end();
  }

 protected:
  OpTypeContainer() {}

 private:
  std::set<std::string> op_type_list_;
};

class GE_FUNC_VISIBILITY OpTypeRegistrar {
 public:
  explicit OpTypeRegistrar(const std::string &op_type) { OpTypeContainer::Instance()->Register(op_type); }
  ~OpTypeRegistrar() {}
};

#define REGISTER_OPTYPE_DECLARE(var_name, str_name) \
  FMK_FUNC_HOST_VISIBILITY FMK_FUNC_DEV_VISIBILITY extern const char *var_name;

#define REGISTER_OPTYPE_DEFINE(var_name, str_name)           \
  const char *var_name = str_name;                           \
  const OpTypeRegistrar g_##var_name##_reg(str_name);

#define IS_OPTYPE_EXISTING(str_name) (OpTypeContainer::Instance()->IsExisting(str_name))
}  // namespace ge

#endif  // INC_FRAMEWORK_COMMON_OP_TYPES_H_
