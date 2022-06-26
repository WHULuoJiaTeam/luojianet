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
#ifndef MINDSPORE_LITE_SRC_OPS_SCHEMA_REGISTER_H_
#define MINDSPORE_LITE_SRC_OPS_SCHEMA_REGISTER_H_
#include <string>
#include <vector>
#include <functional>

namespace mindspore::lite::ops {
using GetSchemaDef = std::function<std::string()>;

class SchemaRegisterImpl {
 public:
  static SchemaRegisterImpl *Instance() {
    static SchemaRegisterImpl instance;
    return &instance;
  }

  void OpPush(GetSchemaDef func) { op_def_funcs_.push_back(func); }

  const std::vector<GetSchemaDef> &GetAllOpDefCreateFuncs() const { return op_def_funcs_; }

  void SetPrimTypeGenFunc(GetSchemaDef func) { prim_type_gen_ = func; }

  GetSchemaDef GetPrimTypeGenFunc() const { return prim_type_gen_; }

  virtual ~SchemaRegisterImpl() = default;

 private:
  std::vector<GetSchemaDef> op_def_funcs_;
  GetSchemaDef prim_type_gen_{nullptr};
};

class SchemaOpRegister {
 public:
  explicit SchemaOpRegister(GetSchemaDef func) { SchemaRegisterImpl::Instance()->OpPush(func); }
  virtual ~SchemaOpRegister() = default;
};

class PrimitiveTypeRegister {
 public:
  explicit PrimitiveTypeRegister(GetSchemaDef func) { SchemaRegisterImpl::Instance()->SetPrimTypeGenFunc(func); }
  virtual ~PrimitiveTypeRegister() = default;
};
}  // namespace mindspore::lite::ops

#endif  // MINDSPORE_LITE_SRC_OPS_SCHEMA_REGISTER_H_
