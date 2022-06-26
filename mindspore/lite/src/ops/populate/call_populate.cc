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
#include "src/ops/populate/populate_register.h"
#include "nnacl/call_parameter.h"
using mindspore::schema::PrimitiveType_Call;

namespace mindspore {
namespace lite {
OpParameter *PopulateCallParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Call();
  if (value == nullptr) {
    MS_LOG(ERROR) << "call param is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<CallParameter *>(malloc(sizeof(CallParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc CallParameter failed.";
    return nullptr;
  }
  memset(reinterpret_cast<void *>(param), 0, sizeof(CallParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->is_tail_call = value->is_tail_call();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Call, PopulateCallParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
