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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/unstack_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateUnstackParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto unstack_prim = primitive->value_as_Unstack();
  if (unstack_prim == nullptr) {
    MS_LOG(ERROR) << "unstack_prim is nullptr";
    return nullptr;
  }
  auto *unstack_param = reinterpret_cast<UnstackParameter *>(malloc(sizeof(UnstackParameter)));
  if (unstack_param == nullptr) {
    MS_LOG(ERROR) << "malloc UnstackParameter failed.";
    return nullptr;
  }
  memset(unstack_param, 0, sizeof(UnstackParameter));

  unstack_param->op_parameter_.type_ = schema::PrimitiveType_Unstack;
  unstack_param->axis_ = unstack_prim->axis();
  return reinterpret_cast<OpParameter *>(unstack_param);
}
}  // namespace

Registry g_unstackV0ParameterRegistry(schema::v0::PrimitiveType_Unstack, PopulateUnstackParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
