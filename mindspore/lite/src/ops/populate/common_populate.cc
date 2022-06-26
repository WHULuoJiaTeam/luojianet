/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
using mindspore::schema::PrimitiveType_AddN;
using mindspore::schema::PrimitiveType_Attention;
using mindspore::schema::PrimitiveType_Depend;
using mindspore::schema::PrimitiveType_SwitchLayer;
using mindspore::schema::PrimitiveType_ZerosLike;

namespace mindspore {
namespace lite {
OpParameter *PopulateCommonParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);

  auto *param = reinterpret_cast<OpParameter *>(malloc(sizeof(OpParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc OpParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(OpParameter));

  param->type_ = primitive->value_type();
  return param;
}
REG_POPULATE(PrimitiveType_AddN, PopulateCommonParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_ZerosLike, PopulateCommonParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_Depend, PopulateCommonParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_Attention, PopulateCommonParameter, SCHEMA_CUR)
REG_POPULATE(PrimitiveType_SwitchLayer, PopulateCommonParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
