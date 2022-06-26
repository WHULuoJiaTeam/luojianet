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
#include "src/ops/populate/v0/arithmetic_populate_v0.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateEltwiseParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto eltwise_prim = primitive->value_as_Eltwise();
  if (eltwise_prim == nullptr) {
    MS_LOG(ERROR) << "eltwise_prim is nullptr";
    return nullptr;
  }
  ArithmeticParameter *param = PopulateArithmeticV0CommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticV0CommonPara failed.";
    return nullptr;
  }

  param->eltwise_mode_ = eltwise_prim->mode();
  param->op_parameter_.type_ = schema::PrimitiveType_Eltwise;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

Registry g_eltwiseV0ParameterRegistry(schema::v0::PrimitiveType_Eltwise, PopulateEltwiseParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
