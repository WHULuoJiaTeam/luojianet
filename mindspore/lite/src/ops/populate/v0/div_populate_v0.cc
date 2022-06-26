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
#include "src/ops/populate/arithmetic_populate.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateDivParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto div_prim = primitive->value_as_Div();
  if (div_prim == nullptr) {
    MS_LOG(ERROR) << "div_prim is nullptr";
    return nullptr;
  }
  auto *param = PopulateArithmeticCommonPara(primitive);
  if (param == nullptr) {
    MS_LOG(ERROR) << "PopulateArithmeticCommonPara failed.";
    return nullptr;
  }
  param->op_parameter_.type_ = schema::PrimitiveType_DivFusion;
  param->activation_type_ = div_prim->activationType();
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

Registry g_divV0ParameterRegistry(schema::v0::PrimitiveType_Div, PopulateDivParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
