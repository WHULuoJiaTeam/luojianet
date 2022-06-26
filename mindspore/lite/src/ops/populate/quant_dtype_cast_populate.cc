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
#include "nnacl/int8/quant_dtype_cast_int8.h"
using mindspore::schema::PrimitiveType_QuantDTypeCast;

namespace mindspore {
namespace lite {
OpParameter *PopulateQuantDTypeCastParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_QuantDTypeCast();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<QuantDTypeCastParameter *>(malloc(sizeof(QuantDTypeCastParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc QuantDTypeCastParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(QuantDTypeCastParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->srcT = value->src_t();
  param->dstT = value->dst_t();
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_QuantDTypeCast, PopulateQuantDTypeCastParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
