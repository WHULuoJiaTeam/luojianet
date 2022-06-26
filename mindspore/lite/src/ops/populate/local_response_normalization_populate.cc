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
#include "nnacl/fp32/local_response_norm_fp32.h"
using mindspore::schema::PrimitiveType_LRN;

namespace mindspore {
namespace lite {
OpParameter *PopulateLocalResponseNormParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_LRN();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<LocalResponseNormParameter *>(malloc(sizeof(LocalResponseNormParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc LocalResponseNormParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(LocalResponseNormParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->depth_radius_ = value->depth_radius();
  param->bias_ = value->bias();
  param->alpha_ = value->alpha();
  param->beta_ = value->beta();
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_LRN, PopulateLocalResponseNormParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
