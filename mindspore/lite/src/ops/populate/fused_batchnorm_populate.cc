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
#include "nnacl/batchnorm_parameter.h"
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore {
namespace lite {
OpParameter *PopulateFusedBatchNorm(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_FusedBatchNorm();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchNormParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(BatchNormParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->epsilon_ = value->epsilon();
  param->momentum_ = value->momentum();
  param->fused_ = true;
  param->is_training_ = static_cast<bool>(value->mode());
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_FusedBatchNorm, PopulateFusedBatchNorm, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
