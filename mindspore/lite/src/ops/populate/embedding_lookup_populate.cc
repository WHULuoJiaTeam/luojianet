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
#include "nnacl/fp32/embedding_lookup_fp32.h"
using mindspore::schema::PrimitiveType_EmbeddingLookupFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateEmbeddingLookupParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_EmbeddingLookupFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<EmbeddingLookupParameter *>(malloc(sizeof(EmbeddingLookupParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc EmbeddingLookupParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(EmbeddingLookupParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->max_norm_ = value->max_norm();
  if (param->max_norm_ < 0) {
    MS_LOG(ERROR) << "Embedding lookup max norm should be positive number, got " << param->max_norm_;
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_EmbeddingLookupFusion, PopulateEmbeddingLookupParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
