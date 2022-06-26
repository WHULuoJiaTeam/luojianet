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
#include "nnacl/transpose.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateTransposeParameter(const void *prim) {
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr";
    return nullptr;
  }
  auto *transpose_param = reinterpret_cast<TransposeParameter *>(malloc(sizeof(TransposeParameter)));
  if (transpose_param == nullptr) {
    MS_LOG(ERROR) << "malloc TransposeParameter failed.";
    return nullptr;
  }
  memset(transpose_param, 0, sizeof(TransposeParameter));

  transpose_param->op_parameter_.type_ = schema::PrimitiveType_Transpose;
  return reinterpret_cast<OpParameter *>(transpose_param);
}
}  // namespace

Registry g_transposeV0ParameterRegistry(schema::v0::PrimitiveType_Transpose, PopulateTransposeParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
