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
#include "nnacl/partial_fusion_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulatePartialParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto partial_prim = primitive->value_as_Partial();
  if (partial_prim == nullptr) {
    MS_LOG(ERROR) << "partial_prim is nullptr";
    return nullptr;
  }
  auto *partial_parameter = reinterpret_cast<PartialParameter *>(malloc(sizeof(PartialParameter)));
  if (partial_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc partial parameter failed.";
    return nullptr;
  }
  memset(reinterpret_cast<void *>(partial_parameter), 0, sizeof(PartialParameter));
  partial_parameter->op_parameter_.type_ = schema::PrimitiveType_PartialFusion;

  partial_parameter->sub_graph_index_ = partial_prim->subGraphIndex();

  return reinterpret_cast<OpParameter *>(partial_parameter);
}
}  // namespace

Registry g_partialV0ParameterRegistry(schema::v0::PrimitiveType_Partial, PopulatePartialParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
