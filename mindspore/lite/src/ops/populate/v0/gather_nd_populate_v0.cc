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
#include "nnacl/fp32/gatherNd_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateGatherNdParameter(const void *prim) {
  GatherNdParameter *gather_nd_param = reinterpret_cast<GatherNdParameter *>(malloc(sizeof(GatherNdParameter)));
  if (gather_nd_param == nullptr) {
    MS_LOG(ERROR) << "malloc GatherNdParameter failed.";
    return nullptr;
  }
  memset(gather_nd_param, 0, sizeof(GatherNdParameter));
  gather_nd_param->op_parameter_.type_ = schema::PrimitiveType_GatherNd;
  return reinterpret_cast<OpParameter *>(gather_nd_param);
}
}  // namespace

Registry g_gatherNdV0ParameterRegistry(schema::v0::PrimitiveType_GatherNd, PopulateGatherNdParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
