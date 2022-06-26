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
#include "nnacl/depth_to_space_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateDepthToSpaceParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto depth_to_space_prim = primitive->value_as_DepthToSpace();
  if (depth_to_space_prim == nullptr) {
    MS_LOG(ERROR) << "depth_to_space_prim is nullptr";
    return nullptr;
  }
  auto *depth_space_param = reinterpret_cast<DepthToSpaceParameter *>(malloc(sizeof(DepthToSpaceParameter)));
  if (depth_space_param == nullptr) {
    MS_LOG(ERROR) << "malloc DepthToSpaceParameter failed.";
    return nullptr;
  }
  memset(depth_space_param, 0, sizeof(DepthToSpaceParameter));

  depth_space_param->op_parameter_.type_ = schema::PrimitiveType_DepthToSpace;
  depth_space_param->block_size_ = depth_to_space_prim->blockSize();
  return reinterpret_cast<OpParameter *>(depth_space_param);
}
}  // namespace
Registry g_depthToSpaceV0ParameterRegistry(schema::v0::PrimitiveType_DepthToSpace, PopulateDepthToSpaceParameter,
                                           SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
