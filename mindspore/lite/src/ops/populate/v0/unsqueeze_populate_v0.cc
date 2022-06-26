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
#include "nnacl/unsqueeze_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateUnsqueezeParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto unsqueeze_prim = primitive->value_as_Unsqueeze();
  if (unsqueeze_prim == nullptr) {
    MS_LOG(ERROR) << "unsqueeze_prim is nullptr";
    return nullptr;
  }
  auto *unsqueeze_param = reinterpret_cast<UnSqueezeParameter *>(malloc(sizeof(UnSqueezeParameter)));
  if (unsqueeze_param == nullptr) {
    MS_LOG(ERROR) << "malloc UnSqueezeParameter failed.";
    return nullptr;
  }
  memset(unsqueeze_param, 0, sizeof(UnSqueezeParameter));
  unsqueeze_param->op_parameter_.type_ = schema::PrimitiveType_Unsqueeze;
  auto flat_axis = unsqueeze_prim->axis();
  if (flat_axis == nullptr) {
    MS_LOG(ERROR) << "flat_axis is nullptr";
    free(unsqueeze_param);
    return nullptr;
  }
  if (flat_axis->size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "unsqueeze's attr axis size is too big, , which cannot be bigger than " << COMM_SHAPE_SIZE;
    free(unsqueeze_param);
    return nullptr;
  }
  unsqueeze_param->num_dim_ = static_cast<int>(flat_axis->size());
  int i = 0;
  for (auto iter = flat_axis->begin(); iter != flat_axis->end(); ++iter) {
    unsqueeze_param->dims_[i++] = *iter;
  }
  return reinterpret_cast<OpParameter *>(unsqueeze_param);
}
}  // namespace

Registry g_unsqueezeV0ParameterRegistry(schema::v0::PrimitiveType_Unsqueeze, PopulateUnsqueezeParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
