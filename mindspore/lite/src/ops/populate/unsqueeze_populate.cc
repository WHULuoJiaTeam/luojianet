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
#include "nnacl/unsqueeze_parameter.h"
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore {
namespace lite {
OpParameter *PopulateUnsqueezeParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto primitive = static_cast<const schema::Primitive *>(prim);
  auto value = primitive->value_as_Unsqueeze();
  MS_CHECK_TRUE_MSG(value != nullptr, nullptr, "value is nullptr");

  auto *param = reinterpret_cast<UnSqueezeParameter *>(malloc(sizeof(UnSqueezeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc UnSqueezeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(UnSqueezeParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto axis = value->axis();
  if (axis == nullptr) {
    MS_LOG(ERROR) << "axis is nullptr";
    free(param);
    return nullptr;
  }
  auto flat_axis = std::vector<int>(axis->begin(), axis->end());
  if (flat_axis.size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Invalid axis size " << flat_axis.size();
    free(param);
    return nullptr;
  }
  param->num_dim_ = flat_axis.size();
  int i = 0;
  for (int &flat_axi : flat_axis) {
    param->dims_[i++] = flat_axi;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Unsqueeze, PopulateUnsqueezeParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
