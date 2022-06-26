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
#include "nnacl/squeeze_parameter.h"
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore {
namespace lite {
OpParameter *PopulateSqueezeParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_Squeeze();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<SqueezeParameter *>(malloc(sizeof(SqueezeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc SqueezeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(SqueezeParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto axis = value->axis();
  if (axis != nullptr) {
    param->axis_size_ = axis->size();
    if (param->axis_size_ > MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "Invalid axis size " << param->axis_size_;
      free(param);
      return nullptr;
    }
    for (size_t i = 0; i < param->axis_size_; i++) {
      param->axis_[i] = *(axis->begin() + i);
    }
  } else {
    param->axis_size_ = 0;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Squeeze, PopulateSqueezeParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
