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
#include "nnacl/slice_parameter.h"
using mindspore::schema::PrimitiveType_SliceFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateSliceParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_SliceFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<SliceParameter *>(malloc(sizeof(SliceParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc SliceParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(SliceParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto axes = value->axes();
  // if begin is not const input, then axis can not be decided in converter
  if (axes != nullptr) {
    if (axes->size() > DIMENSION_8D) {
      MS_LOG(ERROR) << "Invalid axes size: " << axes->size();
      free(param);
      return nullptr;
    }
    for (size_t i = 0; i < axes->size(); ++i) {
      param->axis_[i] = axes->Get(i);
    }
  } else {
    // use default axes
    for (int32_t i = 0; i < DIMENSION_8D; i++) {
      param->axis_[i] = i;
    }
  }
  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_SliceFusion, PopulateSliceParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
