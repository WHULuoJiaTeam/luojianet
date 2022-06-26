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
#include "nnacl/slice_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSliceParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto slice_prim = primitive->value_as_Slice();
  if (slice_prim == nullptr) {
    MS_LOG(ERROR) << "slice_prim is nullptr";
    return nullptr;
  }
  auto *slice_param = reinterpret_cast<SliceParameter *>(malloc(sizeof(SliceParameter)));
  if (slice_param == nullptr) {
    MS_LOG(ERROR) << "malloc SliceParameter failed.";
    return nullptr;
  }
  memset(slice_param, 0, sizeof(SliceParameter));

  slice_param->op_parameter_.type_ = schema::PrimitiveType_SliceFusion;
  auto param_axis = slice_prim->axes();
  if (param_axis != nullptr) {
    if (param_axis->size() > MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "slice's attr axes size is too big, which cannot be bigger than " << MAX_SHAPE_SIZE;
      free(slice_param);
      return nullptr;
    }
    for (size_t i = 0; i < param_axis->size(); ++i) {
      slice_param->axis_[i] = static_cast<int32_t>(param_axis->Get(i));
    }
  } else {
    // use default axes
    for (int32_t i = 0; i < DIMENSION_8D; i++) {
      slice_param->axis_[i] = i;
    }
  }

  return reinterpret_cast<OpParameter *>(slice_param);
}
}  // namespace

Registry g_sliceV0ParameterRegistry(schema::v0::PrimitiveType_Slice, PopulateSliceParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
