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
#include "nnacl/crop_parameter.h"
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore {
namespace lite {
OpParameter *PopulateCropParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_Crop();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<CropParameter *>(malloc(sizeof(CropParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc CropParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(CropParameter));

  auto param_offset = value->offsets();
  if (param_offset == nullptr) {
    MS_LOG(ERROR) << "param_offset is nullptr";
    free(param);
    return nullptr;
  }
  if (param_offset->size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "param offset size(" << param_offset->size() << ") should <= " << COMM_SHAPE_SIZE;
    free(param);
    return nullptr;
  }

  param->op_parameter_.type_ = primitive->value_type();
  param->axis_ = value->axis();
  param->offset_size_ = static_cast<int>(param_offset->size());
  for (size_t i = 0; i < param_offset->size(); ++i) {
    param->offset_[i] = *(param_offset->begin() + i);
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Crop, PopulateCropParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
