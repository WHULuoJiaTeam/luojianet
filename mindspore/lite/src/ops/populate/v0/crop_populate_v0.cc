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
#include "nnacl/crop_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateCropParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto crop_prim = primitive->value_as_Crop();
  if (crop_prim == nullptr) {
    MS_LOG(ERROR) << "crop_prim is nullptr";
    return nullptr;
  }
  auto param_offset = crop_prim->offsets();
  if (param_offset == nullptr) {
    MS_LOG(ERROR) << "param_offset is nullptr";
    return nullptr;
  }
  if (param_offset->size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "crop_param offset size(" << param_offset->size() << ") should <= " << COMM_SHAPE_SIZE;
    return nullptr;
  }
  auto *crop_param = reinterpret_cast<CropParameter *>(malloc(sizeof(CropParameter)));
  if (crop_param == nullptr) {
    MS_LOG(ERROR) << "malloc CropParameter failed.";
    return nullptr;
  }
  memset(crop_param, 0, sizeof(CropParameter));
  crop_param->op_parameter_.type_ = schema::PrimitiveType_Crop;
  crop_param->axis_ = crop_prim->axis();
  crop_param->offset_size_ = static_cast<int>(param_offset->size());
  for (size_t i = 0; i < param_offset->size(); ++i) {
    crop_param->offset_[i] = *(param_offset->begin() + i);
  }
  return reinterpret_cast<OpParameter *>(crop_param);
}
}  // namespace

Registry g_cropV0ParameterRegistry(schema::v0::PrimitiveType_Crop, PopulateCropParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
