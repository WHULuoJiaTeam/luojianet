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
#include "nnacl/fp32/roi_pooling_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateROIPoolingParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto roi_pooling_prim = primitive->value_as_ROIPooling();
  if (roi_pooling_prim == nullptr) {
    MS_LOG(ERROR) << "roi_pooling_prim is nullptr";
    return nullptr;
  }
  auto *roi_pooling_param = reinterpret_cast<ROIPoolingParameter *>(malloc(sizeof(ROIPoolingParameter)));
  if (roi_pooling_param == nullptr) {
    MS_LOG(ERROR) << "malloc ROIPoolingParameter failed.";
    return nullptr;
  }
  memset(roi_pooling_param, 0, sizeof(ROIPoolingParameter));
  roi_pooling_param->op_parameter_.type_ = schema::PrimitiveType_ROIPooling;
  roi_pooling_param->pooledH_ = roi_pooling_prim->pooledH();
  roi_pooling_param->pooledW_ = roi_pooling_prim->pooledW();  // note: origin is pooledH
  roi_pooling_param->scale_ = roi_pooling_prim->scale();
  return reinterpret_cast<OpParameter *>(roi_pooling_param);
}
}  // namespace

Registry g_ROIPoolingV0ParameterRegistry(schema::v0::PrimitiveType_ROIPooling, PopulateROIPoolingParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
