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
#include "nnacl/tensorlist_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateTensorListGetItemParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto tensorList_prim = primitive->value_as_TensorListGetItem();
  if (tensorList_prim == nullptr) {
    MS_LOG(ERROR) << "tensorList_prim is nullptr";
    return nullptr;
  }
  auto *getItem_param = reinterpret_cast<TensorListParameter *>(malloc(sizeof(TensorListParameter)));
  if (getItem_param == nullptr) {
    MS_LOG(ERROR) << "malloc TensorListParameter failed.";
    return nullptr;
  }
  memset(getItem_param, 0, sizeof(TensorListParameter));
  getItem_param->op_parameter_.type_ = schema::PrimitiveType_TensorListGetItem;
  getItem_param->element_dtype_ = tensorList_prim->elementDType();
  return reinterpret_cast<OpParameter *>(getItem_param);
}
}  // namespace

Registry g_tensorListGetItemV0ParameterRegistry(schema::v0::PrimitiveType_TensorListGetItem,
                                                PopulateTensorListGetItemParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
