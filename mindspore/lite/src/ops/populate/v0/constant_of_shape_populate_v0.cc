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
#include "ir/dtype/type_id.h"
#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/constant_of_shape_parameter.h"

namespace mindspore::lite {
namespace {
OpParameter *PopulateConstantOfShapeParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto attr = primitive->value_as_ConstantOfShape();
  MS_CHECK_TRUE_RET(attr != nullptr, nullptr);
  auto value = attr->value();
  MS_CHECK_TRUE_RET(value != nullptr, nullptr);
  if (value->size() == 0 || value->size() > 1) {
    MS_LOG(ERROR) << "The value of constant of shape is empty or more than 1.";
    return nullptr;
  }
  auto *param = reinterpret_cast<ConstantOfShapeParameter *>(malloc(sizeof(ConstantOfShapeParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ConstantOfShapeParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ConstantOfShapeParameter));
  param->op_parameter_.type_ = schema::PrimitiveType_ConstantOfShape;
  param->data_type_ = attr->dataType();
  switch (param->data_type_) {
    case kNumberTypeFloat32:
      param->value_.f32_value_ = value->data()[0];
      break;
    case kNumberTypeInt32:
      param->value_.int32_value_ = static_cast<int32_t>(value->data()[0]);
      break;
    default:
      MS_LOG(ERROR) << "The value of constant of shape is invalid";
      free(param);
      return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

Registry g_constantOfShapeV0ParameterRegistry(schema::v0::PrimitiveType_ConstantOfShape,
                                              PopulateConstantOfShapeParameter, SCHEMA_V0);
}  // namespace mindspore::lite
