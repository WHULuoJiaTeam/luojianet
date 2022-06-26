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
#include "nnacl/l2_norm_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateL2NormParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto l2_norm_prim = primitive->value_as_L2Norm();
  if (l2_norm_prim == nullptr) {
    MS_LOG(ERROR) << "l2_norm_prim is nullptr";
    return nullptr;
  }
  auto *l2_norm_parameter = reinterpret_cast<L2NormParameter *>(malloc(sizeof(L2NormParameter)));
  if (l2_norm_parameter == nullptr) {
    MS_LOG(ERROR) << "malloc L2NormParameter failed.";
    return nullptr;
  }
  memset(l2_norm_parameter, 0, sizeof(L2NormParameter));
  l2_norm_parameter->op_parameter_.type_ = schema::PrimitiveType_L2NormalizeFusion;

  auto axis_vec = l2_norm_prim->axis();
  if (axis_vec == nullptr) {
    MS_LOG(ERROR) << "axis_vec is nullptr";
    free(l2_norm_parameter);
    return nullptr;
  }
  l2_norm_parameter->axis_num_ = axis_vec->size();
  if (((size_t)axis_vec->size()) > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "axis_vec size too big， which cannot be bigger than " << MAX_SHAPE_SIZE;
    free(l2_norm_parameter);
    return nullptr;
  }

  for (size_t i = 0; i < axis_vec->size(); i++) {
    l2_norm_parameter->axis_[i] = *(axis_vec->begin() + i);
  }
  if (l2_norm_prim->epsilon() < 1e-6) {
    l2_norm_parameter->epsilon_ = 1e-6;
  } else {
    l2_norm_parameter->epsilon_ = l2_norm_prim->epsilon();
  }
  if (l2_norm_prim->activationType() == static_cast<int>(schema::v0::ActivationType_RELU)) {
    l2_norm_parameter->act_type_ = ActType_Relu;
  } else if (l2_norm_prim->activationType() == static_cast<int>(schema::v0::ActivationType_RELU6)) {
    l2_norm_parameter->act_type_ = ActType_Relu6;
  } else {
    l2_norm_parameter->act_type_ = ActType_No;
  }
  return reinterpret_cast<OpParameter *>(l2_norm_parameter);
}
}  // namespace

Registry g_l2NormV0ParameterRegistry(schema::v0::PrimitiveType_L2Norm, PopulateL2NormParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
