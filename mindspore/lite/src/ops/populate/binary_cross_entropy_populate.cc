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
#include "nnacl/fp32_grad/binary_cross_entropy.h"
using mindspore::schema::PrimitiveType_BinaryCrossEntropy;

namespace mindspore {
namespace lite {
OpParameter *PopulateBinaryCrossEntropyParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_BinaryCrossEntropy();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<BinaryCrossEntropyParameter *>(malloc(sizeof(BinaryCrossEntropyParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc BinaryCrossEntropy Parameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(BinaryCrossEntropyParameter));

  param->op_parameter_.type_ = primitive->value_type();
  param->reduction = value->reduction();
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_BinaryCrossEntropy, PopulateBinaryCrossEntropyParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
