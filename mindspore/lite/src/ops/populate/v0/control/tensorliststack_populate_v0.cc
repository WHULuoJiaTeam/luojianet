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
OpParameter *PopulateTensorListStackParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto tensorList_prim = primitive->value_as_TensorListStack();
  MS_CHECK_TRUE_MSG(tensorList_prim != nullptr, nullptr, "value is nullptr");
  auto *stack_param = reinterpret_cast<TensorListParameter *>(malloc(sizeof(TensorListParameter)));
  if (stack_param == nullptr) {
    MS_LOG(ERROR) << "malloc TensorListParameter failed.";
    return nullptr;
  }
  memset(stack_param, 0, sizeof(TensorListParameter));
  stack_param->op_parameter_.type_ = schema::PrimitiveType_TensorListStack;
  stack_param->element_dtype_ = tensorList_prim->elementDType();
  stack_param->num_element_ = tensorList_prim->numElements();
  return reinterpret_cast<OpParameter *>(stack_param);
}
}  // namespace

Registry g_tensorListStackV0ParameterRegistry(schema::v0::PrimitiveType_TensorListStack,
                                              PopulateTensorListStackParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
