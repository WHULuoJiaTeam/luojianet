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
#include "nnacl/base/broadcast_to.h"
using mindspore::schema::PrimitiveType_BroadcastTo;

namespace mindspore {
namespace lite {
OpParameter *PopulateBroadcastToParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_BroadcastTo();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<BroadcastToParameter *>(malloc(sizeof(BroadcastToParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc BroadcastToParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(BroadcastToParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto dst_shape = value->shape();
  if (dst_shape == nullptr) {
    MS_LOG(INFO) << "broadcast_to has not shape const tensor.";
  } else {
    param->shape_size_ = dst_shape->size();
    if (param->shape_size_ > MAX_SHAPE_SIZE) {
      MS_LOG(ERROR) << "Invalid shape size: " << param->shape_size_;
      free(param);
      return nullptr;
    }
    for (size_t i = 0; i < param->shape_size_; ++i) {
      param->shape_[i] = dst_shape->Get(i);
    }
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_BroadcastTo, PopulateBroadcastToParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
