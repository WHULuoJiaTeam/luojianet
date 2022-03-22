/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#ifndef ACL_MAPPER_PRIMITIVE_ELTWISE_MAPPER_H
#define ACL_MAPPER_PRIMITIVE_ELTWISE_MAPPER_H

#include "tools/converter/adapter/acl/mapper/primitive_mapper.h"
#include "ops/eltwise.h"

using luojianet_ms::ops::kNameEltwise;

namespace luojianet_ms {
namespace lite {
class EltWiseMapper : public PrimitiveMapper {
 public:
  EltWiseMapper() : PrimitiveMapper(kNameEltwise) {}

  ~EltWiseMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};
}  // namespace lite
}  // namespace luojianet_ms
#endif  // ACL_MAPPER_PRIMITIVE_ELTWISE_MAPPER_H
