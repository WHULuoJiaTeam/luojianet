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

#ifndef ACL_MAPPER_PRIMITIVE_ARITHMETIC_MAPPER_H
#define ACL_MAPPER_PRIMITIVE_ARITHMETIC_MAPPER_H

#include "tools/converter/adapter/acl/mapper/primitive_mapper.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/fusion/sub_fusion.h"

namespace luojianet_ms {
namespace lite {
using luojianet_ms::ops::kNameAddFusion;
using luojianet_ms::ops::kNameDivFusion;
using luojianet_ms::ops::kNameMulFusion;
using luojianet_ms::ops::kNamePowFusion;
using luojianet_ms::ops::kNameSubFusion;

class AddFusionMapper : public PrimitiveMapper {
 public:
  AddFusionMapper() : PrimitiveMapper(kNameAddFusion) {}

  ~AddFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class DivFusionMapper : public PrimitiveMapper {
 public:
  DivFusionMapper() : PrimitiveMapper(kNameDivFusion) {}

  ~DivFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class MulFusionMapper : public PrimitiveMapper {
 public:
  MulFusionMapper() : PrimitiveMapper(kNameMulFusion) {}

  ~MulFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class PowFusionMapper : public PrimitiveMapper {
 public:
  PowFusionMapper() : PrimitiveMapper(kNamePowFusion) {}

  ~PowFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};

class SubFusionMapper : public PrimitiveMapper {
 public:
  SubFusionMapper() : PrimitiveMapper(kNameSubFusion) {}

  ~SubFusionMapper() override = default;

  STATUS Mapper(const CNodePtr &cnode) override;
};
}  // namespace lite
}  // namespace luojianet_ms

#endif  // ACL_MAPPER_PRIMITIVE_ARITHMETIC_MAPPER_H
