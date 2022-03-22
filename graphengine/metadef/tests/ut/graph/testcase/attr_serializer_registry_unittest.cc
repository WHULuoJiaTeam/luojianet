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
#include <gtest/gtest.h>
#include <memory>
#include <iostream>
#define private public
#include "graph/serialization/attr_serializer_registry.h"
#include "graph/serialization/string_serializer.h"
#undef private

#include "proto/ge_ir.pb.h"
#include <string>
#include <vector>
namespace ge {
class AttrSerializerRegistryUt : public testing::Test {};

TEST_F(AttrSerializerRegistryUt, StringReg) {
  REG_GEIR_SERIALIZER(ge::StringSerializer, GetTypeId<std::string>(), proto::AttrDef::kS);
  GeIrAttrSerializer *serializer = AttrSerializerRegistry::GetInstance().GetSerializer(GetTypeId<std::string>());
  GeIrAttrSerializer *deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto::AttrDef::kS);
  ASSERT_NE(serializer, nullptr);
  ASSERT_NE(deserializer, nullptr);
}


}