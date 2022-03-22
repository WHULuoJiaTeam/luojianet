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

#include "attr_serializer_registry.h"

#include "graph/serialization/bool_serializer.h"
#include "graph/serialization/buffer_serializer.h"
#include "graph/serialization/data_type_serializer.h"
#include "graph/serialization/float_serializer.h"
#include "graph/serialization/graph_serializer.h"
#include "graph/serialization/int_serializer.h"
#include "graph/serialization/list_list_float_serializer.h"
#include "graph/serialization/list_list_int_serializer.h"
#include "graph/serialization/list_value_serializer.h"
#include "graph/serialization/named_attrs_serializer.h"
#include "graph/serialization/string_serializer.h"
#include "graph/serialization/tensor_desc_serializer.h"
#include "graph/serialization/tensor_serializer.h"

#include "graph/ge_error_codes.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_log.h"

namespace ge {
REG_GEIR_SERIALIZER(BoolSerializer, GetTypeId<bool>(), proto::AttrDef::kB);
REG_GEIR_SERIALIZER(BufferSerializer, GetTypeId<ge::Buffer>(), proto::AttrDef::kBt);
REG_GEIR_SERIALIZER(DataTypeSerializer, GetTypeId<ge::DataType>(), proto::AttrDef::kDt);
REG_GEIR_SERIALIZER(FloatSerializer, GetTypeId<float>(), proto::AttrDef::kF);
REG_GEIR_SERIALIZER(GraphSerializer, GetTypeId<proto::GraphDef>(), proto::AttrDef::kG);
REG_GEIR_SERIALIZER(IntSerializer, GetTypeId<int64_t>(), proto::AttrDef::kI);
REG_GEIR_SERIALIZER(ListListFloatSerializer,
                    GetTypeId<std::vector<std::vector<float>>>(), proto::AttrDef::kListListFloat);
REG_GEIR_SERIALIZER(ListListIntSerializer,
                    GetTypeId<std::vector<std::vector<int64_t>>>(), proto::AttrDef::kListListInt);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<int64_t>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<std::string>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<float>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<bool>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<GeTensorDesc>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<GeTensor>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<Buffer>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<proto::GraphDef>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<ge::NamedAttrs>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(ListValueSerializer, GetTypeId<std::vector<ge::DataType>>(), proto::AttrDef::kList);
REG_GEIR_SERIALIZER(NamedAttrsSerializer, GetTypeId<ge::NamedAttrs>(), proto::AttrDef::kFunc);
REG_GEIR_SERIALIZER(StringSerializer, GetTypeId<std::string>(), proto::AttrDef::kS);
REG_GEIR_SERIALIZER(GeTensorDescSerializer, GetTypeId<GeTensorDesc>(), proto::AttrDef::kTd);
REG_GEIR_SERIALIZER(GeTensorSerializer, GetTypeId<GeTensor>(), proto::AttrDef::kT);

AttrSerializerRegistry &AttrSerializerRegistry::GetInstance() {
  static AttrSerializerRegistry instance;
  return instance;
}

void AttrSerializerRegistry::RegisterGeIrAttrSerializer(const GeIrAttrSerializerBuilder& builder,
                                                        const TypeId obj_type,
                                                        const proto::AttrDef::ValueCase proto_type) {
  const std::lock_guard<std::mutex> lck_guard(mutex_);
  if (serializer_map_.count(obj_type) > 0U) {
    return;
  }
  std::unique_ptr<GeIrAttrSerializer> serializer = builder();
  serializer_map_[obj_type] = serializer.get();
  deserializer_map_[proto_type] = serializer.get();
  serializer_holder_.push_back(std::move(serializer));
}

GeIrAttrSerializer *AttrSerializerRegistry::GetSerializer(const TypeId obj_type) {
  const auto iter = serializer_map_.find(obj_type);
  if (iter == serializer_map_.end()) {
    // print type
    REPORT_INNER_ERROR("E19999", "Serializer for type has not been registered");
    GELOGE(FAILED, "Serializer for type has not been registered");
    return nullptr;
  }
  return iter->second;
}

GeIrAttrSerializer *AttrSerializerRegistry::GetDeserializer(const proto::AttrDef::ValueCase proto_type) {
  const auto iter = deserializer_map_.find(proto_type);
  if (iter == deserializer_map_.end()) {
    REPORT_INNER_ERROR("E19999",
                       "Deserializer for type [%d] has not been registered",
                       static_cast<int32_t>(proto_type));
    GELOGE(FAILED, "Deserializer for type [%d] has not been registered", static_cast<int32_t>(proto_type));
    return nullptr;
  }
  return iter->second;
}

AttrSerializerRegistrar::AttrSerializerRegistrar(const GeIrAttrSerializerBuilder builder,
                                                 const TypeId obj_type,
                                                 const proto::AttrDef::ValueCase proto_type) {
  if (builder == nullptr) {
    GELOGE(FAILED, "SerializerBuilder is nullptr.");
    return;
  }
  AttrSerializerRegistry::GetInstance().RegisterGeIrAttrSerializer(builder, obj_type, proto_type);
}
}