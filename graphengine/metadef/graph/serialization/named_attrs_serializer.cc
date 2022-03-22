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

#include "named_attrs_serializer.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"

namespace ge {
graphStatus NamedAttrsSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  ge::NamedAttrs named_attrs;
  const graphStatus ret = av.GetValue(named_attrs);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get named attrs.");
    return GRAPH_FAILED;
  }
  auto func = def.mutable_func();

  return Serialize(named_attrs, func);
}

graphStatus NamedAttrsSerializer::Serialize(const ge::NamedAttrs &named_attr, proto::NamedAttrs* proto_attr) {
  GE_CHECK_NOTNULL(proto_attr);
  proto_attr->set_name(named_attr.GetName().c_str());
  auto mutable_attr = proto_attr->mutable_attr();
  GE_CHECK_NOTNULL(mutable_attr);

  auto attrs = AttrUtils::GetAllAttrs(named_attr);
  for (auto attr : attrs) {
    const AnyValue attr_value = attr.second;
    const auto serializer = AttrSerializerRegistry::GetInstance().GetSerializer(attr_value.GetValueTypeId());
    GE_CHECK_NOTNULL(serializer);
    proto::AttrDef attr_def;
    if (serializer->Serialize(attr_value, attr_def) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Attr serialized failed, name:[%s].", attr.first.c_str());
      return FAILED;
    }
    (*mutable_attr)[attr.first] = attr_def;
  }
  return GRAPH_SUCCESS;
}

graphStatus NamedAttrsSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  ge::NamedAttrs value;
  if (Deserialize(def.func(), value) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return av.SetValue(std::move(value));
}

graphStatus NamedAttrsSerializer::Deserialize(const proto::NamedAttrs &proto_attr, ge::NamedAttrs &named_attrs) {
  named_attrs.SetName(proto_attr.name());
  auto proto_attr_map = proto_attr.attr();
  for (auto proto_attr : proto_attr_map) {
    const auto deserializer = AttrSerializerRegistry::GetInstance().GetDeserializer(proto_attr.second.value_case());
    GE_CHECK_NOTNULL(deserializer);
    AnyValue attr_value;
    if (deserializer->Deserialize(proto_attr.second, attr_value) != GRAPH_SUCCESS) {
      GELOGE(FAILED, "Attr deserialized failed, name:[%s].", proto_attr.first.c_str());
      return FAILED;
    }
    if (named_attrs.SetAttr(proto_attr.first, attr_value) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "NamedAttrs [%s] set attr [%s] failed.",
             named_attrs.GetName().c_str(), proto_attr.first.c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

REG_GEIR_SERIALIZER(NamedAttrsSerializer, GetTypeId<ge::NamedAttrs>(), proto::AttrDef::kFunc);
}  // namespace ge
