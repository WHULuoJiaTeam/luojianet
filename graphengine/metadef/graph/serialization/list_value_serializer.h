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

#ifndef METADEF_GRAPH_SERIALIZATION_LIST_VALUE_SERIALIZER_H_
#define METADEF_GRAPH_SERIALIZATION_LIST_VALUE_SERIALIZER_H_

#include <map>

#include "attr_serializer.h"
#include "attr_serializer_registry.h"
#include "proto/ge_ir.pb.h"
#include "graph/ge_attr_value.h"

namespace ge {
using Serializer = graphStatus (*)(const AnyValue &av, proto::AttrDef &def);
using Deserializer = graphStatus (*)(const proto::AttrDef &def, AnyValue &av);
class ListValueSerializer : public GeIrAttrSerializer {
 public:
  ListValueSerializer() = default;
  graphStatus Serialize(const AnyValue &av, proto::AttrDef &def);
  graphStatus Deserialize(const proto::AttrDef &def, AnyValue &av);

 private:
  static graphStatus SerializeListInt(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListString(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListFloat(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListBool(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListGeTensorDesc(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListGeTensor(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListBuffer(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListGraphDef(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListNamedAttrs(const AnyValue &av, proto::AttrDef &def);
  static graphStatus SerializeListDataType(const AnyValue &av, proto::AttrDef &def);

  static graphStatus DeserializeListInt(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListString(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListFloat(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListBool(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListGeTensorDesc(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListGeTensor(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListBuffer(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListGraphDef(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListNamedAttrs(const proto::AttrDef &def, AnyValue &av);
  static graphStatus DeserializeListDataType(const proto::AttrDef &def, AnyValue &av);

};
}  // namespace ge

#endif // METADEF_GRAPH_SERIALIZATION_LIST_VALUE_SERIALIZER_H_
