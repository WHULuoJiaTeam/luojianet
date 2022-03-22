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

#ifndef METADEF_GRAPH_SERIALIZATION_FLOAT_SERIALIZER_H_
#define METADEF_GRAPH_SERIALIZATION_FLOAT_SERIALIZER_H_

#include "attr_serializer.h"
#include "attr_serializer_registry.h"
namespace ge {
class FloatSerializer : public GeIrAttrSerializer {
 public:
  FloatSerializer() = default;
  graphStatus Serialize(const AnyValue &av, proto::AttrDef &def) override;
  graphStatus Deserialize(const proto::AttrDef &def, AnyValue &av) override;
};
}  // namespace ge

#endif // METADEF_GRAPH_SERIALIZATION_FLOAT_SERIALIZER_H_
