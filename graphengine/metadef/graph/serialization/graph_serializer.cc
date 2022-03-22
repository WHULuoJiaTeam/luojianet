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

#include "graph_serializer.h"
#include "graph/debug/ge_util.h"
#include "graph/debug/ge_log.h"
#include "graph/detail/model_serialize_imp.h"

namespace ge {
graphStatus GraphSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  auto graph = def.mutable_g();
  GE_CHECK_NOTNULL(graph);

  if (av.GetValue(*graph) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "Serialize graph failed");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus GraphSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(def.g());
}

REG_GEIR_SERIALIZER(GraphSerializer, GetTypeId<proto::GraphDef>(), proto::AttrDef::kG);
}  // namespace ge
