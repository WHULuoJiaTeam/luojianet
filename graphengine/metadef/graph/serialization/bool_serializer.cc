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

#include "bool_serializer.h"
#include <string>
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_log.h"

namespace ge {
graphStatus BoolSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  bool val;
  const graphStatus ret = av.GetValue(val);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get bool attr.");
    return GRAPH_FAILED;
  }
  def.set_b(val);
  return GRAPH_SUCCESS;
}

graphStatus BoolSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(def.b());
}

REG_GEIR_SERIALIZER(BoolSerializer, GetTypeId<bool>(), proto::AttrDef::kB);
}  // namespace ge
