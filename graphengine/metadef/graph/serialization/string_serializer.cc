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

#include "string_serializer.h"
#include <string>
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_log.h"

namespace ge {
graphStatus StringSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  std::string value;
  const graphStatus ret = av.GetValue(value);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get string attr.");
    return GRAPH_FAILED;
  }
  def.set_s(value);
  return GRAPH_SUCCESS;
}

graphStatus StringSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  return av.SetValue(def.s());
}

REG_GEIR_SERIALIZER(StringSerializer, GetTypeId<std::string>(), proto::AttrDef::kS);
}  // namespace ge
