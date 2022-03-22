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

#include "tensor_serializer.h"
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_util.h"
#include "graph/debug/ge_log.h"
#include "tensor_desc_serializer.h"
#include "graph/ge_tensor.h"

namespace ge {
graphStatus GeTensorSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  GeTensor ge_tensor;
  const graphStatus ret = av.GetValue(ge_tensor);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get tensor attr.");
    return GRAPH_FAILED;
  }

  GeTensorSerializeUtils::GeTensorAsProto(ge_tensor, def.mutable_t());
  return GRAPH_SUCCESS;
}

graphStatus GeTensorSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  GeTensor ge_tensor;
  GeTensorSerializeUtils::AssembleGeTensorFromProto(&def.t(), ge_tensor);
  return av.SetValue(std::move(ge_tensor));
}

REG_GEIR_SERIALIZER(GeTensorSerializer, GetTypeId<GeTensor>(), proto::AttrDef::kT);
}  // namespace ge
