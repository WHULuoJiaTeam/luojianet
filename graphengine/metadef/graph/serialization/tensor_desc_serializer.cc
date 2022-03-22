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

#include "tensor_desc_serializer.h"

#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_util.h"
#include "graph/ge_tensor.h"

namespace ge {
graphStatus GeTensorDescSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  GeTensorDesc tensor_desc;
  const graphStatus ret = av.GetValue(tensor_desc);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get tensor_desc attr.");
    return GRAPH_FAILED;
  }
  GeTensorSerializeUtils::GeTensorDescAsProto(tensor_desc, def.mutable_td());
  return GRAPH_SUCCESS;
}

graphStatus GeTensorDescSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  GeTensorDesc tensor_desc;
  const proto::TensorDescriptor &descriptor = def.td();
  GeTensorSerializeUtils::AssembleGeTensorDescFromProto(&descriptor, tensor_desc);
  return av.SetValue(std::move(tensor_desc));
}

REG_GEIR_SERIALIZER(GeTensorDescSerializer, GetTypeId<GeTensorDesc>(), proto::AttrDef::kTd);
}  // namespace ge
