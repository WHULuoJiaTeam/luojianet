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

#ifndef METADEF_CXX_ATTR_SERIALIZER_H
#define METADEF_CXX_ATTR_SERIALIZER_H

#include <google/protobuf/text_format.h>
#include "proto/ge_ir.pb.h"

#include "graph/any_value.h"

namespace ge {
/**
 * 所有的serializer都应该是无状态的、可并发调用的，全局仅构造一份，后续多线程并发调用
 */
class GeIrAttrSerializer {
 public:
  virtual graphStatus Serialize(const AnyValue &av, proto::AttrDef &def) = 0;
  virtual graphStatus Deserialize(const proto::AttrDef &def, AnyValue &av) = 0;
  virtual ~GeIrAttrSerializer() = default;
};
}  // namespace ge

#endif  //METADEF_CXX_ATTR_SERIALIZER_H
