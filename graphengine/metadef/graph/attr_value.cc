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

#include "external/graph/attr_value.h"
#include "debug/ge_util.h"
#include "graph/ge_attr_value.h"

#define ATTR_VALUE_SET_GET_IMP(type)                 \
  graphStatus AttrValue::GetValue(type &val) const { \
    if (impl != nullptr) {                           \
      return impl->geAttrValue_.GetValue<type>(val); \
    }                                                \
    return GRAPH_FAILED;                             \
  }

namespace ge {
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AttrValue::AttrValue() {
  impl = ComGraphMakeShared<AttrValueImpl>();
}

ATTR_VALUE_SET_GET_IMP(AttrValue::STR)
ATTR_VALUE_SET_GET_IMP(AttrValue::INT)
ATTR_VALUE_SET_GET_IMP(AttrValue::FLOAT)

graphStatus AttrValue::GetValue(AscendString &val) {
  std::string val_get;
  const auto status = GetValue(val_get);
  if (status != GRAPH_SUCCESS) {
    return status;
  }
  val = AscendString(val_get.c_str());
  return GRAPH_SUCCESS;
}
}  // namespace ge
