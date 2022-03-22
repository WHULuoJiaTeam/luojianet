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
#include "list_list_int_serializer.h"
#include <vector>
#include "graph/debug/ge_util.h"
#include "proto/ge_ir.pb.h"
#include "graph/debug/ge_log.h"

namespace ge {
graphStatus ListListIntSerializer::Serialize(const AnyValue &av, proto::AttrDef &def) {
  std::vector<std::vector<int64_t>> list_list_value;
  const graphStatus ret = av.GetValue(list_list_value);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(FAILED, "Failed to get list_list_int attr.");
    return GRAPH_FAILED;
  }
  const auto mutable_list_list = def.mutable_list_list_int();
  GE_CHECK_NOTNULL(mutable_list_list);
  mutable_list_list->clear_list_list_i();
  for (const auto &list_value : list_list_value) {
    const auto list_i = mutable_list_list->add_list_list_i();
    GE_CHECK_NOTNULL(list_i);
    for (const int64_t val : list_value) {
      list_i->add_list_i(val);
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus ListListIntSerializer::Deserialize(const proto::AttrDef &def, AnyValue &av) {
  std::vector<std::vector<int64_t>> values;
  for (size_t idx = 0U; idx < static_cast<size_t>(def.list_list_int().list_list_i_size()); ++idx) {
    std::vector<int64_t> vec;
    for (size_t i = 0U; i < static_cast<size_t>(def.list_list_int().list_list_i(idx).list_i_size()); ++i) {
      vec.push_back(def.list_list_int().list_list_i(idx).list_i(i));
    }
    values.push_back(vec);
  }

  return av.SetValue(std::move(values));
}

REG_GEIR_SERIALIZER(ListListIntSerializer,
                    GetTypeId<std::vector<std::vector<int64_t>>>(), proto::AttrDef::kListListInt);
}  // namespace ge