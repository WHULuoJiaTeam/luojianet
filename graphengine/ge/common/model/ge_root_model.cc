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

#include "common/model/ge_root_model.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
void GeRootModel::SetSubgraphInstanceNameToModel(string instance_name, GeModelPtr ge_model) {
  subgraph_instance_name_to_model_.insert(std::pair<string, GeModelPtr>(instance_name, ge_model));
}

Status GeRootModel::CheckIsUnknownShape(bool &is_dynamic_shape) {
  if (root_graph_ == nullptr) {
    return FAILED;
  }
  is_dynamic_shape = false;
  (void)AttrUtils::GetBool(root_graph_, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, is_dynamic_shape);
  return SUCCESS;
}
}  // namespace ge