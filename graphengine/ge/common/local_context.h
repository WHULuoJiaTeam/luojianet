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

#ifndef GE_GRAPH_COMMON_LOCAL_CONTEXT_H_
#define GE_GRAPH_COMMON_LOCAL_CONTEXT_H_

#include "framework/omg/omg_inner_types.h"

namespace ge {
void SetLocalOmgContext(OmgContext &context);
OmgContext &GetLocalOmgContext();


struct OmeContext {
  bool need_multi_batch = false;
  std::string dynamic_node_type;
  std::vector<NodePtr> data_nodes;
  std::vector<NodePtr> getnext_nosink_nodes;
  std::vector<std::string> dynamic_shape_dims;
  std::vector<std::pair<std::string, std::vector<int64_t>>> user_input_dims;
  std::vector<std::vector<int64_t>> user_real_input_dims;
};

GE_FUNC_VISIBILITY
void SetLocalOmeContext(OmeContext &context);

GE_FUNC_VISIBILITY
OmeContext &GetLocalOmeContext();
}  // namespace ge
#endif  // GE_GRAPH_COMMON_LOCAL_CONTEXT_H_
