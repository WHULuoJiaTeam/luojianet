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
#ifndef ATTR_OPTIONS_H_
#define ATTR_OPTIONS_H_

#include <string>
#include "graph/compute_graph.h"
#include "graph/ge_error_codes.h"

namespace ge {
bool IsOriginalOpFind(OpDescPtr &op_desc, const std::string &op_name);
bool IsOpTypeEqual(const ge::NodePtr &node, const std::string &op_type);
bool IsContainOpType(const std::string &cfg_line, std::string &op_type);
graphStatus KeepDtypeFunc(ComputeGraphPtr &graph, const std::string &cfg_path);
graphStatus WeightCompressFunc(ComputeGraphPtr &graph, const std::string &cfg_path);
}  // namespace
#endif // ATTR_OPTIONS_H_